"""
3D Parallel Training Script for the English-to-Italian Transformer.

Implements all three dimensions of parallelism:
  - Tensor Parallelism (TP): Shards attention heads and FFN layers within a node.
  - Pipeline Parallelism (PP): Splits encoder / decoder across nodes.
  - Data Parallelism (DP): Replicates the sharded pipeline across clusters.

Target topology: DP=2 × PP=2 × TP=2  =  8 GPUs total.

Usage (cloud):
    torchrun --nnodes=4 --nproc_per_node=2 train_3d.py

Usage (local smoke test, 1 GPU):
    torchrun --standalone --nproc_per_node=1 train_3d.py
"""

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
from pipeline import (
    split_model_into_stages,
    pipeline_forward_backward,
    _chunk_batch,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Distributed training (3D Parallelism APIs)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
import torch.distributed.checkpoint as dcp

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
import wandb

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Handle the fact that model might be wrapped in DDP
    model_impl = model.module if hasattr(model, 'module') else model

    # Precompute the encoder output and reuse it for every step
    encoder_output = model_impl.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model_impl.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model_impl.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config, dp_size, dp_rank):
    """
    Build train and validation dataloaders.

    BUG FIX (Sampler): The DistributedSampler now receives dp_size and dp_rank
    instead of defaulting to the global world_size.  This ensures that only
    Data-Parallel replicas see different data shards, while Tensor-Parallel
    and Pipeline-Parallel ranks within the same DP group all receive the
    same batch (which is correct — TP ranks process the same data with
    different weight slices, and PP ranks process the same data through
    different model stages).
    """
    # It only has the train split, so we divide it ourselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    # FIX: Use dp_size and dp_rank so only DP replicas get different shards.
    # TP ranks within the same DP group will see the same data (correct).
    # PP ranks within the same DP group will see the same data (correct).
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=DistributedSampler(
            train_ds,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
        ),
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def apply_tensor_parallelism(model, tp_mesh):
    """
    Applies Tensor Parallelism to the model using PyTorch 2.x DeviceMesh.
    We shard the MultiHeadAttention projections and FeedForward linear layers.
    """
    from model import MultiHeadAttentionBlock, FeedForwardBlock
    
    # Define the parallelization plan for the Attention Block
    # W_q, W_k, W_v are split column-wise (output dimension)
    # W_o is split row-wise (input dimension) to aggregate the result
    attn_parallel_plan = {
        "w_q": ColwiseParallel(),
        "w_k": ColwiseParallel(),
        "w_v": ColwiseParallel(),
        "w_o": RowwiseParallel(),
    }
    
    # Define the parallelization plan for the Feed Forward Block
    ffn_parallel_plan = {
        "linear_1": ColwiseParallel(),
        "linear_2": RowwiseParallel(),
    }
    
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttentionBlock):
            parallelize_module(module, tp_mesh, attn_parallel_plan)
        elif isinstance(module, FeedForwardBlock):
            parallelize_module(module, tp_mesh, ffn_parallel_plan)
            
    return model

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Setup distributed training
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    init_process_group(backend=backend)
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    tp_size = config.get('tp_size', 1)
    pp_size = config.get('pp_size', 1)
    dp_size = config.get('dp_size', 1)
    num_microbatches = config.get('num_microbatches', 2)
    
    # Validation for local smoke testing
    if world_size != tp_size * pp_size * dp_size:
        if global_rank == 0:
            print(f"Warning: WORLD_SIZE ({world_size}) != tp({tp_size}) * pp({pp_size}) * dp({dp_size}). Forcing tp=1, pp=1, dp=1 for local testing.")
        tp_size = pp_size = dp_size = 1

    assert pp_size <= 2, f"Pipeline size (pp_size) must be <= 2 since the model naturally splits into exactly 2 stages (Encoder/Decoder). Got {pp_size}."

    # -----------------------------------------------------------------------
    # Create the 3D DeviceMesh
    #
    # The mesh shape is (dp_size, pp_size, tp_size).
    # Example for 8 GPUs: (2, 2, 2) → ranks laid out as:
    #
    #   Cluster 0 (DP=0):          Cluster 1 (DP=1):
    #     Node 0 (PP=0): [0, 1]      Node 2 (PP=0): [4, 5]
    #     Node 1 (PP=1): [2, 3]      Node 3 (PP=1): [6, 7]
    #
    # Within each node, ranks 0&1 (or 4&5, etc.) are TP peers.
    # Across nodes within a cluster, ranks [0,2] (or [4,6]) are PP peers.
    # Across clusters, ranks [0,4] (or [1,5], etc.) are DP peers.
    # -----------------------------------------------------------------------
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(device_type, (dp_size, pp_size, tp_size), mesh_dim_names=("dp", "pp", "tp"))
    tp_mesh = mesh["tp"]
    pp_mesh = mesh["pp"]
    dp_mesh = mesh["dp"]
    
    pp_rank = pp_mesh.get_local_rank()
    pp_group = pp_mesh.get_group()
    dp_rank = dp_mesh.get_local_rank()
    
    # Define the device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    print(
        f"Rank {global_rank} | device={device} | "
        f"TP_rank={tp_mesh.get_local_rank()}, "
        f"PP_rank={pp_rank}, "
        f"DP_rank={dp_rank}"
    )

    # Make sure the weights folder exists (only on master node)
    if global_rank == 0:
        model_folder_path = Path(config['model_folder'])
        if not model_folder_path.is_absolute():
            model_folder_path = Path(".") / f"{config['datasource']}_{config['model_folder']}"
        model_folder_path.mkdir(parents=True, exist_ok=True)

    # FIX (Bug 1): Pass dp_size and dp_rank so only DP replicas split data
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, dp_size, dp_rank)
    
    # Build the full model first (all ranks build the same model for weight init)
    full_model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    # -----------------------------------------------------------------------
    # Pipeline Parallelism (Phase 3)
    # -----------------------------------------------------------------------
    if pp_size > 1:
        # Split the model into this rank's pipeline stage
        stage = split_model_into_stages(full_model, pp_rank, device)
        del full_model  # free the unused half
        
        # Apply Tensor Parallelism to our stage (Phase 2)
        if tp_size > 1:
            stage = apply_tensor_parallelism(stage, tp_mesh)
        
        # Wrap in DDP for the DP dimension (gradient sync across clusters)
        if dp_size > 1:
            stage = DDP(stage, device_ids=[local_rank] if torch.cuda.is_available() else None,
                        process_group=dp_mesh.get_group())
        
        model_for_optim = stage
    else:
        # No pipeline parallelism — standard path
        full_model = full_model.to(device)
        
        # Apply Tensor Parallelism (Phase 2)
        if tp_size > 1:
            full_model = apply_tensor_parallelism(full_model, tp_mesh)
        
        # Wrap in DDP for DP dimension
        model_for_optim = DDP(
            full_model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            process_group=dp_mesh.get_group(),
        )
        stage = None  # signal: not using PP
    
    # Weights & Biases initialization ONLY on master node
    if global_rank == 0:
        wandb.init(project="pytorch-transformer", config=config)
        wandb.define_metric("global_step")
        wandb.define_metric("validation/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="global_step")

    optimizer = torch.optim.Adam(model_for_optim.parameters(), lr=config['lr'], eps=1e-9)

    # -----------------------------------------------------------------------
    # Checkpoint loading (BUG FIX #2)
    #
    # We use torch.distributed.checkpoint (DCP) which is aware of sharded
    # DTensors from Tensor Parallelism.  DCP save/load are COLLECTIVE
    # operations — ALL ranks must call them, not just rank 0.
    # -----------------------------------------------------------------------
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    # Unwrap DDP to get the underlying model/stage for DCP
    raw_model = model_for_optim.module if hasattr(model_for_optim, 'module') else model_for_optim
    
    if model_filename and Path(model_filename).exists():
        print(f'[Rank {global_rank}] Loading checkpoint from {model_filename}')
        # DCP loads sharded state in-place — all ranks participate
        dcp.load({"model": raw_model}, checkpoint_id=model_filename)
        # Load scalar metadata from a separate small file (rank 0 writes this)
        meta_path = f"{model_filename}_meta.pt"
        if Path(meta_path).exists():
            try:
                meta = torch.load(meta_path, map_location="cpu", weights_only=True)
                initial_epoch = meta.get('epoch', 0) + 1
                global_step = meta.get('global_step', 0)
            except Exception as e:
                print(f"[Rank {global_rank}] Warning: Metadata file corrupt or unreadable, starting epoch from 0. Error: {e}")
    else:
        if global_rank == 0:
            print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(initial_epoch, config['num_epochs']):
        # Set epoch for the distributed sampler (required for proper shuffle)
        train_dataloader.sampler.set_epoch(epoch)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_for_optim.train()
        
        # Only show TQDM progress bar on master node
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}", disable=global_rank != 0)
        
        for batch in batch_iterator:
            
            if pp_size > 1:
                # -------------------------------------------------------
                # Pipeline Parallel path
                # -------------------------------------------------------
                # Split the batch into micro-batches
                micro_batches = _chunk_batch(batch, num_microbatches)
                
                # Run the pipeline schedule (all-forward-all-backward)
                total_loss = pipeline_forward_backward(
                    stage=raw_model,
                    pp_rank=pp_rank,
                    pp_group=pp_group,
                    pp_world_size=pp_size,
                    micro_batches=micro_batches,
                    loss_fn=loss_fn,
                    device=device,
                    d_model=config['d_model'],
                    seq_len=config['seq_len'],
                )
                
                # Average loss across micro-batches (only meaningful on last PP stage)
                avg_loss = total_loss / num_microbatches
                
                if pp_rank == pp_size - 1:
                    batch_iterator.set_postfix({"loss": f"{avg_loss:6.3f}"})
                    if global_rank == 0:
                        wandb.log({'train/loss': avg_loss, 'global_step': global_step})
                
            else:
                # -------------------------------------------------------
                # Standard (non-PP) path
                # -------------------------------------------------------
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)

                proj_output = model_for_optim(encoder_input, encoder_mask, decoder_input, decoder_mask)

                label = batch['label'].to(device)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                if global_rank == 0:
                    wandb.log({'train/loss': loss.item(), 'global_step': global_step})

                loss.backward()

            # Update the weights (works for both PP and non-PP paths)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # -------------------------------------------------------------------
        # End of epoch: validation + checkpoint
        # -------------------------------------------------------------------
        
        # Validation only on rank 0 and only when NOT using PP
        # (greedy_decode needs the full model; with PP the model is split)
        if global_rank == 0 and pp_size <= 1:
            run_validation(
                model_for_optim, val_dataloader, tokenizer_src, tokenizer_tgt,
                config['seq_len'], device,
                lambda msg: batch_iterator.write(msg), global_step,
            )

        # FIX (Bug 2): Use DCP for sharded checkpoint saving.
        # ALL ranks participate in DCP save (it's a collective operation).
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        dcp.save({"model": raw_model}, checkpoint_id=model_filename)
        
        # Save scalar metadata separately (only rank 0)
        if global_rank == 0:
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
            }, f"{model_filename}_meta.pt")
            print(f"[Rank 0] Checkpoint saved to {model_filename}")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    destroy_process_group()
