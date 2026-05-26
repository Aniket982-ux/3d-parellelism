"""
translate.py — Single-GPU inference for the 3D-trained Transformer.

Loads the latest DCP checkpoint (written by train_3d.py) and runs
greedy decoding on an English sentence.

Usage:
    python translate.py "The sky is blue."
    python translate.py 42        # use sentence index from the dataset
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from datasets import load_dataset
from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path
from dataset import BilingualDataset
from model import build_transformer


def translate(sentence: str):
    # -----------------------------------------------------------------------
    # Bootstrap a single-rank process group so DCP can load the checkpoint.
    # We use gloo (CPU-friendly) even when CUDA is available so this works
    # on any machine without an NVLink-capable multi-GPU setup.
    # -----------------------------------------------------------------------
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12399")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = get_config()

    # Load tokenizers (must already exist — built during training)
    tokenizer_src = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_src"])))
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_tgt"])))
    )

    # Build the model architecture (same as training)
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    # -----------------------------------------------------------------------
    # Load the DCP checkpoint written by train_3d.py.
    # DCP merges shards automatically when world_size=1.
    # -----------------------------------------------------------------------
    model_path = latest_weights_file_path(config)
    if model_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Train the model first with train_3d.py."
        )
    print(f"Loading checkpoint: {model_path}")
    dcp.load({"model": model}, checkpoint_id=model_path)

    dist.destroy_process_group()

    # -----------------------------------------------------------------------
    # If sentence is a digit, treat it as a dataset index
    # -----------------------------------------------------------------------
    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        idx = int(sentence)
        ds_raw = load_dataset(
            config["datasource"],
            f"{config['lang_src']}-{config['lang_tgt']}",
            split="all",
        )
        ds = BilingualDataset(
            ds_raw, tokenizer_src, tokenizer_tgt,
            config["lang_src"], config["lang_tgt"], config["seq_len"],
        )
        sentence = ds[idx]["src_text"]
        label = ds[idx]["tgt_text"]
        print(f"{'ID:':<12}{idx}")

    seq_len = config["seq_len"]
    print(f"{'SOURCE:':<12}{sentence}")
    if label:
        print(f"{'TARGET:':<12}{label}")

    # -----------------------------------------------------------------------
    # Greedy decode
    # -----------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source_ids = torch.cat([
            torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
            torch.tensor(
                [tokenizer_src.token_to_id("[PAD]")] * (seq_len - len(source.ids) - 2),
                dtype=torch.int64,
            ),
        ], dim=0).to(device)

        source_mask = (
            (source_ids != tokenizer_src.token_to_id("[PAD]"))
            .unsqueeze(0).unsqueeze(0).int().to(device)
        )

        encoder_output = model.encode(source_ids.unsqueeze(0), source_mask)
        decoder_input = torch.full((1, 1), tokenizer_tgt.token_to_id("[SOS]"),
                                   dtype=torch.int64, device=device)

        print(f"{'PREDICTED:':<12}", end="")
        eos_id = tokenizer_tgt.token_to_id("[EOS]")

        while decoder_input.size(1) < seq_len:
            decoder_mask = torch.triu(
                torch.ones((1, decoder_input.size(1), decoder_input.size(1))),
                diagonal=1,
            ).bool().to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, ~decoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, next_word.unsqueeze(0)], dim=1
            )
            print(tokenizer_tgt.decode([next_word.item()]), end=" ")
            if next_word.item() == eos_id:
                break

    print()
    return tokenizer_tgt.decode(decoder_input[0].tolist())


if __name__ == "__main__":
    translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good student.")