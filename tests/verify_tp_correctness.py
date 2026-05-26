import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

# Add project root to path so we can run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_transformer
from train_3d import apply_tensor_parallelism

def run_worker(rank, world_size, store_path):
    # Setup environment
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # Use FileStore to avoid Windows Gloo hostname resolution bugs
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, store=store)
    
    # Create 3D DeviceMesh with DP=1, PP=1, TP=2
    # PyTorch 2.x device_mesh supports cpu backend for easy local debugging without multiple GPUs
    mesh = init_device_mesh("cpu", (1, 1, 2), mesh_dim_names=("dp", "pp", "tp"))
    tp_mesh = mesh["tp"]
    
    # Build a small model for fast test
    d_model = 128
    h = 4
    d_ff = 512
    src_vocab_size = 100
    tgt_vocab_size = 100
    seq_len = 10
    
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=d_model,
        N=1,
        h=h,
        d_ff=d_ff,
        dropout=0.0
    )
    
    # Try applying tensor parallelism
    try:
        model = apply_tensor_parallelism(model, tp_mesh)
        print(f"[Rank {rank}] Tensor parallelism applied successfully!")
    except Exception as e:
        print(f"[Rank {rank}] ERROR during apply_tensor_parallelism:", e, file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(1)
        
    # Create dummy inputs
    src = torch.randint(0, src_vocab_size, (2, seq_len)) # batch_size=2
    tgt = torch.randint(0, tgt_vocab_size, (2, seq_len))
    src_mask = torch.ones((2, 1, 1, seq_len), dtype=torch.bool)
    tgt_mask = torch.ones((2, 1, seq_len, seq_len), dtype=torch.bool)
    
    # Try forward pass
    try:
        output = model(src, src_mask, tgt, tgt_mask)
        print(f"[Rank {rank}] Forward pass successful! Output shape: {output.shape}")
        
        # Try backward pass
        loss = output.sum()
        loss.backward()
        print(f"[Rank {rank}] Backward pass successful!")
    except Exception as e:
        import traceback
        print(f"[Rank {rank}] ERROR during forward/backward pass:", file=sys.stderr)
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)
        
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    store_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tp_store")
    if os.path.exists(store_file):
        os.remove(store_file)
        
    mp.spawn(run_worker, args=(world_size, store_file), nprocs=world_size, join=True)
    
    if os.path.exists(store_file):
        os.remove(store_file)

