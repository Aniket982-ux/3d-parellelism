"""
Native Pipeline and Tensor Parallelism Test.
Uses the actual codebase functions (`parallelize_module`, `split_model_into_stages`)
on 2 ranks to prove the native pipeline splits it correctly.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_transformer
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from pipeline import split_model_into_stages

def apply_tensor_parallelism(module, tp_mesh):
    from model import MultiHeadAttentionBlock, FeedForwardBlock
    
    attn_parallel_plan = {
        "w_q": ColwiseParallel(),
        "w_k": ColwiseParallel(),
        "w_v": ColwiseParallel(),
        "w_o": RowwiseParallel(),
    }
    ffn_parallel_plan = {
        "linear_1": ColwiseParallel(),
        "linear_2": RowwiseParallel(),
    }
    
    for name, sub_mod in module.named_modules():
        if isinstance(sub_mod, MultiHeadAttentionBlock):
            parallelize_module(sub_mod, tp_mesh, attn_parallel_plan)
        elif isinstance(sub_mod, FeedForwardBlock):
            parallelize_module(sub_mod, tp_mesh, ffn_parallel_plan)
            
    return module

def worker(rank, world_size, store_path):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, store=store)
    
    if rank == 0:
        print("\n=========================================================")
        print("    NATIVE PIPELINE & TENSOR PARALLEL SPLIT TEST         ")
        print("=========================================================\n")

    # -----------------------------------------------------------------
    # TEST 1: NATIVE PIPELINE SPLITTING (PP=2)
    # -----------------------------------------------------------------
    if rank == 0:
        print("[TEST 1] Testing Pipeline Split (Stage 0 vs Stage 1)")
    
    device = torch.device("cpu")
    global_model = build_transformer(500, 400, 16, 16, d_model=64, N=2, h=4, d_ff=128)
    
    stage = split_model_into_stages(global_model, pp_rank=rank, device=device)
    
    stage_params = sum(p.numel() for p in stage.parameters())
    print(f"  -> Rank {rank} assigned Stage {rank}. Total parameters: {stage_params:,}")
    
    if rank == 0:
        assert stage_params < sum(p.numel() for p in global_model.parameters()), "Stage 0 should be smaller than global model"
    
    dist.barrier()
    
    # -----------------------------------------------------------------
    # TEST 2: NATIVE TENSOR PARALLEL SPLITTING (TP=2)
    # -----------------------------------------------------------------
    if rank == 0:
        print("\n[TEST 2] Testing Native Tensor Parallel Sharding (TP=2)")
        
    tp_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    
    global_model_for_tp = build_transformer(500, 400, 16, 16, d_model=64, N=2, h=4, d_ff=128)
    total_params_before_tp = sum(p.numel() for p in global_model_for_tp.parameters())
    
    sharded_model = apply_tensor_parallelism(global_model_for_tp, tp_mesh)
    
    def get_local_param_count(model):
        total = 0
        for p in model.parameters():
            if hasattr(p, "to_local"):
                # For DTensors (sharded parameters), count only the local shard size
                total += p.to_local().numel()
            else:
                total += p.numel()
        return total

    total_params_after_tp = get_local_param_count(sharded_model)
    print(f"  -> Rank {rank} sharded model parameters (local): {total_params_after_tp:,} (Global was {total_params_before_tp:,})")
    
    assert total_params_after_tp < total_params_before_tp, "Model should have fewer params per rank after TP sharding!"
    
    if rank == 0:
        print("\n=========================================================")
        print("  SUCCESS: NATIVE CODEBASE FUNCTIONS VERIFIED!           ")
        print("=========================================================\n")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    import tempfile
    
    world_size = 2
    store_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_native_store")
    if os.path.exists(store_file):
        os.remove(store_file)
        
    mp.spawn(worker, args=(world_size, store_file), nprocs=world_size, join=True)
    
    if os.path.exists(store_file):
        os.remove(store_file)
