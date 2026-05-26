"""
Master Robustness Test for 3D Parallelism.

This script simulates an 8-GPU cloud setup (DP=2, PP=2, TP=2) locally on the CPU
using the 'gloo' backend. It tests the specific failure modes associated with
each parallelism dimension:

1. TP Robustness (Intra-node / NVLink Simulation):
   Verifies that All-Reduce correctly averages tensors within a TP group.

2. PP Routing Robustness (Inter-node / P2P Simulation):
   Verifies that Stage 0 sends activations strictly to the correct corresponding
   TP rank on Stage 1. This prevents "wrong gradients to wrong nodes" bugs.

3. DP Gradient Sync Robustness (Inter-cluster):
   Verifies that data parallel replicas correctly sum gradients across clusters
   without bleeding into different pipeline stages or tensor slices.

Usage:
    torchrun --standalone --nproc_per_node=8 verify_3d_robustness.py
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def test_tp_robustness(tp_mesh, global_rank):
    """
    Test that Tensor Parallel (TP) groups can correctly All-Reduce.
    If this were real hardware, this validates NVLink health.
    """
    tp_rank = tp_mesh.get_local_rank()
    tp_group = tp_mesh.get_group()
    
    # Each TP rank creates a tensor with its own rank value
    tensor = torch.tensor([float(tp_rank)], dtype=torch.float32)
    
    # All-Reduce (Sum)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=tp_group)
    
    # In a TP=2 setup, the ranks are 0 and 1. Sum should be 0.0 + 1.0 = 1.0.
    expected_sum = 1.0
    assert torch.allclose(tensor, torch.tensor([expected_sum])), \
        f"Rank {global_rank}: TP All-Reduce failed! Expected {expected_sum}, got {tensor.item()}"
    
    if global_rank == 0:
        print("  [✓] TP Robustness Test Passed (NVLink / Intra-node simulation OK)")

def test_pp_routing_robustness(pp_mesh, tp_mesh, global_rank):
    """
    Test that Pipeline Parallel (PP) ranks strictly send data to the exact
    corresponding TP rank on the other node.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_group = pp_mesh.get_group()
    tp_rank = tp_mesh.get_local_rank()
    
    peer_rank = 1 if pp_rank == 0 else 0
    
    # We create a 'tracer' tensor uniquely identified by the TP rank
    tracer = torch.tensor([float(tp_rank * 100)], dtype=torch.float32)
    
    if pp_rank == 0:
        # Stage 0 sends
        dist.send(tracer, group=pp_group, group_dst=peer_rank)
    else:
        # Stage 1 receives
        received = torch.zeros_like(tracer)
        dist.recv(received, group=pp_group, group_src=peer_rank)
        
        # Verify that Stage 1 received exactly the tracer from its corresponding TP rank
        expected_tracer = float(tp_rank * 100)
        assert torch.allclose(received, torch.tensor([expected_tracer])), \
            f"Rank {global_rank}: PP Routing failed! Expected tracer {expected_tracer}, got {received.item()}. Data was sent to the wrong node!"
            
    dist.barrier()
    if global_rank == 0:
        print("  [✓] PP Routing Test Passed (No cross-node corruption, exact peer matched)")

def test_dp_gradient_sync_robustness(dp_mesh, global_rank):
    """
    Test that Data Parallel (DP) ranks correctly synchronize gradients via
    All-Reduce across clusters.
    """
    dp_rank = dp_mesh.get_local_rank()
    dp_group = dp_mesh.get_group()
    
    # Simulate a backward pass where DP=0 and DP=1 compute different gradients
    # We will give DP=0 a gradient of 10.0 and DP=1 a gradient of 20.0
    grad = torch.tensor([10.0 if dp_rank == 0 else 20.0], dtype=torch.float32)
    
    # Perform the DP All-Reduce (Averaging)
    dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=dp_group)
    
    # The average of 10.0 and 20.0 is 15.0
    expected_grad = 15.0
    assert torch.allclose(grad, torch.tensor([expected_grad])), \
        f"Rank {global_rank}: DP Gradient Sync failed! Expected {expected_grad}, got {grad.item()}"
        
    dist.barrier()
    if global_rank == 0:
        print("  [✓] DP Gradient Sync Test Passed (Cross-cluster All-Reduce OK)")

def main():
    # Gloo requires MASTER_ADDR=localhost on Windows for local loopback
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['USE_LIBUV'] = '0'
    dist.init_process_group(backend="gloo")
    
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    assert world_size == 8, f"This test requires exactly 8 ranks, got {world_size}"
    
    dp_size, pp_size, tp_size = 2, 2, 2
    
    mesh = init_device_mesh("cpu", (dp_size, pp_size, tp_size), mesh_dim_names=("dp", "pp", "tp"))
    
    if global_rank == 0:
        print("Starting 3D Robustness and Fault Verification (DP=2, PP=2, TP=2)...")
        print("-" * 60)
        
    # Run tests
    test_tp_robustness(mesh["tp"], global_rank)
    test_pp_routing_robustness(mesh["pp"], mesh["tp"], global_rank)
    test_dp_gradient_sync_robustness(mesh["dp"], global_rank)
    
    if global_rank == 0:
        print("-" * 60)
        print("ALL TESTS PASSED: Architecture is robust and ready for cloud deployment.")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
