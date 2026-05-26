import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def main():
    print("Attempting to initialize a native PyTorch DeviceMesh for Tensor Parallelism...")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    try:
        # PyTorch requires the core process group to be alive before a DeviceMesh can exist
        dist.init_process_group("gloo", rank=0, world_size=1)
        
        # Try to create a 1D DeviceMesh for Tensor Parallelism
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
        print("SUCCESS! We can test parallelize_module locally!")
        dist.destroy_process_group()
    except Exception as e:
        print(f"\n[BLOCKED] PyTorch crashed before DeviceMesh could be created: {e}")

if __name__ == "__main__":
    main()
