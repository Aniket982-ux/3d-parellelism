import os
import socket
import torch.distributed as dist

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"Detected Local IP: {local_ip}")
    
    os.environ['MASTER_ADDR'] = local_ip
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = local_ip
    os.environ['USE_LIBUV'] = '0'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    try:
        print("Attempting to initialize Gloo process group...")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        print("SUCCESS! Gloo initialized properly on this interface.")
        dist.destroy_process_group()
    except Exception as e:
        print(f"FAILED: {e}")
