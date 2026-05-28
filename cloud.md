# Cloud Deployment Runbook

This document is the operational runbook for the real multi-node launch.

The repo is already prepared for cloud use:

- `config.py` accepts environment overrides for topology and paths
- `run_cloud_node.sh` applies the validated cloud defaults
- checkpoint names are filesystem-safe even with `Helsinki-NLP/opus_books`
- the critical debug hooks are already present in `train_3d.py` and `pipeline.py`

## Recommended naming and sequence

Use a predictable naming scheme so host mapping and node rank assignment stay
obvious.

### Recommended resource names

- Private network: `pt3d-net`
- Shared drive: `pt3d-weights`
- Nodes: `pt3d-node0`, `pt3d-node1`, `pt3d-node2`, `pt3d-node3`
- In-host aliases: `node0`, `node1`, `node2`, `node3`

### Fixed role mapping

| Cloud node name | Host alias | Node rank | Logical role |
|-----------------|-----------|-----------|--------------|
| `pt3d-node0` | `node0` | 0 | DP=0, PP=0 |
| `pt3d-node1` | `node1` | 1 | DP=0, PP=1 |
| `pt3d-node2` | `node2` | 2 | DP=1, PP=0 |
| `pt3d-node3` | `node3` | 3 | DP=1, PP=1 |

Keep `node0` stable. It is the rendezvous anchor.

## Phase 1: Create cloud resources

Create all resources in the same region.

### 1.1 — VM type to select

You need a VM with **2 GPUs per node** to exercise Tensor Parallelism at
`TP_SIZE=2`. The validated configuration is:

| Field | Value |
|---|---|
| VM SKU | `NC16as_T4_v3` |
| GPUs per node | 2 × Tesla T4 (16 GB each) |
| vCPUs | 16 |
| RAM | 110 GB |
| OS image | Ubuntu Server 22.04 LTS |
| Pricing | Spot (significantly cheaper; use DCP checkpoints to handle preemption) |
| CUDA driver | 535 series (`nvidia-driver-535`) |
| PyTorch build | `torch==2.2.0+cu121` (requires CUDA driver ≥ 12.1) |

Do **not** use `NC8as_T4_v3` (1 GPU per node) — that forces `TP_SIZE=1` and
does not exercise the intra-node tensor parallelism dimension.

### 1.2 — Disable Secure Boot before creating the VMs

Azure VM images default to **Trusted Launch** with Secure Boot enabled.
NVIDIA DKMS kernel modules are not signed, so they will silently fail to load
after driver installation if Secure Boot is active. The symptom is
`torch.cuda.device_count() = 0` even after a successful driver install.

Disable Secure Boot before creating each VM:
> Azure Portal → Virtual Machine → Configuration → Security type →
> uncheck **Secure Boot** → Save

### 1.3 — Networking and firewall

1. Create a **Virtual Network** with a private subnet (e.g. a `/24` CIDR
   block). All 4 nodes must be in the same subnet.
2. Create a **Network Security Group** and attach it to the subnet or each NIC.
   Add an **inbound rule** allowing TCP on port `48123` (the torchrun
   rendezvous port) from the subnet's own CIDR range. Without this rule,
   `torchrun` will hang at rendezvous waiting for nodes that cannot reach
   each other.
3. Create the private network `pt3d-net`.
4. Create the shared drive `pt3d-weights` (Azure Files, standard tier) with
   enough capacity for checkpoints.
5. Create four GPU nodes and attach each one to the same VNet subnet.
6. Name them in order: `pt3d-node0`, `pt3d-node1`, `pt3d-node2`, `pt3d-node3`.

If you want a smaller first cloud test, launch only two nodes and set `DP_SIZE=1`.

## Phase 2: Base setup on every node

Run this on every node.

### 2.1 — Install system dependencies and NVIDIA driver

```bash
sudo apt-get update
sudo apt-get install -y net-tools cifs-utils smbclient git build-essential

# Install NVIDIA driver — must match CUDA 12.x runtime
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

After reboot, verify the driver loaded correctly:

```bash
nvidia-smi
```

Expected: both GPUs listed with driver version 535.x and CUDA version 12.x.
If `nvidia-smi` shows no devices or errors, check that Secure Boot is disabled
(Phase 1.2). This was the root cause of driver failure on 2 of the 4 nodes
during the initial run.

### 2.2 — Mount shared storage

```bash
sudo mkdir -p /mnt/training-data
sudo mount -t cifs //<STORAGE_ACCOUNT>.file.core.windows.net/<FILE_SHARE> /mnt/training-data \
  -o uid=1000,gid=1000,rw,username=<STORAGE_ACCOUNT>,password=<ACCESS_KEY>,_netdev

# Verify mount
mountpoint -q /mnt/training-data || { echo "Shared drive not mounted"; exit 1; }
```

To make this persistent across reboots, add the same mount line to `/etc/fstab`.
This ensures checkpoints survive VM restarts and preemptions.

### 2.3 — Clone repo and install Python dependencies

```bash
git clone <YOUR_REPO_URL>
cd <repo-dir>

python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with the correct CUDA 12.1 build explicitly
pip install torch==2.2.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

wandb login
```

Installing PyTorch separately before `requirements.txt` ensures pip resolves
the correct `cu121` wheel. If you let pip resolve it from `requirements.txt`
directly it may pull a CPU-only build depending on your pip version.

## Phase 3: Record the values you need from each node

On each node, collect:

```bash
hostname
hostname -I
ip route get 10.0.0.1
nvidia-smi
```

Write down:

- the private IP of each node
- the NIC name from the `dev` field of `ip route get 10.0.0.1`
- confirmation that both GPUs are visible on each node

You will use the same NIC name in `NCCL_SOCKET_IFNAME` on that node.

## Phase 4: Add host mappings on every node

On every node, add all private IP mappings to `/etc/hosts`.

```bash
sudo tee -a /etc/hosts <<EOF
10.x.x.x node0
10.x.x.x node1
10.x.x.x node2
10.x.x.x node3
EOF
```

Why this matters:

- PP traffic is bidirectional
- a missing reverse mapping can look like a silent pipeline hang

## Phase 5: Preflight checks on every node

Run these before the real launch.

```bash
source .venv/bin/activate

python -c "import torch; print(torch.__version__); print(torch.cuda.device_count())"
python -c "import datasets, huggingface_hub; print(datasets.__version__, huggingface_hub.__version__)"

mountpoint -q /mnt/training-data && echo ok
nc -zv node0 48123
```

Expected:

- `torch.cuda.device_count()` should be `2`
- `datasets` should be `2.15.0`
- `huggingface_hub` should be `0.23.0`
- shared drive mount should succeed
- port `48123` should be reachable from peer nodes

## Phase 6: Environment variables used by the launch helper

`run_cloud_node.sh` expects these inputs:

- `NNODES`
- `NODE_RANK`
- `MASTER_ADDR`
- `NCCL_SOCKET_IFNAME`

It defaults the rest to the cloud-safe values already validated in the repo:

```bash
MASTER_PORT=48123
NPROC_PER_NODE=2
TP_SIZE=2
PP_SIZE=2
DP_SIZE=2
NUM_MICROBATCHES=2
MODEL_FOLDER=/mnt/training-data/weights
NCCL_IB_DISABLE=1
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
NCCL_DEBUG=WARN
```

## Phase 7: Full 8-GPU launch sequence

Run these in this order.

### Node 0

```bash
cd <repo-dir>
source .venv/bin/activate
NNODES=4 NODE_RANK=0 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

### Node 1

```bash
cd <repo-dir>
source .venv/bin/activate
NNODES=4 NODE_RANK=1 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

### Node 2

```bash
cd <repo-dir>
source .venv/bin/activate
NNODES=4 NODE_RANK=2 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

### Node 3

```bash
cd <repo-dir>
source .venv/bin/activate
NNODES=4 NODE_RANK=3 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

If you are doing a smaller first test:

```bash
DP_SIZE=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
DP_SIZE=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

## What changes on each node

Only these fields change from node to node:

- `NODE_RANK`
- the local NIC value in `NCCL_SOCKET_IFNAME` if interface names differ
- optionally `MASTER_ADDR` only if you intentionally choose a different node0

Everything else should stay the same across the cluster.

## Recommended first cloud attempt

Use this order:

1. Two-node run with `DP_SIZE=1`.
2. If that passes, do the full four-node run with `DP_SIZE=2`.

That isolates PP+TP first, then adds the DP layer.

## Critical debug knobs already in the code

### Pipeline routing logs

```bash
DEBUG_PP=1 NNODES=4 NODE_RANK=0 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

This logs each stage-boundary `send` and `recv` in `pipeline.py`.

### DP synchronization probe

```bash
DEBUG_DP_SYNC=1 NNODES=4 NODE_RANK=0 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

This prints the first-step max gradient difference across DP replicas.

### NCCL transport debug

```bash
NCCL_DEBUG=INFO NNODES=4 NODE_RANK=0 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

### Disable P2P if hardware transport is suspicious

```bash
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 NNODES=4 NODE_RANK=0 MASTER_ADDR=<NODE0_PRIVATE_IP> NCCL_SOCKET_IFNAME=<NIC> ./run_cloud_node.sh
```

## Most likely remaining failure points

At this stage, the code-side logic has already been validated locally. The next
errors, if any, are most likely to come from environment issues:

1. Wrong private NIC name in `NCCL_SOCKET_IFNAME`
2. Node-to-node port reachability problems on the rendezvous port
3. Shared drive mount missing or mounted differently across nodes
4. Driver or CUDA runtime mismatch on the cloud image
5. NCCL transport issues specific to the cloud provider

## Decision point

Yes, the repo is ready for the cloud phase.

What is left is not more local logic validation. What remains is the real
multi-node environment check, which can only happen on the cloud hardware.

## Monitoring and checkpointing

### Weights & Biases

`train_3d.py` logs training metrics to W&B from global rank 0 only. Every
other rank stays silent to avoid duplicate logging.

Before launching, run `wandb login` on every node and authenticate with your
API key. The run will appear under project `pytorch-transformer` in your W&B
workspace. You will see `train/loss` and `global_step` charts update in real
time as training progresses, along with system-level GPU metrics (memory clock,
SM utilization, power draw).

If you want to disable W&B logging entirely (e.g. for a quick preflight smoke
run), set `WANDB_MODE=disabled` in the environment before launching.

### Distributed Checkpointing (DCP)

At the end of every epoch, `train_3d.py` saves the model using PyTorch's
Distributed Checkpoint API (`dcp.save` with `FileSystemWriter`). Each rank
writes its own shard to the shared Azure Files drive — no single rank holds
the full model in memory during save.

This matters on Spot VMs because Azure can preempt a node at any time. With
sharded checkpoints on shared storage, training can be resumed from the last
completed epoch by any replacement cluster without data loss.

On resume, `dcp.load` with `FileSystemReader` reloads each rank's shard
independently. The `MODEL_FOLDER` environment variable controls where
checkpoints are written and read from. Set it to a path on the shared drive
(e.g. `/mnt/training-data/weights`) so all nodes read and write to the same
location.