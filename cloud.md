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

1. Create the private network `pt3d-net`.
2. Create the shared drive `pt3d-weights` with enough capacity for checkpoints.
3. Create four GPU nodes and attach each one to the same private network.
4. Name them in order: `pt3d-node0`, `pt3d-node1`, `pt3d-node2`, `pt3d-node3`.
5. Attach or mount the shared drive on every node.

If you want a smaller first cloud test, launch only two nodes and set `DP_SIZE=1`.

## Phase 2: Base setup on every node

Run this on every node.

```bash
sudo apt-get update
sudo apt-get install -y net-tools cifs-utils smbclient git

sudo mkdir -p /mnt/training-data
sudo mount -t cifs //DRIVE_IP/SHARE_NAME /mnt/training-data \
  -o uid=1000,gid=1000,rw,user,username=DRIVE_USERNAME,_netdev

mountpoint -q /mnt/training-data || { echo "Shared drive not mounted"; exit 1; }

git clone <YOUR_REPO_URL>
cd <repo-dir>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

wandb login
```

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