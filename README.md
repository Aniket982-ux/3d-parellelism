# Distributed 3D Transformer 🚀

A distributed English-to-Italian Transformer built from scratch in PyTorch, showcasing **3D Parallelism** (Data + Pipeline + Tensor Parallelism) for multi-node GPU cloud training.

---

## Architecture & Topology

The training grid is `DP=2 × PP=2 × TP=2 = 8 GPUs` spread across 4 nodes.

| Dimension | Size | Scope | Communication |
|-----------|------|-------|---------------|
| **Tensor Parallelism (TP=2)** | 2 GPUs/node | Shards Attention heads & FFN layers within a single node | `All-Reduce` via PCIe / NVLink |
| **Pipeline Parallelism (PP=2)** | 2 nodes/cluster | Stage 0 = Encoder, Stage 1 = Decoder | P2P `send`/`recv` across nodes |
| **Data Parallelism (DP=2)** | 2 clusters | Replicates the full TP+PP pipeline | `All-Reduce` gradient sync across clusters |

### 8-GPU Logical Map

```text
Cluster 0 (DP=0):                    Cluster 1 (DP=1):
         TP=0    TP=1                          TP=0    TP=1
PP=0  [ Rank 0, Rank 1 ]  Node 0     PP=0  [ Rank 4, Rank 5 ]  Node 2
PP=1  [ Rank 2, Rank 3 ]  Node 1     PP=1  [ Rank 6, Rank 7 ]  Node 3
```

---

## Project Structure

```text
├── train_3d.py          # Main entry point — 3D parallel training loop
├── model.py             # Encoder-Decoder Transformer (source of truth)
├── pipeline.py          # Pipeline stage wrappers + AFAB schedule + P2P comms
├── dataset.py           # Bilingual dataset, tokenization, masking
├── config.py            # Hyperparameters + 3D topology (tp/pp/dp sizes)
├── translate.py         # Greedy decode / inference utility
├── requirements.txt     # Python dependencies (PyTorch 2.2+)
├── run_local_8gpu.bat   # Local 8-GPU launch helper (Windows)
├── tests/               # Verification & simulation scripts
│   ├── simulate_full_training_step.py
│   ├── simulate_math_only.py
│   ├── test_device_mesh.py
│   ├── test_gloo_ip.py
│   ├── verify_3d_robustness.py
│   ├── verify_all_3d_math_rigorous.py
│   ├── verify_native_split.py
│   └── verify_tp_correctness.py
└── 3d_parallelism_roadmap.md
```

---

## Cloud Deployment (Paperspace)

### Step 1 — Provision Hardware

Create all resources in the **same Paperspace region** (e.g. `East Coast NY2`):

1. **Private Network** — create one and assign every node to it.
2. **4 × P4000x2 GPU Nodes** — use the **ML-in-a-Box** template (CUDA + PyTorch pre-installed).
3. **Network Drive (≥ 250 GB)** — shared filesystem for checkpoints (required by DCP).

> **Downscaled option (4 GPUs / 2 nodes):** Set `dp_size: 1` in `config.py` and use `nnodes=2` instead.

### Step 2 — Configure Every Node

SSH into **each** node and run:

```bash
# 1. Install network utilities
sudo apt-get update && sudo apt-get install -y net-tools cifs-utils smbclient

# 2. Mount the shared network drive
sudo mkdir -p /mnt/training-data
sudo mount -t cifs //DRIVE_IP/SHARE_NAME /mnt/training-data \
    -o uid=1000,gid=1000,rw,user,username=DRIVE_USERNAME
# (enter password when prompted)

# 3. Clone the repo & install dependencies
git clone <YOUR_REPO_URL>
cd pytorch-transformer-distributed
pip install -r requirements.txt

# 4. Login to Weights & Biases (logging)
wandb login
```

### Step 3 — Get Private IPs

On each node run `ifconfig` and note the `10.x.x.x` private IP. Add all four IP→hostname mappings to `/etc/hosts` on the **master node** (Node 0).

### Step 4 — Update Config for Cloud

Edit `config.py` to point checkpoints at the shared mount:

```python
"model_folder": "/mnt/training-data/weights"
```

---

## Running the Training

### Local Smoke Test (single node)

```bash
torchrun --standalone --nproc_per_node=2 train_3d.py
```

The script auto-detects `WORLD_SIZE` and falls back to a `1×1×2` mesh if only 2 GPUs are available.

### Full 8-GPU Cloud Run (4 nodes × 2 GPUs)

Run on **every node**, replacing `MASTER_IP` with Node 0's private IP:

```bash
# Node 0 (master)
torchrun --nproc_per_node=2 --nnodes=4 --node_rank=0 \
    --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:48123 \
    train_3d.py

# Node 1
torchrun --nproc_per_node=2 --nnodes=4 --node_rank=1 \
    --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:48123 \
    train_3d.py

# Node 2
torchrun --nproc_per_node=2 --nnodes=4 --node_rank=2 \
    --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:48123 \
    train_3d.py

# Node 3
torchrun --nproc_per_node=2 --nnodes=4 --node_rank=3 \
    --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:48123 \
    train_3d.py
```

### Downscaled 4-GPU Run (2 nodes × 2 GPUs)

Set `"dp_size": 1` in `config.py`, then:

```bash
# Node 0 (master)
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 \
    --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:48123 \
    train_3d.py

# Node 1
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 \
    --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:48123 \
    train_3d.py
```

---

## Fault Tolerance & Debugging

| Feature | Details |
|---------|---------|
| **Resilient Checkpoints** | Uses `torch.distributed.checkpoint` (DCP) which handles sharded `DTensor` states. On crash, `torchrun` elastic-restarts and DCP resumes seamlessly. |
| **Strict Pipeline Routing** | `pipeline.py` uses validated `peer_rank` math — gradients can never route to the wrong stage. |
| **Pipeline Debug Mode** | Every `dist.send`/`dist.recv` in `pipeline.py` has debug logging gated behind `DEBUG_PP=1`. |

### Debug Commands

```bash
# Enable pipeline P2P logging (shows every send/recv per micro-batch)
DEBUG_PP=1 torchrun ... train_3d.py

# Full NCCL debug output (verbose — logs every collective)
NCCL_DEBUG=INFO torchrun ... train_3d.py

# Disable PCIe P2P to isolate hardware faults
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 torchrun ... train_3d.py
```

### What the Pipeline Debug Logs Look Like

```text
[PP Rank 0 | MB 0] Sending encoder_output to peer 1...
[PP Rank 0 | MB 0] Successfully sent encoder_output.
[PP Rank 1 | MB 0] Waiting to receive encoder_output from peer 0...
[PP Rank 1 | MB 0] Successfully received encoder_output.
```

If training hangs, the **last printed line** tells you exactly which node and which direction (forward/backward) is stuck.

---

## Key Design Decisions

- **Native PyTorch 2.x APIs only** — `DeviceMesh`, `parallelize_module`, DCP. No external frameworks.
- **AFAB pipeline schedule** — All-Forward-All-Backward for simplicity and debuggability.
- **DistributedSampler scoped to DP only** — TP and PP ranks within the same DP group always see identical batches.
- **Rank 0 owns validation & logging** — `wandb` and `run_validation` only execute on the global master.
