# Distributed 3D Transformer

An English-to-Italian Transformer training project built around native PyTorch
3D parallelism:

- Tensor Parallelism across GPUs inside a node
- Pipeline Parallelism across model stages on different nodes
- Data Parallelism across replicated TP+PP pipelines

The target production topology is `DP=2 x PP=2 x TP=2 = 8 GPUs` across 4
nodes. Local validation has already covered the single-GPU smoke path plus
8-rank CPU simulations for end-to-end 3D routing and communication behavior.

See [cloud.md](cloud.md) for the full cloud deployment runbook.

## Features

- Native PyTorch `DeviceMesh` and tensor parallel APIs
- Encoder/decoder pipeline split with explicit point-to-point stage routing
- Data parallel gradient synchronization on top of the TP+PP graph
- DCP checkpoint save/load for sharded distributed model states
- Environment-driven topology overrides via `TP_SIZE`, `PP_SIZE`, `DP_SIZE`
- Local 1-GPU smoke test for WSL/Linux via `tests/run_linux_smoke.sh`
- Cloud node launch helper via `run_cloud_node.sh`
- Built-in debug hooks for pipeline routing, startup topology, and DP sync

## Architecture

### 3D topology

| Dimension | Size | Role | Communication |
|-----------|------|------|---------------|
| TP | 2 | Shard attention and FFN layers inside each node | All-reduce / collectives |
| PP | 2 | Split model into encoder stage and decoder stage | P2P send/recv |
| DP | 2 | Replicate the full TP+PP stack across clusters | Gradient all-reduce |

### 8-GPU logical map

```text
Cluster 0 (DP=0):                    Cluster 1 (DP=1):
                 TP=0    TP=1                          TP=0    TP=1
PP=0  [ Rank 0, Rank 1 ]  Node 0     PP=0  [ Rank 4, Rank 5 ]  Node 2
PP=1  [ Rank 2, Rank 3 ]  Node 1     PP=1  [ Rank 6, Rank 7 ]  Node 3
```

### Training flow

1. Each rank resolves its `(dp, pp, tp)` coordinate from the device mesh.
2. The full transformer is split by pipeline stage.
3. The local stage is tensor-parallelized when `TP_SIZE > 1`.
4. Data parallel wraps the stage or full model when `DP_SIZE > 1`.
5. Micro-batches flow forward across the PP stages and gradients flow back.
6. Optimizer updates are checkpointed with DCP.

## Critical runtime safeguards

- Startup summary: `train_3d.py` prints backend, world size, resolved topology,
    checkpoint directory, and rendezvous variables on rank 0.
- Pipeline debug logging: set `DEBUG_PP=1` to log every pipeline `send` and
    `recv` at the stage boundary.
- DP sync debug: set `DEBUG_DP_SYNC=1` to print the first-step max gradient diff
    across DP replicas.
- Cloud safety: multi-node runs fail fast if `MODEL_FOLDER` is not an absolute
    shared path.
- NCCL safety: `run_cloud_node.sh` enables `NCCL_ASYNC_ERROR_HANDLING=1` and
    `NCCL_IB_DISABLE=1` by default.

## Repository layout

```text
train_3d.py           Main 3D training entry point
pipeline.py           Pipeline stage split and forward/backward schedule
model.py              Transformer definition
dataset.py            Dataset wrapping and masking
config.py             Hyperparameters, topology, checkpoint naming
tests/run_linux_smoke.sh    Local 1-GPU WSL/Linux smoke test
run_cloud_node.sh     Per-node cloud launch helper
tests/                Local verification and simulation scripts
cloud.md              Cloud setup and launch runbook
```

## Validation status

The following checks have already passed locally:

- `bash tests/run_linux_smoke.sh` on WSL with a single local GPU
- `python tests/simulate_full_training_step.py` for full 8-rank 3D logic emulation
- `USE_LIBUV=0 torchrun --standalone --nproc_per_node=8 tests/verify_3d_robustness.py`
    for TP/PP/DP communication correctness

That means the remaining unverified surface is the actual cloud environment:
multi-node NCCL transport, private networking, rendezvous reachability, and
shared-drive mounting.

## Local usage

### WSL/Linux smoke test

```bash
bash tests/run_linux_smoke.sh
```

### Full local 3D simulation on CPU

```bash
python tests/simulate_full_training_step.py
USE_LIBUV=0 torchrun --standalone --nproc_per_node=8 tests/verify_3d_robustness.py
```

## Dependencies

`requirements.txt` pins the Hugging Face dataset stack that was validated during
local smoke testing:

- `datasets==2.15.0`
- `huggingface-hub==0.23.0`

This avoids the repo-id compatibility break that appeared with newer package
combinations.

## Cloud phase

The configuration and launch helpers are ready for the first cloud attempt.
Use [cloud.md](cloud.md) as the step-by-step sequence for provisioning,
configuring, and launching the real multi-node run.
