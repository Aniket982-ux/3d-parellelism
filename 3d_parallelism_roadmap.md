# 3D Parallelism Roadmap

## Objective
Build a staged distributed-training version of the current English-to-Italian Transformer that demonstrates tensor parallelism, pipeline parallelism, and data parallelism on top of the existing encoder-decoder architecture. The goal is a portfolio-grade project that is easy to understand, debug, and explain, not a production-scale LLM stack.

## Current Baseline
The repository currently contains a standard Transformer implementation in `model.py`, dataset and masking logic in `dataset.py`, configuration and checkpoint helpers in `config.py`, and a training entry point in `train.py`. The existing code already supports data-parallel training patterns and experiment logging. Supporting files such as `train_wb.py`, `model_tp.py`, and the notebooks are auxiliary or duplicate artifacts.

## Hardware Constraint
The first implementation cycle will be done locally on an NVIDIA RTX 3050 with 4 GB VRAM. That means local validation must stay conservative: small batch sizes, short smoke tests, and single-step forward/backward checks where needed. The full 3D topology is a cloud target, not a local target.

## Target Architecture
The final topology is a 3D parallel layout:
- Tensor parallelism inside each node.
- Pipeline parallelism across two nodes in one cluster.
- Data parallelism across multiple clusters.

The initial target layout is `TP=2, PP=2, DP=2`, but only the smoke-test portions of that design should be exercised locally. The cloud implementation will be the place to run the full topology.

## Implementation Phases

### Phase 1: Baseline and correctness guards
Keep the current Transformer as the reference implementation. Add shape and rank checks so future distributed versions can be compared against a known-good path. This phase is mostly about correctness, not speed.

### Phase 2: Tensor parallelism first
Split the heavy matrix operations inside `MultiHeadAttentionBlock` and `FeedForwardBlock` across two GPUs on one node. Validate that the forward pass and backward pass both complete correctly before adding more complexity.

### Phase 3: Pipeline parallelism second
Split the model into pipeline stages, naturally around the encoder-decoder boundary. Add micro-batching and a schedule that moves activations forward and gradients backward cleanly across nodes.

### Phase 4: Data parallelism last
Replicate the full tensor-parallel plus pipeline-parallel stack across multiple clusters and synchronize gradients only within the data-parallel group. Keep logging, validation, and checkpoint writes on a single rank.

### Phase 5: Config and checkpoint cleanup
Add explicit topology parameters such as `tp_size`, `pp_size`, `dp_size`, and world size. Make checkpoint naming and resume logic compatible with sharded execution.

### Phase 6: Documentation and cleanup
Update the main project documentation to explain the objective, the staged rollout, and the launch assumptions. Decide what to do with duplicate or incomplete files once the new path is proven.

## Repo Map
- `model.py` is the source of truth for the current Transformer architecture.
- `train.py` is the training orchestrator and eventual distributed launch point.
- `dataset.py` handles tokenization, padding, and masks.
- `config.py` defines training and checkpoint settings.
- `train_wb.py` is a duplicate or incomplete training variant.
- `model_tp.py` is currently a duplicate model file.
- The notebooks are supporting artifacts, not the core training path.

## Verification Strategy
- Confirm the current baseline still runs correctly before any topology changes.
- After tensor parallelism, verify a one-node two-GPU run completes a full forward and backward pass.
- After pipeline parallelism, verify activations and gradients cross the stage boundary without deadlock.
- After data parallelism, verify replicas stay synchronized and only rank 0 handles validation, logging, and saving.
- Verify resume-from-checkpoint works in the sharded setup.

## Decisions
- Keep the current encoder-decoder Transformer as the reference architecture.
- Treat the project as a staged prototype for learning and portfolio value, not as a production training system.
- Start with the smallest useful topology and scale only after each phase is individually correct.
- Prefer native PyTorch distributed APIs where possible.
- Use local RTX 3050 testing only for correctness smoke tests, not full-scale execution.

## Notes
This roadmap is intentionally staged so each phase can be validated before moving to the next one. The local machine is for proving correctness. The cloud cluster is for proving the full 3D topology.
