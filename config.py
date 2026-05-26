from pathlib import Path


def get_config():
    return {
        # -----------------------------------------------------------------------
        # Model Hyperparameters
        # -----------------------------------------------------------------------
        "d_model": 512,         # Transformer hidden dimension
        "seq_len": 350,         # Max token sequence length
        "batch_size": 8,        # Per-DP-replica batch size
        "num_epochs": 20,
        "lr": 1e-4,

        # -----------------------------------------------------------------------
        # Dataset & Language
        # -----------------------------------------------------------------------
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",

        # -----------------------------------------------------------------------
        # 3D Parallelism Topology
        #
        # Target (cloud, 8 GPUs across 4 nodes):
        #   tp_size=2, pp_size=2, dp_size=2  →  2*2*2 = 8 total ranks
        #
        # Downscaled (cloud, 4 GPUs across 2 nodes):
        #   tp_size=2, pp_size=2, dp_size=1  →  2*2*1 = 4 total ranks
        #
        # Local smoke test (1–2 GPUs, single node):
        #   train_3d.py auto-detects WORLD_SIZE and falls back to 1×1×N mesh.
        # -----------------------------------------------------------------------
        "tp_size": 2,           # Tensor Parallel: GPUs per node (intra-node)
        "pp_size": 2,           # Pipeline Parallel: nodes per cluster (Stage 0=Encoder, Stage 1=Decoder)
        "dp_size": 2,           # Data Parallel: number of clusters

        # Number of micro-batches for the AFAB pipeline schedule.
        # Must evenly divide batch_size. Increase to hide pipeline bubble.
        "num_microbatches": 2,

        # -----------------------------------------------------------------------
        # Checkpointing
        #
        # LOCAL:  leave as "weights"  →  saves to ./opus_books_weights/
        # CLOUD:  change to "/mnt/training-data/weights"  →  saves to shared
        #         network drive (required for multi-node DCP to work correctly)
        # -----------------------------------------------------------------------
        "model_folder": "weights",          # ← change to /mnt/training-data/weights on cloud
        "model_basename": "tmodel_",

        # Preload a checkpoint at startup:
        #   "latest"  →  auto-load the most recent epoch
        #   "00"      →  load a specific epoch (zero-padded string)
        #   None      →  start from scratch
        "preload": "latest",

        # -----------------------------------------------------------------------
        # Tokenizer & Logging
        # -----------------------------------------------------------------------
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",   # TensorBoard run name (rank 0 only)
    }


def get_weights_file_path(config, epoch: str) -> str:
    """Return the DCP checkpoint directory path for a given epoch string."""
    base_folder = Path(config['model_folder'])
    if base_folder.is_absolute():
        # E.g. /mnt/training-data/weights -> /mnt/training-data/weights/opus_books_tmodel_00.pt
        model_folder = base_folder
        model_filename = f"{config['datasource']}_{config['model_basename']}{epoch}.pt"
    else:
        # Keep original behavior for relative paths: e.g. "weights" -> "./opus_books_weights/tmodel_00.pt"
        model_folder = Path(".") / f"{config['datasource']}_{config['model_folder']}"
        model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)


def latest_weights_file_path(config):
    """Return the path to the most recent checkpoint, or None if none exist."""
    base_folder = Path(config['model_folder'])
    if base_folder.is_absolute():
        model_folder = base_folder
        model_filename = f"{config['datasource']}_{config['model_basename']}*"
    else:
        model_folder = Path(".") / f"{config['datasource']}_{config['model_folder']}"
        model_filename = f"{config['model_basename']}*"
    
    if not model_folder.exists():
        return None
    weights_files = list(model_folder.glob(model_filename))
    if not weights_files:
        return None
    weights_files.sort()
    return str(weights_files[-1])
