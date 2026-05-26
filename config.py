import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc


def get_datasource_slug(config) -> str:
    """Return a filesystem-safe datasource token for checkpoints and logs."""
    return str(config["datasource"]).replace("/", "_")


def get_config():
    return {
        # -----------------------------------------------------------------------
        # Model Hyperparameters
        # -----------------------------------------------------------------------
        "d_model": 512,         # Transformer hidden dimension
        "seq_len": 350,         # Max token sequence length
        "batch_size": _env_int("BATCH_SIZE", 8),        # Per-DP-replica batch size
        "num_epochs": _env_int("NUM_EPOCHS", 20),
        "lr": _env_float("LR", 1e-4),

        # -----------------------------------------------------------------------
        # Dataset & Language
        # -----------------------------------------------------------------------
        "datasource": "Helsinki-NLP/opus_books",
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
        # Local smoke test (single node, mismatched WORLD_SIZE):
        #   train_3d.py currently falls back to 1×1×1.
        #
        # All topology values can be overridden without editing this file:
        #   TP_SIZE, PP_SIZE, DP_SIZE
        # -----------------------------------------------------------------------
        "tp_size": _env_int("TP_SIZE", 2),           # Tensor Parallel: GPUs per node (intra-node)
        "pp_size": _env_int("PP_SIZE", 2),           # Pipeline Parallel: nodes per cluster (Stage 0=Encoder, Stage 1=Decoder)
        "dp_size": _env_int("DP_SIZE", 2),           # Data Parallel: number of clusters

        # Number of micro-batches for the AFAB pipeline schedule.
        # Must evenly divide batch_size. Increase to hide pipeline bubble.
        "num_microbatches": _env_int("NUM_MICROBATCHES", 2),

        # -----------------------------------------------------------------------
        # Checkpointing
        #
        # LOCAL:  leave as "weights"  →  saves to ./Helsinki-NLP_opus_books_weights/
        # CLOUD:  set MODEL_FOLDER=/mnt/training-data/weights or edit this value
        #         so checkpoints live on the shared network drive.
        # -----------------------------------------------------------------------
        "model_folder": os.environ.get("MODEL_FOLDER", "weights"),
        "model_basename": "tmodel_",

        # Preload a checkpoint at startup:
        #   "latest"  →  auto-load the most recent epoch
        #   "00"      →  load a specific epoch (zero-padded string)
        #   None      →  start from scratch
        "preload": os.environ.get("PRELOAD", "latest"),

        # -----------------------------------------------------------------------
        # Tokenizer & Logging
        # -----------------------------------------------------------------------
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": os.environ.get("EXPERIMENT_NAME", "runs/tmodel"),   # TensorBoard run name (rank 0 only)
    }


def get_weights_file_path(config, epoch: str) -> str:
    """Return the DCP checkpoint directory path for a given epoch string."""
    base_folder = Path(config['model_folder'])
    datasource_slug = get_datasource_slug(config)
    if base_folder.is_absolute():
        # E.g. /mnt/training-data/weights -> /mnt/training-data/weights/Helsinki-NLP_opus_books_tmodel_00.pt
        model_folder = base_folder
        model_filename = f"{datasource_slug}_{config['model_basename']}{epoch}.pt"
    else:
        # Keep original behavior for relative paths: e.g. "weights" -> "./Helsinki-NLP_opus_books_weights/tmodel_00.pt"
        model_folder = Path(".") / f"{datasource_slug}_{config['model_folder']}"
        model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)


def latest_weights_file_path(config):
    """Return the path to the most recent checkpoint, or None if none exist."""
    base_folder = Path(config['model_folder'])
    datasource_slug = get_datasource_slug(config)
    if base_folder.is_absolute():
        model_folder = base_folder
        model_filename = f"{datasource_slug}_{config['model_basename']}*"
    else:
        model_folder = Path(".") / f"{datasource_slug}_{config['model_folder']}"
        model_filename = f"{config['model_basename']}*"
    
    if not model_folder.exists():
        return None
    weights_files = list(model_folder.glob(model_filename))
    if not weights_files:
        return None
    weights_files.sort()
    return str(weights_files[-1])
