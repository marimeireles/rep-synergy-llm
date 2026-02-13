"""Utility functions: seeding, device setup, model loading."""

import os
import random
import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility across torch, numpy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_config: str = "auto") -> torch.device:
    """Return the appropriate torch device."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def load_model_and_tokenizer(model_name: str, device: torch.device = None,
                              revision: str = None, torch_dtype: str = "float32",
                              hf_token: str = None):
    """
    Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model ID (e.g., "EleutherAI/pythia-1b")
        device: target device (auto-detected if None)
        revision: model revision/checkpoint (e.g., "step1000" for Pythia)
        torch_dtype: dtype string — "float32", "float16", or "bfloat16"
        hf_token: HuggingFace access token (for gated models like Gemma)

    Returns:
        (model, tokenizer) with model in eval mode on the given device
    """
    if device is None:
        device = get_device()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    pt_dtype = dtype_map.get(torch_dtype, torch.float32)

    rev_str = f" (revision={revision})" if revision else ""
    logger.info(f"Loading model: {model_name}{rev_str} with dtype={torch_dtype}")

    # Use local_files_only when TRANSFORMERS_OFFLINE is set (compute nodes have no internet)
    local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_only,
        token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=pt_dtype,
        local_files_only=local_only,
        token=hf_token,
    )
    model = model.to(device)
    model.eval()

    # Many tokenizers don't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded on {device}. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def get_result_path(results_dir: str, subdir: str, model_id: str,
                    suffix: str, revision: str = None) -> str:
    """
    Build a result file path with model_id and optional revision.

    Examples:
        get_result_path("results", "activations", "pythia1b", "activations.npz")
        -> "results/activations/pythia1b_activations.npz"

        get_result_path("results", "activations", "pythia1b", "activations.npz", "step1000")
        -> "results/activations/pythia1b_step1000_activations.npz"
    """
    if revision:
        filename = f"{model_id}_{revision}_{suffix}"
    else:
        filename = f"{model_id}_{suffix}"
    return os.path.join(results_dir, subdir, filename)


def load_dotenv(path: str = ".env"):
    """
    Load variables from a .env file into os.environ.
    Skips blank lines and comments (#). No external dependency needed.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            # Don't overwrite vars already set in the shell
            if key not in os.environ:
                os.environ[key] = value


def setup_logging(level=logging.INFO):
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs(results_dir: str):
    """Create all result subdirectories."""
    for subdir in ["activations", "phiid_scores", "ablation", "figures"]:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
