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


def load_model_and_tokenizer(model_name: str, device: torch.device = None):
    """
    Load a HuggingFace causal LM and its tokenizer.
    Returns (model, tokenizer) with model in eval mode on the given device.
    """
    if device is None:
        device = get_device()

    logger.info(f"Loading model: {model_name}")
    # Use local_files_only when TRANSFORMERS_OFFLINE is set (compute nodes have no internet)
    local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        local_files_only=local_only,
    )
    model = model.to(device)
    model.eval()

    # Pythia tokenizer doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded on {device}. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


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
