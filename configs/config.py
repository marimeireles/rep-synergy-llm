"""
Configuration for single-model and multi-model PhiID pipelines.

Old CONFIG dict is preserved for backward compatibility with scripts 01-06.
New multi-model system uses BASE_CONFIG + MODEL_CONFIGS + get_config().
"""

# --- Legacy single-model config (used by scripts/01-06) ---

CONFIG = {
    "model_name": "EleutherAI/pythia-1b",
    "num_layers": 16,
    "num_heads_per_layer": 8,
    "num_total_heads": 128,  # 16 * 8
    "num_tokens_to_generate": 100,
    "seed": 42,
    "device": "auto",
    "results_dir": "results",
    # Generation: greedy decoding (deterministic, reproducible)
    "do_sample": False,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    # PhiID
    "phiid_tau": 1,
    "phiid_kind": "gaussian",
    "phiid_redundancy": "MMI",
    # Ablation
    "num_random_ablation_seeds": 5,
}

# --- Multi-model config system ---

BASE_CONFIG = {
    "num_tokens_to_generate": 100,
    "seed": 42,
    "device": "auto",
    "results_dir": "results",
    "do_sample": False,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "phiid_tau": 1,
    "phiid_kind": "gaussian",
    "phiid_redundancy": "MMI",
    "num_random_ablation_seeds": 5,
}

MODEL_CONFIGS = {
    "pythia-1b": {
        "model_name": "EleutherAI/pythia-1b",
        "model_id": "pythia1b",
        "torch_dtype": "float32",
    },
    "gemma3-4b": {
        "model_name": "google/gemma-3-4b-pt",
        "model_id": "gemma3_4b",
        "torch_dtype": "float16",
    },
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-8B",
        "model_id": "qwen3_8b",
        "torch_dtype": "float16",
    },
}

# Pythia-1B training checkpoints for Figure 3a
PYTHIA_CHECKPOINTS = [
    "step1",
    "step64",
    "step512",
    "step2000",
    "step8000",
    "step20000",
    "step50000",
    "step100000",
    "step143000",
]


def get_config(model_key: str) -> dict:
    """
    Get merged config for a model.

    Architecture params (num_layers, num_heads, etc.) are NOT included here —
    they are auto-detected from the model via detect_model_spec().

    Args:
        model_key: key into MODEL_CONFIGS (e.g., "pythia-1b", "gemma3-4b")

    Returns:
        merged dict of BASE_CONFIG + MODEL_CONFIGS[model_key]
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    merged = {**BASE_CONFIG, **MODEL_CONFIGS[model_key]}
    return merged
