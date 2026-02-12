#!/usr/bin/env python
"""
Phase 1: Extract per-head attention activations during autoregressive generation.

For each of 60 prompts, generates 100 tokens and records the L2 norm
of each attention head's output at every generation step.

Outputs:
    results/activations/pythia1b_activations.npz
        - activations: (60, 128, 100)  — 128 = 16 layers * 8 heads
        - tokens: list of 60 lists of 100 token ids
    results/activations/pythia1b_logits.npz
        - logits: (60, 100, vocab_size)
"""

import os
import sys
import json
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CONFIG
from src.utils import set_seed, get_device, load_model_and_tokenizer, setup_logging, ensure_dirs
from src.prompts import PROMPTS
from src.activation_extraction import HeadActivationExtractor

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    set_seed(CONFIG["seed"])
    ensure_dirs(CONFIG["results_dir"])

    act_path = os.path.join(CONFIG["results_dir"], "activations", "pythia1b_activations.npz")
    logits_path = os.path.join(CONFIG["results_dir"], "activations", "pythia1b_logits.npz")

    if os.path.exists(act_path) and os.path.exists(logits_path):
        logger.info("Activations already extracted. Verifying...")
        data = np.load(act_path)
        activations = data['activations']
        logger.info(f"  Shape: {activations.shape}")
        expected = (60, CONFIG["num_total_heads"], 100)
        assert activations.shape == expected, f"Unexpected shape: {activations.shape}, expected {expected}"
        assert np.all(np.isfinite(activations)), "Non-finite values in activations"
        assert np.all(activations >= 0), "Negative L2 norms found"
        logger.info("  Verification passed. Skipping extraction.")
        return

    device = get_device(CONFIG["device"])
    logger.info(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(CONFIG["model_name"], device)
    extractor = HeadActivationExtractor(model, tokenizer, device)

    logger.info(f"Extracting activations for {len(PROMPTS)} prompts, "
                f"{CONFIG['num_tokens_to_generate']} tokens each")

    all_activations, all_tokens, all_logits = extractor.extract_all_prompts(
        PROMPTS, num_tokens=CONFIG["num_tokens_to_generate"]
    )

    # Save activations
    np.savez(act_path, activations=all_activations)
    logger.info(f"Saved activations to {act_path} — shape: {all_activations.shape}")

    # Save logits (large file)
    np.savez_compressed(logits_path, logits=all_logits)
    logger.info(f"Saved logits to {logits_path} — shape: {all_logits.shape}")

    # Save tokens as JSON
    tokens_path = os.path.join(CONFIG["results_dir"], "activations", "pythia1b_tokens.json")
    with open(tokens_path, 'w') as f:
        json.dump(all_tokens, f)
    logger.info(f"Saved tokens to {tokens_path}")

    # Save prompt token IDs (needed by ablation script)
    prompt_token_ids = [tokenizer.encode(p) for p in PROMPTS]
    prompt_ids_path = os.path.join(CONFIG["results_dir"], "activations", "pythia1b_prompt_token_ids.json")
    with open(prompt_ids_path, 'w') as f:
        json.dump(prompt_token_ids, f)
    logger.info(f"Saved prompt token IDs to {prompt_ids_path}")

    # Verification
    expected = (60, CONFIG["num_total_heads"], CONFIG["num_tokens_to_generate"])
    assert all_activations.shape == expected, f"Unexpected shape: {all_activations.shape}, expected {expected}"
    assert np.all(np.isfinite(all_activations)), "Non-finite values!"
    assert np.all(all_activations >= 0), "Negative L2 norms!"

    # Per-layer stats
    num_layers = CONFIG["num_layers"]
    num_heads_per_layer = CONFIG["num_heads_per_layer"]
    for layer in range(num_layers):
        start = layer * num_heads_per_layer
        end = start + num_heads_per_layer
        layer_acts = all_activations[:, start:end, :]
        logger.info(f"Layer {layer:2d}: min={layer_acts.min():.4f}, "
                    f"max={layer_acts.max():.4f}, mean={layer_acts.mean():.4f}")

    logger.info("Phase 1 complete: activation extraction done.")


if __name__ == "__main__":
    main()
