#!/usr/bin/env python
"""
Phase 5: Random baseline comparison.

Initialize Pythia-1B with random weights and run the same PhiID pipeline.
Expected: flat syn-red profile (no inverted-U).

Outputs:
    results/phiid_scores/pythia1b_random_phiid.npz
    results/phiid_scores/pythia1b_random_head_rankings.csv
"""

import os
import sys
import json
import logging
import multiprocessing

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CONFIG
from src.utils import set_seed, get_device, setup_logging, ensure_dirs
from src.prompts import PROMPTS
from src.activation_extraction import HeadActivationExtractor
from src.phiid_computation import compute_all_pairs_phiid
from src.head_ranking import compute_head_scores, compute_syn_red_rank, build_ranking_dataframe

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    set_seed(CONFIG["seed"])
    ensure_dirs(CONFIG["results_dir"])

    rankings_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_random_head_rankings.csv")

    if os.path.exists(rankings_path):
        logger.info("Random baseline rankings already exist. Skipping.")
        return

    device = get_device(CONFIG["device"])
    logger.info(f"Using device: {device}")

    # Initialize random model (same architecture, random weights)
    logger.info("Initializing Pythia-1B with random weights...")
    local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    config = AutoConfig.from_pretrained(CONFIG["model_name"], local_files_only=local_only)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract activations
    logger.info("Extracting activations from random model...")
    extractor = HeadActivationExtractor(model, tokenizer, device)
    all_activations, all_tokens, all_logits = extractor.extract_all_prompts(
        PROMPTS, num_tokens=CONFIG["num_tokens_to_generate"]
    )
    logger.info(f"Random activations shape: {all_activations.shape}")

    # Save random activations
    rand_act_path = os.path.join(CONFIG["results_dir"], "activations", "pythia1b_random_activations.npz")
    np.savez(rand_act_path, activations=all_activations)

    # Free model from GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute PhiID
    phiid_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_random_phiid.npz")

    if os.path.exists(phiid_path):
        logger.info("Random PhiID already computed. Loading...")
        data = np.load(phiid_path)
        sts_matrix = data['sts_matrix']
        rtr_matrix = data['rtr_matrix']
    else:
        num_heads = CONFIG["num_total_heads"]
        max_workers = min(multiprocessing.cpu_count(), 32)

        sts_matrix, rtr_matrix = compute_all_pairs_phiid(
            activations=all_activations,
            num_heads=num_heads,
            tau=CONFIG["phiid_tau"],
            kind=CONFIG["phiid_kind"],
            redundancy=CONFIG["phiid_redundancy"],
            max_workers=max_workers,
            save_path=phiid_path,
            checkpoint_interval=5000,
        )

    # Rank heads
    num_heads = CONFIG["num_total_heads"]
    synergy_per_head, redundancy_per_head = compute_head_scores(
        sts_matrix, rtr_matrix, num_heads
    )
    syn_red_rank, syn_rank, red_rank = compute_syn_red_rank(
        synergy_per_head, redundancy_per_head
    )
    df = build_ranking_dataframe(
        synergy_per_head, redundancy_per_head,
        syn_red_rank, syn_rank, red_rank,
        CONFIG["num_layers"], CONFIG["num_heads_per_layer"],
    )
    df.to_csv(rankings_path, index=False)
    logger.info(f"Saved random head rankings to {rankings_path}")
    logger.info("Phase 5 complete.")


if __name__ == "__main__":
    main()
