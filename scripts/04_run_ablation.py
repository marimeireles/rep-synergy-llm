#!/usr/bin/env python
"""
Phase 4: Iterative ablation experiments.

Remove heads one at a time in decreasing syn-red order (most synergistic first)
and measure KL divergence from original model. Compare against random ordering.

Outputs:
    results/ablation/pythia1b_ablation.csv
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CONFIG
from src.utils import set_seed, get_device, load_model_and_tokenizer, setup_logging, ensure_dirs
from src.prompts import PROMPTS
from src.ablation import HeadAblationEngine

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    set_seed(CONFIG["seed"])
    ensure_dirs(CONFIG["results_dir"])

    save_path = os.path.join(CONFIG["results_dir"], "ablation", "pythia1b_ablation.csv")

    if os.path.exists(save_path):
        logger.info("Ablation results already exist. Skipping.")
        df = pd.read_csv(save_path)
        logger.info(f"  {len(df)} rows, order types: {df['order_type'].unique()}")
        return

    # Load prerequisites
    rankings_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_head_rankings.csv")
    logits_path = os.path.join(CONFIG["results_dir"], "activations", "pythia1b_logits.npz")
    tokens_path = os.path.join(CONFIG["results_dir"], "activations", "pythia1b_tokens.json")

    for path, name in [(rankings_path, "head rankings"), (logits_path, "logits"), (tokens_path, "tokens")]:
        if not os.path.exists(path):
            logger.error(f"{name} not found at {path}. Run previous scripts first.")
            sys.exit(1)

    rankings_df = pd.read_csv(rankings_path)
    original_logits = np.load(logits_path)['logits']  # (60, 100, vocab_size)
    with open(tokens_path, 'r') as f:
        original_tokens = json.load(f)

    logger.info(f"Loaded logits shape: {original_logits.shape}")
    logger.info(f"Loaded {len(original_tokens)} token sequences")

    # Load model
    device = get_device(CONFIG["device"])
    model, tokenizer = load_model_and_tokenizer(CONFIG["model_name"], device)

    # Tokenize all prompts
    prompt_token_ids = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        prompt_token_ids.append(ids)

    # Initialize ablation engine
    engine = HeadAblationEngine(model, tokenizer, device)

    num_layers = CONFIG["num_layers"]
    num_heads_per_layer = CONFIG["num_heads_per_layer"]

    # Synergistic order: most synergistic first (highest syn_red_score)
    sorted_heads = rankings_df.sort_values('syn_red_score', ascending=False)
    syn_red_order = sorted_heads['head_idx'].tolist()

    logger.info("Running synergistic-order ablation...")
    syn_results = engine.run_iterative_ablation(
        prompts=PROMPTS,
        prompt_token_ids=prompt_token_ids,
        original_tokens_per_prompt=original_tokens,
        original_logits_per_prompt=original_logits,
        head_order=syn_red_order,
        order_name='syn_red',
        num_layers=num_layers,
        num_heads_per_layer=num_heads_per_layer,
    )

    all_results = syn_results

    # Random orderings
    num_seeds = CONFIG["num_random_ablation_seeds"]
    for seed_idx in range(num_seeds):
        rng = np.random.RandomState(CONFIG["seed"] + seed_idx + 1)
        random_order = list(range(CONFIG["num_total_heads"]))
        rng.shuffle(random_order)

        logger.info(f"Running random ablation (seed {seed_idx + 1}/{num_seeds})...")
        rand_results = engine.run_iterative_ablation(
            prompts=PROMPTS,
            prompt_token_ids=prompt_token_ids,
            original_tokens_per_prompt=original_tokens,
            original_logits_per_prompt=original_logits,
            head_order=random_order,
            order_name=f'random_{seed_idx}',
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
        )
        all_results.extend(rand_results)

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved ablation results to {save_path} ({len(df)} rows)")
    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
