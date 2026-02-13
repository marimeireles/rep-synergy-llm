#!/usr/bin/env python
"""
MATH benchmark perturbation experiments (Fig 4b).

Compares accuracy when perturbing synergistic core, redundant core,
and random subsets of attention heads via Gaussian noise injection.

Usage:
  python scripts/09_math_perturbation.py --model gemma3-4b
  python scripts/09_math_perturbation.py --model qwen3-8b --sigma 0.1 --num-problems 500
  python scripts/09_math_perturbation.py --model gemma3-4b --sigma 0.2 --num-random-seeds 3
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from src.utils import (
    set_seed, get_device, load_model_and_tokenizer,
    setup_logging, ensure_dirs, get_result_path, load_dotenv,
)
from src.model_registry import detect_model_spec
from src.perturbation import GaussianNoisePerturbation
from src.math_eval import evaluate_math_accuracy
from src.visualization import plot_math_perturbation

logger = logging.getLogger(__name__)


def run_perturbation_experiment(
    config, device, hf_token,
    sigma, num_problems, num_random_seeds, core_fraction,
):
    """Run all perturbation conditions and collect results."""
    model_id = config["model_id"]
    results_dir = config["results_dir"]

    # Load head rankings (needed to define synergistic/redundant cores)
    rankings_path = get_result_path(results_dir, "phiid_scores", model_id,
                                    "head_rankings.csv")
    if not os.path.exists(rankings_path):
        logger.error(f"Head rankings not found at {rankings_path}. Run phases 1-3 first.")
        sys.exit(1)

    rankings_df = pd.read_csv(rankings_path)
    total_heads = len(rankings_df)
    num_core = max(1, int(total_heads * core_fraction))

    # Define head subsets
    sorted_by_score = rankings_df.sort_values('syn_red_score', ascending=False)
    synergistic_heads = sorted_by_score['head_idx'].tolist()[:num_core]
    redundant_heads = sorted_by_score['head_idx'].tolist()[-num_core:]

    logger.info(f"Total heads: {total_heads}")
    logger.info(f"Core size (top/bottom {core_fraction*100:.0f}%): {num_core} heads")
    logger.info(f"Synergistic heads (top {num_core}): layers "
                f"{sorted(set(h // rankings_df['head_in_layer'].nunique() for h in synergistic_heads[:10]))}...")
    logger.info(f"Redundant heads (bottom {num_core}): layers "
                f"{sorted(set(h // rankings_df['head_in_layer'].nunique() for h in redundant_heads[:10]))}...")

    all_results = []

    # Define conditions: (name, head_indices, seed)
    conditions = [
        ("baseline", [], 42),
        ("synergistic", synergistic_heads, 42),
        ("redundant", redundant_heads, 42),
    ]
    # Multiple random seeds
    for seed_idx in range(num_random_seeds):
        rng = np.random.RandomState(config["seed"] + seed_idx + 100)
        random_heads = rng.choice(total_heads, size=num_core, replace=False).tolist()
        conditions.append((f"random", random_heads, config["seed"] + seed_idx + 100))

    for cond_name, head_indices, noise_seed in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition: {cond_name} ({len(head_indices)} heads perturbed)")
        logger.info(f"{'='*60}")

        # Load fresh model for each condition (ensures clean weights)
        model, tokenizer = load_model_and_tokenizer(
            config["model_name"], device,
            torch_dtype=config["torch_dtype"],
            hf_token=hf_token,
        )
        spec = detect_model_spec(model)

        # Apply perturbation (skip for baseline)
        if head_indices:
            perturber = GaussianNoisePerturbation(model, spec, sigma=sigma)
            perturber.perturb_heads(head_indices, seed=noise_seed)

        # Evaluate on MATH
        accuracy, eval_results = evaluate_math_accuracy(
            model, tokenizer, device,
            num_problems=num_problems,
            max_new_tokens=512,
        )

        all_results.append({
            "condition": cond_name.split("_seed")[0] if "seed" in cond_name else cond_name,
            "accuracy": accuracy,
            "num_correct": sum(1 for r in eval_results if r["correct"]),
            "num_total": len(eval_results),
            "num_heads_perturbed": len(head_indices),
            "sigma": sigma,
            "seed": noise_seed,
        })

        logger.info(f"  {cond_name}: accuracy = {accuracy:.4f}")

        # Free model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="MATH benchmark perturbation experiments (Fig 4b)"
    )
    parser.add_argument("--model", required=True,
                        choices=["pythia-1b", "gemma3-4b", "qwen3-8b"],
                        help="Model to evaluate")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Noise magnitude as fraction of param std (default: 0.1)")
    parser.add_argument("--num-problems", type=int, default=500,
                        help="Number of MATH problems to evaluate (default: 500)")
    parser.add_argument("--num-random-seeds", type=int, default=3,
                        help="Number of random subset seeds (default: 3)")
    parser.add_argument("--core-fraction", type=float, default=0.25,
                        help="Fraction of heads in syn/red core (default: 0.25 = top/bottom 25%%)")
    args = parser.parse_args()

    setup_logging()
    load_dotenv()

    config = get_config(args.model)
    set_seed(config["seed"])
    ensure_dirs(config["results_dir"])

    hf_token = os.environ.get("HF_TOKEN")
    device = get_device(config["device"])

    logger.info(f"MATH perturbation: model={args.model}, sigma={args.sigma}, "
                f"problems={args.num_problems}, device={device}")

    results = run_perturbation_experiment(
        config, device, hf_token,
        sigma=args.sigma,
        num_problems=args.num_problems,
        num_random_seeds=args.num_random_seeds,
        core_fraction=args.core_fraction,
    )

    # Save results
    model_id = config["model_id"]
    results_dir = config["results_dir"]
    results_df = pd.DataFrame(results)

    csv_path = os.path.join(results_dir, "ablation",
                            f"{model_id}_math_perturbation.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # Generate figure
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{model_id}_math_perturbation.png")

    plot_math_perturbation(
        results_df,
        title=f"{model_id}: MATH Accuracy under Gaussian Noise Perturbation (sigma={args.sigma})",
        save_path=fig_path,
    )

    # Print summary
    logger.info("\n=== Results Summary ===")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['condition']:15s}: accuracy={row['accuracy']:.4f} "
                    f"({int(row['num_correct'])}/{int(row['num_total'])})")

    logger.info("MATH perturbation experiment complete.")


if __name__ == "__main__":
    main()
