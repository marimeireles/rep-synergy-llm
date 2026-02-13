#!/usr/bin/env python
"""
Multi-model comparison visualization.

Loads results from Pythia-1B, Gemma 3 4B, and Qwen 3 8B,
generates overlaid comparison figures.

Usage:
  python scripts/08_compare_models.py

Outputs:
  results/figures/multi_model_profiles.png   (Fig 2c overlaid)
  results/figures/multi_model_ablation.png   (Fig 4a overlaid)
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config, MODEL_CONFIGS
from src.utils import setup_logging, ensure_dirs, get_result_path
from src.visualization import plot_multi_model_profiles, plot_multi_model_ablation

logger = logging.getLogger(__name__)

# Display name mapping
MODEL_DISPLAY_NAMES = {
    "pythia-1b": "Pythia-1B",
    "gemma3-4b": "Gemma 3 4B",
    "qwen3-8b": "Qwen 3 8B",
}


def main():
    setup_logging()

    results_dir = "results"
    ensure_dirs(results_dir)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Collect rankings from all available models
    model_rankings = {}
    model_ablations = {}

    for model_key in MODEL_CONFIGS:
        config = get_config(model_key)
        model_id = config["model_id"]
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)

        # Rankings (for PhiID profile)
        rankings_path = get_result_path(results_dir, "phiid_scores", model_id, "head_rankings.csv")
        if os.path.exists(rankings_path):
            model_rankings[display_name] = pd.read_csv(rankings_path)
            logger.info(f"Loaded rankings for {display_name}")
        else:
            logger.warning(f"Rankings not found for {display_name} at {rankings_path}")

        # Ablation results
        ablation_path = get_result_path(results_dir, "ablation", model_id, "ablation.csv")
        if os.path.exists(ablation_path):
            model_ablations[display_name] = pd.read_csv(ablation_path)
            logger.info(f"Loaded ablation for {display_name}")
        else:
            logger.warning(f"Ablation not found for {display_name} at {ablation_path}")

    # Generate multi-model PhiID profile comparison
    if len(model_rankings) >= 2:
        plot_multi_model_profiles(
            model_rankings,
            title="Syn-Red Score by Layer Depth — Model Comparison",
            save_path=os.path.join(fig_dir, "multi_model_profiles.png"),
        )
    else:
        logger.warning(f"Need at least 2 models for comparison, found {len(model_rankings)}")

    # Generate multi-model ablation comparison
    if len(model_ablations) >= 2:
        plot_multi_model_ablation(
            model_ablations,
            title="Ablation Curves — Model Comparison",
            save_path=os.path.join(fig_dir, "multi_model_ablation.png"),
        )
    else:
        logger.warning(f"Need at least 2 models for ablation comparison, found {len(model_ablations)}")

    # Also generate individual model figures if they don't already exist
    from src.visualization import plot_phiid_profile, plot_ablation_curves

    for model_key, display_name in MODEL_DISPLAY_NAMES.items():
        if display_name in model_rankings:
            profile_path = os.path.join(fig_dir, f"{get_config(model_key)['model_id']}_phiid_profile.png")
            if not os.path.exists(profile_path):
                plot_phiid_profile(
                    model_rankings[display_name],
                    title=f"{display_name}: Syn-Red Score by Layer Depth",
                    save_path=profile_path,
                )

        if display_name in model_ablations:
            ablation_fig_path = os.path.join(fig_dir, f"{get_config(model_key)['model_id']}_ablation_curves.png")
            if not os.path.exists(ablation_fig_path):
                plot_ablation_curves(
                    model_ablations[display_name],
                    title=f"{display_name}: Ablation — Synergistic vs Random Order",
                    save_path=ablation_fig_path,
                )

    logger.info(f"All comparison figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
