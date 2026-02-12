#!/usr/bin/env python
"""
Phase 6: Generate all visualization figures.

Outputs:
    results/figures/phiid_profile.png        (Fig 2c)
    results/figures/synergy_redundancy.png   (Fig 2a)
    results/figures/head_ranking_heatmap.png (Fig 2b)
    results/figures/ablation_curves.png      (Fig 4a)
    results/figures/trained_vs_random.png    (Fig 3a)
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CONFIG
from src.utils import setup_logging, ensure_dirs
from src.visualization import (
    plot_phiid_profile,
    plot_synergy_redundancy_heatmaps,
    plot_head_ranking_heatmap,
    plot_ablation_curves,
    plot_trained_vs_random,
)

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    ensure_dirs(CONFIG["results_dir"])
    fig_dir = os.path.join(CONFIG["results_dir"], "figures")

    # Load trained model results
    rankings_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_head_rankings.csv")
    phiid_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_pairwise_phiid.npz")

    if not os.path.exists(rankings_path):
        logger.error(f"Head rankings not found at {rankings_path}. Run script 03 first.")
        sys.exit(1)

    trained_df = pd.read_csv(rankings_path)

    # Fig 2c: PhiID profile
    plot_phiid_profile(
        trained_df,
        title="Pythia-1B: Syn-Red Score by Layer Depth",
        save_path=os.path.join(fig_dir, "phiid_profile.png"),
    )

    # Fig 2a: Synergy/Redundancy heatmaps
    if os.path.exists(phiid_path):
        data = np.load(phiid_path)
        plot_synergy_redundancy_heatmaps(
            data['sts_matrix'], data['rtr_matrix'],
            title_prefix="Pythia-1B",
            save_path=os.path.join(fig_dir, "synergy_redundancy.png"),
        )

    # Fig 2b: Head ranking heatmap
    plot_head_ranking_heatmap(
        trained_df,
        num_layers=CONFIG["num_layers"],
        num_heads_per_layer=CONFIG["num_heads_per_layer"],
        title="Pythia-1B: Syn-Red Score per Head",
        save_path=os.path.join(fig_dir, "head_ranking_heatmap.png"),
    )

    # Fig 4a: Ablation curves
    ablation_path = os.path.join(CONFIG["results_dir"], "ablation", "pythia1b_ablation.csv")
    if os.path.exists(ablation_path):
        ablation_df = pd.read_csv(ablation_path)
        plot_ablation_curves(
            ablation_df,
            title="Pythia-1B: Ablation — Synergistic vs Random Order",
            save_path=os.path.join(fig_dir, "ablation_curves.png"),
        )
    else:
        logger.warning(f"Ablation results not found at {ablation_path}. Skipping ablation plot.")

    # Fig 3a: Trained vs Random
    random_rankings_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_random_head_rankings.csv")
    if os.path.exists(random_rankings_path):
        random_df = pd.read_csv(random_rankings_path)
        plot_trained_vs_random(
            trained_df, random_df,
            title="Pythia-1B: Trained vs Random Initialization",
            save_path=os.path.join(fig_dir, "trained_vs_random.png"),
        )
    else:
        logger.warning(f"Random baseline rankings not found at {random_rankings_path}. Skipping comparison plot.")

    logger.info(f"All figures saved to {fig_dir}/")
    logger.info("Phase 6 complete.")


if __name__ == "__main__":
    main()
