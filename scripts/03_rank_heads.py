#!/usr/bin/env python
"""
Phase 3: Rank attention heads by synergy-redundancy score.

For each head, average synergy/redundancy across all pairs involving it,
then rank by syn_red_score = synergy_rank - redundancy_rank (normalized).

Outputs:
    results/phiid_scores/pythia1b_head_rankings.csv
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CONFIG
from src.utils import setup_logging, ensure_dirs
from src.head_ranking import compute_head_scores, compute_syn_red_rank, build_ranking_dataframe

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    ensure_dirs(CONFIG["results_dir"])

    save_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_head_rankings.csv")

    if os.path.exists(save_path):
        logger.info("Head rankings already exist. Loading and displaying...")
        df = pd.read_csv(save_path)
        layer_means = df.groupby('layer')['syn_red_score'].mean()
        logger.info("Per-layer average syn_red_score:")
        for layer, score in layer_means.items():
            logger.info(f"  Layer {layer:2d}: {score:.4f}")
        return

    # Load PhiID results
    phiid_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_pairwise_phiid.npz")
    if not os.path.exists(phiid_path):
        logger.error(f"PhiID results not found at {phiid_path}. Run script 02 first.")
        sys.exit(1)

    data = np.load(phiid_path)
    sts_matrix = data['sts_matrix']
    rtr_matrix = data['rtr_matrix']

    num_heads = CONFIG["num_total_heads"]
    num_layers = CONFIG["num_layers"]
    num_heads_per_layer = CONFIG["num_heads_per_layer"]

    # Compute per-head scores
    synergy_per_head, redundancy_per_head = compute_head_scores(
        sts_matrix, rtr_matrix, num_heads
    )

    # Compute rankings
    syn_red_rank, syn_rank, red_rank = compute_syn_red_rank(
        synergy_per_head, redundancy_per_head
    )

    # Build DataFrame
    df = build_ranking_dataframe(
        synergy_per_head, redundancy_per_head,
        syn_red_rank, syn_rank, red_rank,
        num_layers, num_heads_per_layer,
    )

    # Save
    df.to_csv(save_path, index=False)
    logger.info(f"Saved head rankings to {save_path}")

    # Print top-10 most synergistic and most redundant
    logger.info("\nTop-10 most synergistic heads (highest syn_red_score):")
    top_syn = df.nlargest(10, 'syn_red_score')
    for _, row in top_syn.iterrows():
        logger.info(f"  Head {int(row['head_idx']):3d} (L{int(row['layer']):2d}H{int(row['head_in_layer']):2d}): "
                    f"score={row['syn_red_score']:.4f}")

    logger.info("\nTop-10 most redundant heads (lowest syn_red_score):")
    top_red = df.nsmallest(10, 'syn_red_score')
    for _, row in top_red.iterrows():
        logger.info(f"  Head {int(row['head_idx']):3d} (L{int(row['layer']):2d}H{int(row['head_in_layer']):2d}): "
                    f"score={row['syn_red_score']:.4f}")

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
