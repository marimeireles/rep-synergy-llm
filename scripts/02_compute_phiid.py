#!/usr/bin/env python
"""
Phase 2: Compute PhiID (Integrated Information Decomposition) for all pairs
of attention heads.

128 heads -> C(128,2) = 8,128 pairs, each averaged over 60 prompts.

Outputs:
    results/phiid_scores/pythia1b_pairwise_phiid.npz
        - sts_matrix: (128, 128) — pairwise synergy
        - rtr_matrix: (128, 128) — pairwise redundancy
"""

import os
import sys
import logging
import multiprocessing

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CONFIG
from src.utils import setup_logging, ensure_dirs
from src.phiid_computation import compute_all_pairs_phiid

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    ensure_dirs(CONFIG["results_dir"])

    save_path = os.path.join(CONFIG["results_dir"], "phiid_scores", "pythia1b_pairwise_phiid.npz")

    if os.path.exists(save_path):
        logger.info("PhiID results already exist. Verifying...")
        data = np.load(save_path)
        sts = data['sts_matrix']
        rtr = data['rtr_matrix']
        logger.info(f"  sts shape: {sts.shape}, rtr shape: {rtr.shape}")
        logger.info(f"  sts range: [{sts.min():.6f}, {sts.max():.6f}]")
        logger.info(f"  rtr range: [{rtr.min():.6f}, {rtr.max():.6f}]")
        logger.info("  Skipping computation.")
        return

    # Load activations
    act_path = os.path.join(CONFIG["results_dir"], "activations", "pythia1b_activations.npz")
    if not os.path.exists(act_path):
        logger.error(f"Activations not found at {act_path}. Run script 01 first.")
        sys.exit(1)

    activations = np.load(act_path)['activations']
    logger.info(f"Loaded activations: shape {activations.shape}")

    num_heads = CONFIG["num_total_heads"]

    # Determine worker count
    max_workers = min(multiprocessing.cpu_count(), 32)
    logger.info(f"Using {max_workers} workers")

    sts_matrix, rtr_matrix = compute_all_pairs_phiid(
        activations=activations,
        num_heads=num_heads,
        tau=CONFIG["phiid_tau"],
        kind=CONFIG["phiid_kind"],
        redundancy=CONFIG["phiid_redundancy"],
        max_workers=max_workers,
        save_path=save_path,
        checkpoint_interval=5000,
    )

    logger.info(f"PhiID computation complete.")
    logger.info(f"  sts range: [{sts_matrix.min():.6f}, {sts_matrix.max():.6f}]")
    logger.info(f"  rtr range: [{rtr_matrix.min():.6f}, {rtr_matrix.max():.6f}]")
    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
