#!/usr/bin/env python
"""
Graph analysis of synergistic and redundant cores (Fig 3b, 3c).

Loads pairwise PhiID matrices, builds thresholded graphs,
computes graph-theoretic metrics, and generates visualizations.

Usage:
  python scripts/07_graph_analysis.py --model gemma3-4b
  python scripts/07_graph_analysis.py --model qwen3-8b
  python scripts/07_graph_analysis.py --model pythia-1b --top-pct 0.1
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from src.utils import setup_logging, ensure_dirs, get_result_path
from src.graph_analysis import compute_graph_metrics, build_thresholded_graph
from src.visualization import plot_graph_cores

logger = logging.getLogger(__name__)


def _infer_architecture(total_heads, model_id):
    """Infer (num_heads_per_layer, num_layers) from total_heads and model_id."""
    known = {
        "pythia1b": (8, 16),
        "gemma3_4b": (8, 34),
        "qwen3_8b": (32, 36),
    }
    if model_id in known:
        return known[model_id]
    for nhl in [32, 16, 12, 8, 6, 4]:
        if total_heads % nhl == 0:
            return (nhl, total_heads // nhl)
    return (1, total_heads)


def main():
    parser = argparse.ArgumentParser(description="Graph analysis for PhiID cores (Fig 3b, 3c)")
    parser.add_argument("--model", required=True,
                        choices=["pythia-1b", "gemma3-4b", "qwen3-8b"],
                        help="Model to analyze")
    parser.add_argument("--top-pct", type=float, default=0.1,
                        help="Fraction of strongest edges to keep (default: 0.1 = top 10%%)")
    parser.add_argument("--revision", default=None,
                        help="Model revision (e.g., step1000 for Pythia checkpoints)")
    args = parser.parse_args()

    setup_logging()

    config = get_config(args.model)
    model_id = config["model_id"]
    results_dir = config["results_dir"]
    ensure_dirs(results_dir)

    rev_suffix = f"_{args.revision}" if args.revision else ""

    # Load pairwise PhiID matrices
    phiid_path = get_result_path(results_dir, "phiid_scores", model_id,
                                 "pairwise_phiid.npz", args.revision)
    if not os.path.exists(phiid_path):
        logger.error(f"PhiID results not found at {phiid_path}. Run phases 1-2 first.")
        sys.exit(1)

    data = np.load(phiid_path)
    sts_matrix = data['sts_matrix']
    rtr_matrix = data['rtr_matrix']
    num_heads = sts_matrix.shape[0]

    logger.info(f"Loaded PhiID matrices: {num_heads} heads, shape {sts_matrix.shape}")

    num_heads_per_layer, num_layers = _infer_architecture(num_heads, model_id)
    logger.info(f"Architecture: {num_layers} layers x {num_heads_per_layer} heads/layer")

    # --- Fig 3c: Graph metrics ---
    logger.info(f"Computing graph metrics (top {args.top_pct*100:.0f}% edges)...")
    metrics = compute_graph_metrics(sts_matrix, rtr_matrix, top_pct=args.top_pct)

    metrics_path = os.path.join(results_dir, "phiid_scores",
                                f"{model_id}{rev_suffix}_graph_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    logger.info(f"Saved graph metrics to {metrics_path}")

    # Print summary
    logger.info("=== Graph Metrics Summary ===")
    logger.info(f"Synergistic core:")
    logger.info(f"  Edges: {metrics['syn_num_edges']}")
    logger.info(f"  Components: {metrics['syn_num_components']}")
    logger.info(f"  Global efficiency: {metrics['syn_global_efficiency']:.4f}")
    logger.info(f"  Modularity: {metrics['syn_modularity']:.4f}")
    logger.info(f"Redundant core:")
    logger.info(f"  Edges: {metrics['red_num_edges']}")
    logger.info(f"  Components: {metrics['red_num_components']}")
    logger.info(f"  Global efficiency: {metrics['red_global_efficiency']:.4f}")
    logger.info(f"  Modularity: {metrics['red_modularity']:.4f}")

    # --- Fig 3b: Graph visualization ---
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{model_id}{rev_suffix}_graph_cores.png")

    logger.info("Generating graph visualization...")
    plot_graph_cores(
        sts_matrix, rtr_matrix,
        num_layers=num_layers,
        num_heads_per_layer=num_heads_per_layer,
        top_pct=args.top_pct,
        title_prefix=model_id,
        save_path=fig_path,
    )

    logger.info("Graph analysis complete.")


if __name__ == "__main__":
    main()
