"""
Compute per-head balanced syn-red scores from PhiID data for any model.

Uses the pair_balance method: balance(i,j) = sts(i,j) / (sts(i,j) + rtr(i,j))
Then per-head syn_balance = mean balance across all pairs involving that head.
Min-max normalize to get syn_red_score in [0,1].

Usage:
  python scripts/rank_heads_balanced_generic.py --model gemma3-4b-it
  python scripts/rank_heads_balanced_generic.py --model gemma3-12b-it
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.head_ranking import (
    compute_head_scores,
    compute_pair_balance_scores,
    get_head_layer_mapping,
)

# Architecture lookup: model_id -> (num_heads_per_layer, num_layers)
ARCHITECTURES = {
    "pythia1b": (8, 16),
    "gemma3_4b": (8, 34),
    "gemma3_4b_it": (8, 34),
    "gemma3_12b_it": (8, 26),
    "qwen3_8b": (32, 36),
}

# model CLI key -> model_id prefix for file paths
MODEL_ID_MAP = {
    "pythia-1b": "pythia1b",
    "gemma3-4b": "gemma3_4b",
    "gemma3-4b-it": "gemma3_4b_it",
    "gemma3-12b-it": "gemma3_12b_it",
    "qwen3-8b": "qwen3_8b",
}


def main():
    parser = argparse.ArgumentParser(
        description="Compute balanced head rankings for any model")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_ID_MAP.keys()),
                        help='Model key')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory')
    args = parser.parse_args()

    model_id = MODEL_ID_MAP[args.model]
    num_heads_per_layer, num_layers = ARCHITECTURES[model_id]
    num_heads = num_layers * num_heads_per_layer

    input_file = os.path.join(
        args.results_dir, "phiid_scores", f"{model_id}_pairwise_phiid.npz")
    output_file = os.path.join(
        args.results_dir, "phiid_scores", f"{model_id}_head_rankings_balanced.csv")

    print(f"Model: {args.model} ({model_id})")
    print(f"Architecture: {num_layers} layers x {num_heads_per_layer} heads = {num_heads} total")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Load data
    print(f"\nLoading {input_file}")
    data = np.load(input_file)
    sts_matrix = data["sts_matrix"]
    rtr_matrix = data["rtr_matrix"]
    print(f"  sts_matrix: shape={sts_matrix.shape}, "
          f"range=[{sts_matrix.min():.6f}, {sts_matrix.max():.6f}]")
    print(f"  rtr_matrix: shape={rtr_matrix.shape}, "
          f"range=[{rtr_matrix.min():.6f}, {rtr_matrix.max():.6f}]")

    assert sts_matrix.shape[0] == num_heads, (
        f"Expected {num_heads} heads, got {sts_matrix.shape[0]}")

    # Compute per-head avg synergy and redundancy (for reference)
    synergy_per_head, redundancy_per_head = compute_head_scores(
        sts_matrix, rtr_matrix, num_heads)

    # Compute pair balance scores
    syn_balance, syn_red_score = compute_pair_balance_scores(
        sts_matrix, rtr_matrix, num_heads)

    # Build dataframe
    mapping = get_head_layer_mapping(num_layers, num_heads_per_layer)
    rows = []
    for h in range(num_heads):
        layer, head_in_layer = mapping[h]
        rows.append({
            "head_idx": h,
            "layer": layer,
            "head_in_layer": head_in_layer,
            "avg_synergy": synergy_per_head[h],
            "avg_redundancy": redundancy_per_head[h],
            "syn_balance": syn_balance[h],
            "syn_red_score": syn_red_score[h],
        })

    df = pd.DataFrame(rows)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  syn_red_score range: "
          f"[{df['syn_red_score'].min():.4f}, {df['syn_red_score'].max():.4f}]")

    # Per-layer profile
    print(f"\n{'Layer':>5}  {'Mean Score':>10}  {'Std':>8}  {'Bar'}")
    print("-" * 50)
    layer_stats = df.groupby("layer")["syn_red_score"].agg(["mean", "std"])
    for layer_idx, row in layer_stats.iterrows():
        bar_len = int(row["mean"] * 40)
        bar = "#" * bar_len
        print(f"{layer_idx:5d}  {row['mean']:10.4f}  {row['std']:8.4f}  |{bar}")


if __name__ == "__main__":
    main()
