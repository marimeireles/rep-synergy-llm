"""
Quick script to compute per-head balanced syn-red scores from concatenated PhiID data.

Uses the pair_balance method: balance(i,j) = sts(i,j) / (sts(i,j) + rtr(i,j))
Then per-head syn_balance = mean balance across all pairs involving that head.
Min-max normalize to get syn_red_score in [0,1].

Pythia-1B: 16 layers x 8 heads = 128 total heads.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
project_root = "/lustre07/scratch/marimeir/rep-synergy-llm"
sys.path.insert(0, project_root)

from src.head_ranking import (
    compute_head_scores,
    compute_pair_balance_scores,
    get_head_layer_mapping,
)

# --- Config ---
NUM_LAYERS = 16
NUM_HEADS_PER_LAYER = 8
NUM_HEADS = NUM_LAYERS * NUM_HEADS_PER_LAYER  # 128

INPUT_FILE = os.path.join(project_root, "results/phiid_scores/pythia1b_concat_phiid.npz")
OUTPUT_FILE = os.path.join(project_root, "results/phiid_scores/pythia1b_concat_head_rankings_balanced.csv")

# --- Load data ---
print(f"Loading {INPUT_FILE}")
data = np.load(INPUT_FILE)
sts_matrix = data["sts_matrix"]
rtr_matrix = data["rtr_matrix"]
print(f"  sts_matrix: shape={sts_matrix.shape}, range=[{sts_matrix.min():.6f}, {sts_matrix.max():.6f}]")
print(f"  rtr_matrix: shape={rtr_matrix.shape}, range=[{rtr_matrix.min():.6f}, {rtr_matrix.max():.6f}]")

# --- Compute per-head avg synergy and redundancy (for reference) ---
synergy_per_head, redundancy_per_head = compute_head_scores(sts_matrix, rtr_matrix, NUM_HEADS)

# --- Compute pair balance scores ---
syn_balance, syn_red_score = compute_pair_balance_scores(sts_matrix, rtr_matrix, NUM_HEADS)

# --- Build dataframe ---
mapping = get_head_layer_mapping(NUM_LAYERS, NUM_HEADS_PER_LAYER)
rows = []
for h in range(NUM_HEADS):
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

# --- Save ---
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")
print(f"  Shape: {df.shape}")
print(f"  syn_red_score range: [{df['syn_red_score'].min():.4f}, {df['syn_red_score'].max():.4f}]")

# --- Per-layer profile ---
print("\n=== Per-Layer Average syn_red_score (Balanced Method) ===")
print(f"{'Layer':>5}  {'Mean Score':>10}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'Bar'}")
print("-" * 70)

layer_stats = df.groupby("layer")["syn_red_score"].agg(["mean", "std", "min", "max"])
for layer_idx, row in layer_stats.iterrows():
    bar_len = int(row["mean"] * 40)
    bar = "#" * bar_len
    print(f"{layer_idx:5d}  {row['mean']:10.4f}  {row['std']:8.4f}  {row['min']:8.4f}  {row['max']:8.4f}  |{bar}")

# --- Also print per-layer average syn_balance (raw, before normalization) ---
print("\n=== Per-Layer Average syn_balance (raw, before min-max normalization) ===")
layer_balance = df.groupby("layer")["syn_balance"].agg(["mean", "std"])
for layer_idx, row in layer_balance.iterrows():
    bar_len = int(row["mean"] * 80)  # scale for visibility
    bar = "#" * bar_len
    print(f"  Layer {layer_idx:2d}: {row['mean']:.6f} +/- {row['std']:.6f}  |{bar}")

# --- Top 10 most synergistic heads ---
print("\n=== Top 10 Most Synergistic Heads ===")
top_syn = df.nlargest(10, "syn_red_score")
print(top_syn[["head_idx", "layer", "head_in_layer", "syn_balance", "syn_red_score"]].to_string(index=False))

# --- Top 10 most redundant heads ---
print("\n=== Top 10 Most Redundant Heads ===")
top_red = df.nsmallest(10, "syn_red_score")
print(top_red[["head_idx", "layer", "head_in_layer", "syn_balance", "syn_red_score"]].to_string(index=False))
