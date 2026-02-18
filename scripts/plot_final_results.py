#!/usr/bin/env python
"""
Generate final figures for all 3 models using the paper's rank_diff method.

Produces:
  1. Per-model layer profile (inverted-U test)
  2. Per-model ablation curve (syn-red vs random)
  3. Combined 3-model comparison figure
  4. Summary statistics printed to stdout
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

OUTPUT_DIR = "final_results"

MODELS = {
    "gemma3_4b": {
        "label": "Gemma 3 4B (pretrained)",
        "num_heads_per_layer": 8,
        "num_layers": 34,
        "color": "tab:blue",
    },
    "qwen3_8b": {
        "label": "Qwen 3 8B",
        "num_heads_per_layer": 32,
        "num_layers": 36,
        "color": "tab:orange",
    },
    "gemma3_12b_it": {
        "label": "Gemma 3 12B (instruct)",
        "num_heads_per_layer": 16,
        "num_layers": 48,
        "color": "tab:green",
    },
}


def load_model_data(model_id):
    """Load rankings and ablation data for one model."""
    rankings = pd.read_csv(os.path.join(OUTPUT_DIR, f"{model_id}_rankings.csv"))
    ablation = pd.read_csv(os.path.join(OUTPUT_DIR, f"{model_id}_ablation.csv"))
    return rankings, ablation


def compute_metrics(rankings, ablation, num_layers):
    """Compute key metrics for one model."""
    metrics = {}

    # Per-layer average syn_red_score
    layer_scores = rankings.groupby("layer")["syn_red_score"].mean()
    metrics["layer_scores"] = layer_scores

    # Inverted-U correlation
    x = np.array(layer_scores.index) / (num_layers - 1)
    quadratic = -(x - 0.5)**2 + 0.25
    r, p = pearsonr(layer_scores.values, quadratic)
    metrics["invU_corr"] = r
    metrics["invU_pval"] = p

    # Thirds test
    third = num_layers // 3
    early = layer_scores.iloc[:third].mean()
    mid = layer_scores.iloc[third:2*third].mean()
    late = layer_scores.iloc[2*third:].mean()
    metrics["early"] = early
    metrics["mid"] = mid
    metrics["late"] = late
    metrics["invU_thirds"] = mid > early and mid > late

    # Syn-red correlation
    r_sr, _ = pearsonr(rankings["avg_synergy"], rankings["avg_redundancy"])
    metrics["syn_red_corr"] = r_sr

    # Ablation AUC ratio
    syn_red = ablation[ablation["order_type"] == "syn_red"].copy()
    random_orders = [c for c in ablation["order_type"].unique() if c.startswith("random")]
    random_dfs = [ablation[ablation["order_type"] == r].copy() for r in random_orders]

    num_heads = len(rankings)

    # Align on common x-axis (syn_red may have fewer steps than random)
    syn_steps = set(syn_red["num_heads_removed"].values)
    common_steps = sorted(syn_steps)

    syn_red_aligned = syn_red[syn_red["num_heads_removed"].isin(common_steps)]
    fractions = syn_red_aligned["num_heads_removed"].values / num_heads

    auc_syn = np.trapz(syn_red_aligned["mean_kl_div"].values, fractions)

    # Interpolate random onto common steps
    random_all = []
    for rdf in random_dfs:
        rdf_interp = np.interp(
            syn_red_aligned["num_heads_removed"].values,
            rdf["num_heads_removed"].values,
            rdf["mean_kl_div"].values,
        )
        random_all.append(rdf_interp)
    random_all = np.array(random_all)
    random_mean = np.mean(random_all, axis=0)
    random_std = np.std(random_all, axis=0)
    auc_random = np.trapz(random_mean, fractions)

    metrics["auc_syn"] = auc_syn
    metrics["auc_random"] = auc_random
    metrics["auc_ratio"] = auc_syn / auc_random if auc_random > 0 else float("inf")
    metrics["syn_red_kl"] = syn_red_aligned
    metrics["random_mean_kl"] = random_mean
    metrics["random_std_kl"] = random_std
    metrics["fractions"] = fractions

    return metrics


def plot_per_model(model_id, info, rankings, metrics):
    """Generate per-model figure with profile + ablation side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    num_layers = info["num_layers"]
    layer_scores = metrics["layer_scores"]
    x_layers = np.array(layer_scores.index) / (num_layers - 1)

    # Layer profile
    ax = axes[0]
    ax.plot(x_layers, layer_scores.values, "o-", color=info["color"],
            markersize=3, linewidth=1.5)
    ax.set_xlabel("Normalized Layer Depth")
    ax.set_ylabel("Mean Syn-Red Score (rank_diff)")
    ax.set_title(f"Layer Profile (r_quad={metrics['invU_corr']:.3f})")
    ax.grid(True, alpha=0.3)

    # Ablation curve
    ax = axes[1]
    syn_kl = metrics["syn_red_kl"]["mean_kl_div"].values
    rand_kl = metrics["random_mean_kl"]
    rand_std = metrics["random_std_kl"]
    fracs = metrics["fractions"]

    ax.plot(fracs, syn_kl, "-", color="red", linewidth=2, label="Syn-red order")
    ax.plot(fracs, rand_kl, "--", color="gray", linewidth=1.5, label="Random (mean)")
    ax.fill_between(fracs, rand_kl - rand_std, rand_kl + rand_std,
                    color="gray", alpha=0.2, label="Random (±1 std)")
    ax.set_xlabel("Fraction of Heads Removed")
    ax.set_ylabel("KL Divergence")
    ax.set_title(f"Ablation (AUC ratio={metrics['auc_ratio']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{info['label']} — Paper's Method (rank_diff)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{model_id}_results.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(all_data):
    """Generate combined comparison figure for all 3 models."""
    n = len(all_data)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 9), squeeze=False)

    for col, (model_id, (info, rankings, metrics)) in enumerate(all_data.items()):
        num_layers = info["num_layers"]
        layer_scores = metrics["layer_scores"]
        x_layers = np.array(layer_scores.index) / (num_layers - 1)

        # Row 0: Layer profile
        ax = axes[0, col]
        ax.plot(x_layers, layer_scores.values, "o-", color=info["color"],
                markersize=3, linewidth=1.5)
        ax.set_xlabel("Normalized Layer Depth")
        ax.set_title(f"{info['label']}\nr_quad={metrics['invU_corr']:.3f}, "
                     f"thirds={'YES' if metrics['invU_thirds'] else 'NO'}")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Mean Syn-Red Score")

        # Row 1: Ablation curve
        ax = axes[1, col]
        syn_kl = metrics["syn_red_kl"]["mean_kl_div"].values
        rand_kl = metrics["random_mean_kl"]
        rand_std = metrics["random_std_kl"]
        fracs = metrics["fractions"]

        ax.plot(fracs, syn_kl, "-", color="red", linewidth=2, label="Syn-red")
        ax.plot(fracs, rand_kl, "--", color="gray", linewidth=1.5, label="Random")
        ax.fill_between(fracs, rand_kl - rand_std, rand_kl + rand_std,
                        color="gray", alpha=0.2)
        ax.set_xlabel("Fraction Removed")
        ax.set_title(f"AUC ratio={metrics['auc_ratio']:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("KL Divergence")

    fig.suptitle("PhiID Analysis — Paper's Method (rank_diff) — All Models", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "combined_results.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    all_data = {}

    print("=" * 70)
    print("FINAL RESULTS — Paper's rank_diff method")
    print("=" * 70)

    for model_id, info in MODELS.items():
        print(f"\n--- {info['label']} ---")
        rankings, ablation = load_model_data(model_id)
        metrics = compute_metrics(rankings, ablation, info["num_layers"])

        print(f"  Heads: {len(rankings)} ({info['num_layers']}L x {info['num_heads_per_layer']}H)")
        print(f"  Syn-red correlation: r={metrics['syn_red_corr']:.4f}")
        print(f"  Inverted-U (quadratic): r={metrics['invU_corr']:.4f} (p={metrics['invU_pval']:.2e})")
        print(f"  Thirds test: early={metrics['early']:.3f} mid={metrics['mid']:.3f} "
              f"late={metrics['late']:.3f} → {'PASS' if metrics['invU_thirds'] else 'FAIL'}")
        print(f"  Ablation AUC: syn={metrics['auc_syn']:.4f} random={metrics['auc_random']:.4f} "
              f"ratio={metrics['auc_ratio']:.4f}")

        plot_per_model(model_id, info, rankings, metrics)
        all_data[model_id] = (info, rankings, metrics)

    plot_combined(all_data)

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Model':<28} {'Syn-Red r':>9} {'InvU r':>7} {'Thirds':>7} {'AUC ratio':>10}")
    print("-" * 65)
    for model_id, (info, rankings, metrics) in all_data.items():
        print(f"{info['label']:<28} {metrics['syn_red_corr']:>9.4f} "
              f"{metrics['invU_corr']:>7.4f} "
              f"{'PASS' if metrics['invU_thirds'] else 'FAIL':>7} "
              f"{metrics['auc_ratio']:>10.4f}")

    print(f"\nFigures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
