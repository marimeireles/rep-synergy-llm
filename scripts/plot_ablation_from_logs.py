#!/usr/bin/env python
"""
Plot ablation curves (synergistic vs random ordering) for Pythia-1B and Qwen 3 8B
by parsing SLURM log files.

Reproduces paper Figure 4a style: solid line for synergistic ordering,
dashed line for random mean with shaded std region.

Usage:
    python scripts/plot_ablation_from_logs.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Log file paths
# ---------------------------------------------------------------------------
PYTHIA_LOG = Path(
    "/lustre07/scratch/marimeir/rep-synergy-llm/results/"
    "slurm_pythia_ablation_concat_56525751.err"
)
QWEN_LOG = Path(
    "/lustre07/scratch/marimeir/rep-synergy-llm/results/"
    "slurm_qwen_ablation_balanced_56525753.err"
)
OUTPUT_PATH = Path(
    "/lustre07/scratch/marimeir/rep-synergy-llm/results/figures/"
    "ablation_curves_improved.png"
)

# Regex to extract ablation data lines
# Matches: [syn_red] Removed 16/128 heads, mean KL = 0.1357
# Also:    [random_0] Removed 16/128 heads, mean KL = 0.1850
PATTERN = re.compile(
    r'\[(syn_red|random_\d+)\] Removed (\d+)/(\d+) heads, mean KL = ([\d.]+)'
)


def parse_log(log_path):
    """
    Parse a SLURM log file and extract ablation data.

    Returns:
        syn_red: dict mapping num_removed -> kl_value
        randoms: dict mapping seed_name -> {num_removed -> kl_value}
        total_heads: int
    """
    syn_red = {}
    randoms = defaultdict(dict)
    total_heads = None

    with open(log_path, 'r') as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                order_name = match.group(1)
                num_removed = int(match.group(2))
                total = int(match.group(3))
                kl_val = float(match.group(4))

                if total_heads is None:
                    total_heads = total
                else:
                    assert total_heads == total, (
                        f"Inconsistent total heads: {total_heads} vs {total}"
                    )

                if order_name == 'syn_red':
                    syn_red[num_removed] = kl_val
                else:
                    randoms[order_name][num_removed] = kl_val

    return syn_red, randoms, total_heads


def build_arrays(syn_red_dict, randoms_dict, total_heads):
    """
    Build sorted arrays from parsed data, prepending the (0, 0) origin point.

    Returns:
        frac_syn: array of fractions for syn_red
        kl_syn: array of KL values for syn_red
        frac_rand: array of fractions for random (common x-axis)
        kl_rand_mean: array of mean KL across random seeds
        kl_rand_std: array of std KL across random seeds
    """
    # --- Synergistic ordering ---
    sorted_removed = sorted(syn_red_dict.keys())
    frac_syn = np.array([0.0] + [r / total_heads for r in sorted_removed])
    kl_syn = np.array([0.0] + [syn_red_dict[r] for r in sorted_removed])

    # --- Random orderings ---
    # Collect all unique num_removed values across random seeds
    all_removed = set()
    for seed_data in randoms_dict.values():
        all_removed.update(seed_data.keys())
    sorted_rand_removed = sorted(all_removed)

    num_seeds = len(randoms_dict)
    if num_seeds == 0:
        return frac_syn, kl_syn, np.array([]), np.array([]), np.array([])

    # Build a matrix: rows = removal steps, cols = seeds
    kl_matrix = np.full((len(sorted_rand_removed), num_seeds), np.nan)
    seed_names = sorted(randoms_dict.keys())
    for j, seed_name in enumerate(seed_names):
        seed_data = randoms_dict[seed_name]
        for i, num_rem in enumerate(sorted_rand_removed):
            if num_rem in seed_data:
                kl_matrix[i, j] = seed_data[num_rem]

    frac_rand = np.array([0.0] + [r / total_heads for r in sorted_rand_removed])
    kl_rand_mean = np.array([0.0] + list(np.nanmean(kl_matrix, axis=1)))
    kl_rand_std = np.array([0.0] + list(np.nanstd(kl_matrix, axis=1)))

    return frac_syn, kl_syn, frac_rand, kl_rand_mean, kl_rand_std


def plot_single_model(ax, frac_syn, kl_syn, frac_rand, kl_rand_mean, kl_rand_std,
                      title, num_random_seeds):
    """Plot ablation curves for a single model on the given axes."""
    # Synergistic ordering (solid line)
    ax.plot(frac_syn, kl_syn, '-', color='#d62728', linewidth=2.2,
            label='Synergistic order', zorder=3)

    # Random ordering (dashed line with shaded std)
    if len(frac_rand) > 0:
        seed_label = f'Random order (mean +/- std, n={num_random_seeds})'
        ax.plot(frac_rand, kl_rand_mean, '--', color='#7f7f7f', linewidth=2.0,
                label=seed_label, zorder=2)
        ax.fill_between(
            frac_rand,
            kl_rand_mean - kl_rand_std,
            kl_rand_mean + kl_rand_std,
            color='#7f7f7f', alpha=0.20, zorder=1,
        )

    ax.set_xlabel('Fraction of Heads Deactivated', fontsize=12)
    ax.set_ylabel('Behaviour Divergence (KL)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def main():
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Parse Pythia-1B log ---
    print(f"Parsing Pythia log: {PYTHIA_LOG}")
    pythia_syn, pythia_rand, pythia_total = parse_log(PYTHIA_LOG)
    print(f"  Total heads: {pythia_total}")
    print(f"  syn_red data points: {len(pythia_syn)}")
    print(f"  Random seeds: {len(pythia_rand)} "
          f"({', '.join(sorted(pythia_rand.keys()))})")

    p_frac_syn, p_kl_syn, p_frac_rand, p_kl_rand_mean, p_kl_rand_std = \
        build_arrays(pythia_syn, pythia_rand, pythia_total)

    # --- Parse Qwen 3 8B log ---
    print(f"\nParsing Qwen log: {QWEN_LOG}")
    qwen_syn, qwen_rand, qwen_total = parse_log(QWEN_LOG)
    print(f"  Total heads: {qwen_total}")
    print(f"  syn_red data points: {len(qwen_syn)}")
    print(f"  Random seeds: {len(qwen_rand)} "
          f"({', '.join(sorted(qwen_rand.keys()))})")

    q_frac_syn, q_kl_syn, q_frac_rand, q_kl_rand_mean, q_kl_rand_std = \
        build_arrays(qwen_syn, qwen_rand, qwen_total)

    # --- Create side-by-side plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    plot_single_model(
        ax1, p_frac_syn, p_kl_syn, p_frac_rand, p_kl_rand_mean, p_kl_rand_std,
        title=f'Pythia-1B ({pythia_total} heads)',
        num_random_seeds=len(pythia_rand),
    )

    plot_single_model(
        ax2, q_frac_syn, q_kl_syn, q_frac_rand, q_kl_rand_mean, q_kl_rand_std,
        title=f'Qwen 3 8B ({qwen_total} heads)',
        num_random_seeds=len(qwen_rand),
    )

    fig.suptitle(
        'Ablation Curves: Synergistic vs Random Head Removal Order',
        fontsize=16, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved figure to: {OUTPUT_PATH}")

    # --- Print summary statistics ---
    print("\n--- Pythia-1B Summary ---")
    print(f"  Final KL (all heads removed, syn_red): {p_kl_syn[-1]:.4f}")
    if len(p_kl_rand_mean) > 0:
        print(f"  Final KL (all heads removed, random mean): {p_kl_rand_mean[-1]:.4f}")
    # Check if syn_red is consistently above random
    if len(p_frac_rand) > 0 and len(p_frac_syn) > 0:
        # Interpolate syn_red onto random x-grid for comparison
        syn_interp = np.interp(p_frac_rand, p_frac_syn, p_kl_syn)
        diff = syn_interp - p_kl_rand_mean
        above_count = np.sum(diff > 0)
        print(f"  syn_red above random at {above_count}/{len(diff)} points "
              f"(mean diff = {np.mean(diff):+.4f})")

    print("\n--- Qwen 3 8B Summary ---")
    print(f"  Final KL (all heads removed, syn_red): {q_kl_syn[-1]:.4f}")
    if len(q_kl_rand_mean) > 0:
        print(f"  Final KL (all heads removed, random mean): {q_kl_rand_mean[-1]:.4f}")
    if len(q_frac_rand) > 0 and len(q_frac_syn) > 0:
        syn_interp = np.interp(q_frac_rand, q_frac_syn, q_kl_syn)
        diff = syn_interp - q_kl_rand_mean
        above_count = np.sum(diff > 0)
        print(f"  syn_red above random at {above_count}/{len(diff)} points "
              f"(mean diff = {np.mean(diff):+.4f})")


if __name__ == '__main__':
    main()
