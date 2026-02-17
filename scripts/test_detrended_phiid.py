#!/usr/bin/env python
"""
Test whether detrending activation time series before PhiID fixes the
positive syn-red correlation and produces cleaner inverted-U profiles.

Rationale:
  - Brain fMRI papers (Luppi 2022) detrend + bandpass filter before PhiID
  - Our LLM activation time series show clear downward drift during generation
  - This non-stationarity inflates TDMI equally for sts and rtr, creating
    the positive correlation (r=0.82-0.90) that kills the rank_diff method
  - Detrending removes the shared drift and may reveal true temporal dynamics

Methods tested:
  1. raw        — no preprocessing (baseline, same as original pipeline)
  2. linear     — scipy.signal.detrend(type='linear') per prompt per head
  3. first_diff — x[t] - x[t-1], focuses on changes rather than levels
  4. poly2      — remove quadratic trend (captures curved drift better)

Usage:
  python scripts/test_detrended_phiid.py --model gemma3-4b-it --max-workers 32
"""

import argparse
import logging
import os
import sys
import time
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.head_ranking import (
    compute_head_scores, compute_syn_red_rank,
    get_head_layer_mapping, build_ranking_dataframe,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ARCHITECTURES = {
    "pythia1b": (8, 16),
    "gemma3_4b": (8, 34),
    "gemma3_4b_it": (8, 34),
    "gemma3_12b_it": (16, 48),
    "qwen3_8b": (32, 36),
}

MODEL_ID_MAP = {
    "pythia-1b": "pythia1b",
    "gemma3-4b": "gemma3_4b",
    "gemma3-4b-it": "gemma3_4b_it",
    "gemma3-12b-it": "gemma3_12b_it",
    "qwen3-8b": "qwen3_8b",
}


# ── Detrending methods ──────────────────────────────────────────────────

def detrend_linear(activations):
    """Remove linear trend per head per prompt."""
    out = np.empty_like(activations)
    n_prompts, n_heads, n_steps = activations.shape
    for p in range(n_prompts):
        for h in range(n_heads):
            out[p, h, :] = signal.detrend(activations[p, h, :], type='linear')
    return out


def detrend_first_diff(activations):
    """First-difference: x[t] - x[t-1]. Output has n_steps-1 timesteps."""
    return np.diff(activations, axis=2)


def detrend_poly2(activations):
    """Remove quadratic (degree-2 polynomial) trend per head per prompt."""
    out = np.empty_like(activations)
    n_prompts, n_heads, n_steps = activations.shape
    t = np.arange(n_steps)
    for p in range(n_prompts):
        for h in range(n_heads):
            coeffs = np.polyfit(t, activations[p, h, :], 2)
            trend = np.polyval(coeffs, t)
            out[p, h, :] = activations[p, h, :] - trend
    return out


def transform_log(activations):
    """Log-transform: log(x). Stabilizes variance and reduces skew."""
    return np.log(activations + 1e-10)


def transform_log_detrend(activations):
    """Log-transform then linear detrend."""
    return detrend_linear(np.log(activations + 1e-10))


DETREND_METHODS = {
    "raw": lambda x: x,
    "linear": detrend_linear,
    "first_diff": detrend_first_diff,
    "poly2": detrend_poly2,
    "log": transform_log,
    "log_detrend": transform_log_detrend,
}


# ── PhiID computation ────────────────────────────────────────────────────

def _compute_pair_per_prompt(args):
    """Compute PhiID for one pair, averaged across all prompts."""
    from phyid.calculate import calc_PhiID

    act_i, act_j, tau, kind, redundancy = args
    num_prompts = act_i.shape[0]
    sts_values = []
    rtr_values = []

    for p in range(num_prompts):
        src = act_i[p]
        trg = act_j[p]
        # Skip if constant or near-constant
        if np.std(src) < 1e-10 or np.std(trg) < 1e-10:
            continue
        try:
            atoms, _ = calc_PhiID(src, trg, tau=tau, kind=kind, redundancy=redundancy)
            sts_values.append(float(np.mean(atoms['sts'])))
            rtr_values.append(float(np.mean(atoms['rtr'])))
        except Exception:
            pass

    if len(sts_values) == 0:
        return 0.0, 0.0
    return float(np.mean(sts_values)), float(np.mean(rtr_values))


def compute_phiid_all_pairs(activations, num_heads, max_workers=None):
    """Compute per-prompt PhiID for all pairs, return sts and rtr matrices."""
    all_pairs = list(combinations(range(num_heads), 2))
    total_pairs = len(all_pairs)

    sts_matrix = np.zeros((num_heads, num_heads))
    rtr_matrix = np.zeros((num_heads, num_heads))

    work_items = []
    for (i, j) in all_pairs:
        work_items.append((
            activations[:, i, :],
            activations[:, j, :],
            1, "gaussian", "MMI",
        ))

    if max_workers == 1:
        for idx, (i, j) in enumerate(tqdm(all_pairs, desc="PhiID")):
            sts_val, rtr_val = _compute_pair_per_prompt(work_items[idx])
            sts_matrix[i, j] = sts_matrix[j, i] = sts_val
            rtr_matrix[i, j] = rtr_matrix[j, i] = rtr_val
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {}
            for idx, item in enumerate(work_items):
                future = executor.submit(_compute_pair_per_prompt, item)
                future_to_pair[future] = all_pairs[idx]

            for future in tqdm(as_completed(future_to_pair), total=total_pairs, desc="PhiID"):
                i, j = future_to_pair[future]
                try:
                    sts_val, rtr_val = future.result()
                    sts_matrix[i, j] = sts_matrix[j, i] = sts_val
                    rtr_matrix[i, j] = rtr_matrix[j, i] = rtr_val
                except Exception as e:
                    logger.warning(f"Pair ({i},{j}) failed: {e}")

    return sts_matrix, rtr_matrix


# ── Analysis ─────────────────────────────────────────────────────────────

def analyze_results(sts_matrix, rtr_matrix, num_heads, num_layers,
                    num_heads_per_layer, method_name, output_dir, model_name):
    """Run full analysis on one method's PhiID results."""
    results = {"method": method_name}

    # Pairwise syn-red correlation
    upper_tri = np.triu_indices(num_heads, k=1)
    sts_vals = sts_matrix[upper_tri]
    rtr_vals = rtr_matrix[upper_tri]
    pw_corr, pw_pval = pearsonr(sts_vals, rtr_vals)
    results["pairwise_syn_red_corr"] = pw_corr
    logger.info(f"  Pairwise syn-red correlation: r={pw_corr:.4f}")

    # Per-head scores
    synergy_per_head, redundancy_per_head = compute_head_scores(
        sts_matrix, rtr_matrix, num_heads)
    head_corr, _ = pearsonr(synergy_per_head, redundancy_per_head)
    results["head_syn_red_corr"] = head_corr
    logger.info(f"  Per-head syn-red correlation: r={head_corr:.4f}")

    # Standard ranking (rank_diff) — paper's method
    syn_red_rank, syn_rank, red_rank = compute_syn_red_rank(
        synergy_per_head, redundancy_per_head)

    # Per-layer profiles
    mapping = get_head_layer_mapping(num_layers, num_heads_per_layer)
    layer_depth_norm = np.array([mapping[h][0] / (num_layers - 1) for h in range(num_heads)])
    quadratic = -(layer_depth_norm - 0.5)**2 + 0.25

    quad_corr, _ = pearsonr(syn_red_rank, quadratic)
    results["invU_corr"] = quad_corr
    logger.info(f"  Inverted-U corr (rank_diff): r={quad_corr:.4f}")

    # Per-layer table
    logger.info(f"  {'Layer':>5}  {'Score':>8}")
    logger.info(f"  {'-'*16}")
    layer_scores = []
    for layer in range(num_layers):
        heads_in_layer = [h for h in range(num_heads) if mapping[h][0] == layer]
        score_mean = np.mean([syn_red_rank[h] for h in heads_in_layer])
        layer_scores.append(score_mean)
        logger.info(f"  {layer:5d}  {score_mean:8.4f}")

    results["layer_scores"] = layer_scores

    # Save matrices and rankings
    prefix = f"{output_dir}/{model_name}_{method_name}"
    np.savez(f"{prefix}_pairwise_phiid.npz",
             sts_matrix=sts_matrix, rtr_matrix=rtr_matrix)

    df = build_ranking_dataframe(
        synergy_per_head, redundancy_per_head, syn_red_rank, syn_rank, red_rank,
        num_layers, num_heads_per_layer)
    df.to_csv(f"{prefix}_rankings.csv", index=False)

    return results


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_comparison(all_results, num_layers, output_dir, model_name):
    """Plot side-by-side comparison of all detrending methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in all_results]
    n_methods = len(methods)
    x = np.arange(num_layers) / (num_layers - 1)

    # ── Figure 1: Layer profiles (rank_diff) ──
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4),
                             squeeze=False, sharey=True)

    for col, r in enumerate(all_results):
        axes[0, col].plot(x, r["layer_scores"], 'b-o', markersize=3)
        axes[0, col].set_title(f'{r["method"]}\nr_quad={r["invU_corr"]:.3f}')
        axes[0, col].set_xlabel('Normalized Layer Depth')
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel('Mean Syn-Red Score (rank_diff)')

    fig.suptitle(f'{model_name}: Detrending Method Comparison — Layer Profiles (rank_diff)',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{model_name}_detrend_profiles.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved profiles plot")

    # ── Figure 2: Correlation summary ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['gray', 'steelblue', 'coral', 'seagreen'][:n_methods]

    # Pairwise syn-red correlation
    vals = [r["pairwise_syn_red_corr"] for r in all_results]
    bars = axes[0].bar(methods, vals, color=colors)
    axes[0].set_ylabel('Pearson r')
    axes[0].set_title('Pairwise Syn-Red Correlation\n(lower/negative = better)')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=9)

    # Per-head syn-red correlation
    vals = [r["head_syn_red_corr"] for r in all_results]
    bars = axes[1].bar(methods, vals, color=colors)
    axes[1].set_ylabel('Pearson r')
    axes[1].set_title('Per-Head Syn-Red Correlation\n(lower/negative = better)')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=9)

    # Inverted-U correlation (rank_diff only)
    vals = [r["invU_corr"] for r in all_results]
    bars = axes[2].bar(methods, vals, color=colors)
    axes[2].set_ylabel('Correlation with inverted-U')
    axes[2].set_title('Inverted-U Strength (rank_diff)\n(higher = better)')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=9)

    fig.suptitle(f'{model_name}: Detrending Method Comparison — Key Metrics', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{model_name}_detrend_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved summary plot")


def plot_sample_timeseries(activations_dict, output_dir, model_name):
    """Plot sample time series for each detrending method to visualize the effect."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(activations_dict.keys())
    n_methods = len(methods)
    # Pick 3 heads from different layers
    sample_heads = [0, 68, 136]  # layers 0, 8, 17 approximately
    sample_prompt = 0

    fig, axes = plt.subplots(len(sample_heads), n_methods,
                             figsize=(4 * n_methods, 3 * len(sample_heads)),
                             squeeze=False)

    for col, method in enumerate(methods):
        act = activations_dict[method]
        for row, h in enumerate(sample_heads):
            ts = act[sample_prompt, h, :]
            axes[row, col].plot(ts, linewidth=0.8)
            axes[row, col].set_title(f'{method} — head {h}', fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(f'Activation')
            if row == len(sample_heads) - 1:
                axes[row, col].set_xlabel('Generation step')
            axes[row, col].grid(True, alpha=0.2)

    fig.suptitle(f'{model_name}: Effect of Detrending on Activation Time Series', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/{model_name}_detrend_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved time series visualization")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test detrended PhiID")
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_ID_MAP.keys()))
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='testing_approaches/03_detrended')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['raw', 'linear', 'first_diff', 'poly2', 'log', 'log_detrend'],
                        choices=list(DETREND_METHODS.keys()),
                        help='Preprocessing methods to test')
    args = parser.parse_args()

    model_id = MODEL_ID_MAP[args.model]
    num_heads_per_layer, num_layers = ARCHITECTURES[model_id]
    num_heads = num_layers * num_heads_per_layer

    os.makedirs(args.output_dir, exist_ok=True)

    # Load activations
    act_path = os.path.join(args.results_dir, 'activations', f'{model_id}_activations.npz')
    logger.info(f"Loading activations from {act_path}")
    data = np.load(act_path)
    activations = data['activations']
    n_prompts, n_heads_check, n_steps = activations.shape
    assert n_heads_check == num_heads, f"Expected {num_heads} heads, got {n_heads_check}"
    logger.info(f"Shape: {activations.shape} ({num_layers}L x {num_heads_per_layer}H = {num_heads} heads)")

    # ── Diagnostic: autocorrelation before/after detrending ──
    logger.info("\n" + "=" * 60)
    logger.info("AUTOCORRELATION DIAGNOSTICS")
    logger.info("=" * 60)
    sample_heads_diag = np.random.RandomState(42).choice(num_heads, 20, replace=False)

    activations_dict = {}
    for method_name in args.methods:
        logger.info(f"\n--- {method_name} ---")
        detrend_fn = DETREND_METHODS[method_name]
        t0 = time.time()
        act_dt = detrend_fn(activations)
        logger.info(f"  Detrending took {time.time()-t0:.1f}s, output shape: {act_dt.shape}")
        activations_dict[method_name] = act_dt

        # Autocorrelation stats
        all_acs = []
        for h in sample_heads_diag:
            for p in range(n_prompts):
                ts = act_dt[p, h, :]
                if len(ts) > 2 and np.std(ts) > 1e-10:
                    ac = np.corrcoef(ts[:-1], ts[1:])[0, 1]
                    if np.isfinite(ac):
                        all_acs.append(ac)
        logger.info(f"  Lag-1 autocorrelation: mean={np.mean(all_acs):.4f}, "
                    f"median={np.median(all_acs):.4f}, std={np.std(all_acs):.4f}")

        # Stationarity check: variance in first half vs second half
        var_ratios = []
        for h in sample_heads_diag:
            for p in range(n_prompts):
                ts = act_dt[p, h, :]
                mid = len(ts) // 2
                v1 = np.var(ts[:mid])
                v2 = np.var(ts[mid:])
                if v1 > 1e-10 and v2 > 1e-10:
                    var_ratios.append(v2 / v1)
        logger.info(f"  Variance ratio (2nd half / 1st half): "
                    f"mean={np.mean(var_ratios):.3f}, median={np.median(var_ratios):.3f}")

    # Plot sample time series
    plot_sample_timeseries(activations_dict, args.output_dir, model_id)

    # ── Compute PhiID for each method ──
    all_results = []
    for method_name in args.methods:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"COMPUTING PhiID: {method_name}")
        logger.info(f"{'=' * 60}")

        act_dt = activations_dict[method_name]
        t0 = time.time()
        sts_matrix, rtr_matrix = compute_phiid_all_pairs(
            act_dt, num_heads, max_workers=args.max_workers)
        elapsed = time.time() - t0
        logger.info(f"  PhiID done in {elapsed/60:.1f} minutes")

        logger.info(f"\n--- Analysis: {method_name} ---")
        results = analyze_results(
            sts_matrix, rtr_matrix, num_heads, num_layers,
            num_heads_per_layer, method_name, args.output_dir, model_id)
        all_results.append(results)

    # ── Comparison plot ──
    plot_comparison(all_results, num_layers, args.output_dir, model_id)

    # ── Summary table ──
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"{'Method':<12} {'PW corr':>8} {'Head corr':>10} {'InvU':>9}")
    logger.info("-" * 42)
    for r in all_results:
        logger.info(f"{r['method']:<12} {r['pairwise_syn_red_corr']:>8.4f} "
                    f"{r['head_syn_red_corr']:>10.4f} "
                    f"{r['invU_corr']:>9.4f}")

    best = max(all_results, key=lambda r: r["invU_corr"])
    logger.info(f"\nBest inverted-U (rank_diff): {best['method']} "
                f"(r={best['invU_corr']:.4f})")


if __name__ == '__main__':
    main()
