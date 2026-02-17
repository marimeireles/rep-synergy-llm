"""
Approach 1: Per-pair balance normalization — sts/(sts+rtr) per pair before averaging.

This removes the positive correlation between synergy and redundancy caused by
overall coupling magnitude. Instead of ranking heads by raw synergy and redundancy
magnitudes, we measure the synergy *fraction* per pair.

Uses ONLY existing pairwise PhiID matrices — no recomputation needed.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configs: model_id -> (num_heads_per_layer, num_layers)
MODELS = {
    'pythia1b': (8, 16),
    'qwen3_8b': (32, 36),
    'gemma3_4b': (8, 34),
    'gemma3_4b_it': (8, 34),
    'gemma3_12b_it': (16, 48),
}


def load_phiid_matrices(model_id):
    """Load pairwise sts and rtr matrices."""
    path = os.path.join(RESULTS_DIR, 'phiid_scores', f'{model_id}_pairwise_phiid.npz')
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data['sts_matrix'], data['rtr_matrix']


def compute_standard_ranking(sts_matrix, rtr_matrix, num_heads):
    """Paper's method: rank(syn) - rank(red), min-max normalized."""
    syn_per_head = np.zeros(num_heads)
    red_per_head = np.zeros(num_heads)
    for h in range(num_heads):
        mask = np.ones(num_heads, dtype=bool)
        mask[h] = False
        syn_per_head[h] = np.mean(sts_matrix[h, mask])
        red_per_head[h] = np.mean(rtr_matrix[h, mask])

    syn_rank = rankdata(syn_per_head, method='ordinal')
    red_rank = rankdata(red_per_head, method='ordinal')
    raw_score = syn_rank - red_rank
    score_min, score_max = raw_score.min(), raw_score.max()
    if score_max > score_min:
        normalized = (raw_score - score_min) / (score_max - score_min)
    else:
        normalized = np.full(num_heads, 0.5)

    return syn_per_head, red_per_head, normalized


def compute_balanced_ranking(sts_matrix, rtr_matrix, num_heads):
    """Per-pair balance: sts/(sts+rtr) per pair, then average per head."""
    denom = sts_matrix + rtr_matrix
    denom[denom < 1e-10] = 1.0
    balance = sts_matrix / denom

    syn_score = np.zeros(num_heads)
    for h in range(num_heads):
        mask = np.ones(num_heads, dtype=bool)
        mask[h] = False
        syn_score[h] = np.mean(balance[h, mask])

    score_min, score_max = syn_score.min(), syn_score.max()
    if score_max > score_min:
        normalized = (syn_score - score_min) / (score_max - score_min)
    else:
        normalized = np.full(num_heads, 0.5)

    return syn_score, normalized


def assess_inverted_u(layer_scores):
    """Check for inverted-U pattern using thirds comparison."""
    n = len(layer_scores)
    third = max(1, n // 3)
    early = np.mean(layer_scores[:third])
    mid = np.mean(layer_scores[third:2*third])
    late = np.mean(layer_scores[2*third:])
    inverted_u = (mid > early) and (mid > late)
    return inverted_u, early, mid, late


def quadratic_fit(layer_scores):
    """Fit quadratic to layer scores, return coefficient and peak."""
    x = np.linspace(0, 1, len(layer_scores))
    coeffs = np.polyfit(x, layer_scores, 2)
    peak = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else 0.5
    return coeffs[0], peak


def analyze_model(model_id, nhl, nlayers):
    """Run both standard and balanced analysis for one model."""
    sts, rtr = load_phiid_matrices(model_id)
    if sts is None:
        return None

    num_heads = sts.shape[0]
    expected = nhl * nlayers
    if num_heads != expected:
        print(f"  WARNING: {model_id} has {num_heads} heads in PhiID matrix, expected {expected}")
        # Use actual matrix size
        if num_heads % nhl == 0:
            nlayers = num_heads // nhl
        else:
            print(f"  Cannot infer architecture, skipping")
            return None

    results = {'model_id': model_id, 'num_heads': num_heads,
               'num_layers': nlayers, 'heads_per_layer': nhl}

    # --- Standard method ---
    syn_per_head, red_per_head, std_scores = compute_standard_ranking(sts, rtr, num_heads)
    r_std, _ = pearsonr(syn_per_head, red_per_head)
    rho_std, _ = spearmanr(syn_per_head, red_per_head)
    results['standard_pearson_r'] = r_std
    results['standard_spearman_r'] = rho_std

    # Per-layer profile (standard)
    std_layer_scores = []
    for l in range(nlayers):
        start, end = l * nhl, (l + 1) * nhl
        std_layer_scores.append(np.mean(std_scores[start:end]))
    std_layer_scores = np.array(std_layer_scores)

    inv_u_std, e_std, m_std, l_std = assess_inverted_u(std_layer_scores)
    q_std, peak_std = quadratic_fit(std_layer_scores)
    results['standard_inverted_u'] = inv_u_std
    results['standard_quad_coeff'] = q_std
    results['standard_peak'] = peak_std
    results['standard_mid_minus_early'] = m_std - e_std
    results['standard_mid_minus_late'] = m_std - l_std

    # --- Balanced method ---
    bal_raw, bal_scores = compute_balanced_ranking(sts, rtr, num_heads)

    # Correlation in balanced space: does the balance score correlate with raw syn or red?
    r_bal_syn, _ = pearsonr(bal_raw, syn_per_head)
    r_bal_red, _ = pearsonr(bal_raw, red_per_head)
    results['balance_corr_with_syn'] = r_bal_syn
    results['balance_corr_with_red'] = r_bal_red

    # Per-pair correlation check
    mask = np.triu(np.ones((num_heads, num_heads), dtype=bool), k=1)
    sts_pairs = sts[mask]
    rtr_pairs = rtr[mask]
    denom = sts_pairs + rtr_pairs
    denom[denom < 1e-10] = 1.0
    balance_pairs = sts_pairs / denom
    r_pair_raw, _ = pearsonr(sts_pairs, rtr_pairs)
    results['pair_level_corr_raw'] = r_pair_raw
    results['pair_balance_mean'] = np.mean(balance_pairs)
    results['pair_balance_std'] = np.std(balance_pairs)

    # Per-layer profile (balanced)
    bal_layer_scores = []
    for l in range(nlayers):
        start, end = l * nhl, (l + 1) * nhl
        bal_layer_scores.append(np.mean(bal_scores[start:end]))
    bal_layer_scores = np.array(bal_layer_scores)

    inv_u_bal, e_bal, m_bal, l_bal = assess_inverted_u(bal_layer_scores)
    q_bal, peak_bal = quadratic_fit(bal_layer_scores)
    results['balanced_inverted_u'] = inv_u_bal
    results['balanced_quad_coeff'] = q_bal
    results['balanced_peak'] = peak_bal
    results['balanced_mid_minus_early'] = m_bal - e_bal
    results['balanced_mid_minus_late'] = m_bal - l_bal

    # --- Ranking agreement ---
    # How much do the two methods agree on which heads are most synergistic?
    top_k = max(1, num_heads // 10)  # top 10%
    std_top = set(np.argsort(std_scores)[-top_k:])
    bal_top = set(np.argsort(bal_scores)[-top_k:])
    results['top10_overlap'] = len(std_top & bal_top) / top_k

    # Save rankings CSV
    mapping = []
    for l in range(nlayers):
        for h in range(nhl):
            mapping.append((l, h))

    df = pd.DataFrame({
        'head_idx': range(num_heads),
        'layer': [m[0] for m in mapping],
        'head_in_layer': [m[1] for m in mapping],
        'avg_synergy': syn_per_head,
        'avg_redundancy': red_per_head,
        'standard_score': std_scores,
        'balance_raw': bal_raw,
        'balanced_score': bal_scores,
    })
    csv_path = os.path.join(OUTPUT_DIR, f'{model_id}_balanced_rankings.csv')
    df.to_csv(csv_path, index=False)

    # --- Plot comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Layer profiles: standard vs balanced
    x = np.linspace(0, 1, nlayers)
    axes[0].plot(x, std_layer_scores, 'b-o', label='Standard (rank_diff)', markersize=4)
    axes[0].plot(x, bal_layer_scores, 'r-o', label='Balanced (sts/(sts+rtr))', markersize=4)
    axes[0].set_xlabel('Normalized Layer Depth')
    axes[0].set_ylabel('Average Syn-Red Score')
    axes[0].set_title(f'{model_id}: Layer Profiles')
    axes[0].legend(fontsize=8)

    # 2) Scatter: syn vs red (standard)
    axes[1].scatter(syn_per_head, red_per_head, alpha=0.3, s=10)
    axes[1].set_xlabel('Per-head Avg Synergy')
    axes[1].set_ylabel('Per-head Avg Redundancy')
    axes[1].set_title(f'r={r_std:.3f} (positive correlation problem)')

    # 3) Head heatmap (balanced)
    heatmap_data = bal_scores.reshape(nlayers, nhl)
    im = axes[2].imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
    axes[2].set_xlabel('Head Index')
    axes[2].set_ylabel('Layer')
    axes[2].set_title(f'{model_id}: Balanced Score Heatmap')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'{model_id}_balance_analysis.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return results


def main():
    print("=" * 70)
    print("APPROACH 1: Per-Pair Balance Normalization")
    print("Method: sts/(sts+rtr) per pair before averaging per head")
    print("=" * 70)

    all_results = []

    for model_id, (nhl, nlayers) in MODELS.items():
        print(f"\n--- {model_id} ({nlayers} layers x {nhl} heads = {nhl*nlayers} total) ---")
        result = analyze_model(model_id, nhl, nlayers)
        if result is None:
            print(f"  Skipped (no data)")
            continue
        all_results.append(result)

        print(f"  Standard method:")
        print(f"    Pearson r(syn,red)  = {result['standard_pearson_r']:.3f}")
        print(f"    Spearman r(syn,red) = {result['standard_spearman_r']:.3f}")
        print(f"    Inverted-U: {result['standard_inverted_u']}  (quad={result['standard_quad_coeff']:.3f}, peak={result['standard_peak']:.2f})")
        print(f"    Mid-Early: {result['standard_mid_minus_early']:+.4f}, Mid-Late: {result['standard_mid_minus_late']:+.4f}")

        print(f"  Balanced method:")
        print(f"    Balance corr w/ raw syn: {result['balance_corr_with_syn']:.3f}")
        print(f"    Balance corr w/ raw red: {result['balance_corr_with_red']:.3f}")
        print(f"    Pair-level mean balance: {result['pair_balance_mean']:.4f} +/- {result['pair_balance_std']:.4f}")
        print(f"    Inverted-U: {result['balanced_inverted_u']}  (quad={result['balanced_quad_coeff']:.3f}, peak={result['balanced_peak']:.2f})")
        print(f"    Mid-Early: {result['balanced_mid_minus_early']:+.4f}, Mid-Late: {result['balanced_mid_minus_late']:+.4f}")

        print(f"  Ranking agreement:")
        print(f"    Top-10% overlap: {result['top10_overlap']*100:.1f}%")
        print(f"    Pair-level r(sts,rtr) raw: {result['pair_level_corr_raw']:.3f}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<20} {'Std r(S,R)':>10} {'Std InvU':>8} {'Std Quad':>8} "
          f"{'Bal InvU':>8} {'Bal Quad':>8} {'Bal Peak':>8} {'Overlap':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['model_id']:<20} {r['standard_pearson_r']:>10.3f} "
              f"{'YES' if r['standard_inverted_u'] else 'no':>8} {r['standard_quad_coeff']:>8.3f} "
              f"{'YES' if r['balanced_inverted_u'] else 'no':>8} {r['balanced_quad_coeff']:>8.3f} "
              f"{r['balanced_peak']:>8.2f} {r['top10_overlap']*100:>7.1f}%")

    # Save summary JSON
    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")

    # Combined profile plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for r in all_results:
        model_id = r['model_id']
        nhl, nlayers = MODELS[model_id]
        sts, rtr = load_phiid_matrices(model_id)
        num_heads = sts.shape[0]

        # Standard
        _, _, std_scores = compute_standard_ranking(sts, rtr, num_heads)
        std_layer = [np.mean(std_scores[l*nhl:(l+1)*nhl]) for l in range(nlayers)]
        x = np.linspace(0, 1, nlayers)
        axes[0].plot(x, std_layer, '-o', label=model_id, markersize=3)

        # Balanced
        _, bal_scores = compute_balanced_ranking(sts, rtr, num_heads)
        bal_layer = [np.mean(bal_scores[l*nhl:(l+1)*nhl]) for l in range(nlayers)]
        axes[1].plot(x, bal_layer, '-o', label=model_id, markersize=3)

    axes[0].set_title('Standard Method (rank_diff)')
    axes[0].set_xlabel('Normalized Layer Depth')
    axes[0].set_ylabel('Avg Syn-Red Score')
    axes[0].legend(fontsize=7)

    axes[1].set_title('Balanced Method (sts/(sts+rtr))')
    axes[1].set_xlabel('Normalized Layer Depth')
    axes[1].set_ylabel('Avg Balance Score')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR, 'combined_profiles.png')
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"Saved combined plot to {combined_path}")


if __name__ == '__main__':
    main()
