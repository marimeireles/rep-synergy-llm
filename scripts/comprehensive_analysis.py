#!/usr/bin/env python
"""
Comprehensive analysis of PhiID + ablation results across all models.
Reproducing: "A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning"
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "/lustre07/scratch/marimeir/rep-synergy-llm/results"

MODELS = [
    ("Pythia-1B (standard)",
     "phiid_scores/pythia1b_head_rankings.csv",
     "ablation/pythia1b_ablation.csv"),
    ("Pythia-1B (concat prompts)",
     "phiid_scores/pythia1b_concat_head_rankings.csv",
     "ablation/pythia1b_ablation_concat.csv"),
    ("Pythia-1B (concat balanced)",
     "phiid_scores/pythia1b_concat_head_rankings_balanced.csv",
     None),
    ("Pythia-1B (random weights)",
     "phiid_scores/pythia1b_random_head_rankings.csv",
     None),
    ("Qwen 3 8B (standard)",
     "phiid_scores/qwen3_8b_head_rankings.csv",
     "ablation/qwen3_8b_ablation.csv"),
    ("Qwen 3 8B (balanced)",
     "phiid_scores/qwen3_8b_head_rankings_balanced.csv",
     "ablation/qwen3_8b_ablation_balanced.csv"),
    ("Gemma 3 4B (pretrained)",
     "phiid_scores/gemma3_4b_head_rankings.csv",
     "ablation/gemma3_4b_ablation.csv"),
    ("Gemma 3 4B (balanced)",
     "phiid_scores/gemma3_4b_head_rankings_balanced.csv",
     "ablation/gemma3_4b_ablation_balanced.csv"),
    ("Gemma 3 4B (instruct)",
     "phiid_scores/gemma3_4b_it_head_rankings.csv",
     "ablation/gemma3_4b_it_ablation.csv"),
    ("Gemma 3 12B (instruct)",
     "phiid_scores/gemma3_12b_it_head_rankings.csv",
     "ablation/gemma3_12b_it_ablation.csv"),
]


def load_csv(relpath):
    fullpath = os.path.join(RESULTS_DIR, relpath)
    if not os.path.exists(fullpath):
        return None
    return pd.read_csv(fullpath)


def detect_inverted_u(layer_means):
    n = len(layer_means)
    if n < 3:
        return {"inverted_u_thirds": False, "negative_quadratic": False,
                "quad_coeff": 0, "peak_x_normalized": float('nan'),
                "early_mean": 0, "mid_mean": 0, "late_mean": 0,
                "mid_minus_early": 0, "mid_minus_late": 0}
    
    third = n // 3
    remainder = n - 3 * third
    early_end = third
    mid_end = 2 * third + min(remainder, 1)
    
    early = layer_means[:early_end]
    middle = layer_means[early_end:mid_end]
    late = layer_means[mid_end:]
    
    early_mean = np.mean(early)
    mid_mean = np.mean(middle)
    late_mean = np.mean(late)
    
    x = np.arange(n, dtype=float)
    x_norm = x / (n - 1) if n > 1 else x
    coeffs = np.polyfit(x_norm, layer_means, 2)
    quad_coeff = coeffs[0]
    
    if abs(quad_coeff) > 1e-10:
        peak_x = -coeffs[1] / (2 * coeffs[0])
    else:
        peak_x = float('nan')
    
    is_inverted_u = (mid_mean > early_mean) and (mid_mean > late_mean)
    has_neg_quad = quad_coeff < 0
    
    return {
        "inverted_u_thirds": is_inverted_u,
        "negative_quadratic": has_neg_quad,
        "quad_coeff": quad_coeff,
        "peak_x_normalized": peak_x,
        "early_mean": early_mean,
        "mid_mean": mid_mean,
        "late_mean": late_mean,
        "mid_minus_early": mid_mean - early_mean,
        "mid_minus_late": mid_mean - late_mean,
    }


def analyze_rankings(label, df):
    print(f"\n{'='*70}")
    print(f"  HEAD RANKINGS: {label}")
    print(f"{'='*70}")
    
    print(f"  Columns: {list(df.columns)}")
    print(f"  Num heads: {len(df)}")
    
    num_layers = df['layer'].nunique()
    heads_per_layer = len(df) // num_layers
    print(f"  Layers: {num_layers}, Heads/layer: {heads_per_layer}")
    
    # Determine score column
    score_col = 'syn_red_score'
    if score_col not in df.columns:
        if 'pair_balance_score' in df.columns:
            score_col = 'pair_balance_score'
        else:
            print(f"  WARNING: No score column found. Available: {list(df.columns)}")
            return None
    
    has_syn_red = ('avg_synergy' in df.columns and 'avg_redundancy' in df.columns)
    has_syn_balance = 'syn_balance' in df.columns
    
    # Per-layer average
    layer_means = df.groupby('layer')[score_col].mean().values
    layer_stds = df.groupby('layer')[score_col].std().values
    
    print(f"\n  Per-layer mean {score_col}:")
    print(f"  {'Layer':>5}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}")
    for i, (m, s) in enumerate(zip(layer_means, layer_stds)):
        print(f"  {i:>5}  {m:>8.4f}  {s:>8.4f}")
    
    print(f"\n  Overall: mean={np.mean(layer_means):.4f}, std={np.std(layer_means):.4f}, "
          f"range=[{np.min(layer_means):.4f}, {np.max(layer_means):.4f}]")
    
    # Inverted-U detection
    iu = detect_inverted_u(layer_means)
    print(f"\n  Inverted-U Analysis:")
    print(f"    Thirds test (mid > early AND mid > late): {iu['inverted_u_thirds']}")
    print(f"    Early mean: {iu['early_mean']:.4f}")
    print(f"    Middle mean: {iu['mid_mean']:.4f}")
    print(f"    Late mean: {iu['late_mean']:.4f}")
    print(f"    Mid - Early: {iu['mid_minus_early']:+.4f}")
    print(f"    Mid - Late:  {iu['mid_minus_late']:+.4f}")
    print(f"    Quadratic coefficient: {iu['quad_coeff']:.6f} "
          f"({'negative=inverted-U' if iu['negative_quadratic'] else 'positive=U-shape'})")
    print(f"    Peak location (0=first, 1=last layer): {iu['peak_x_normalized']:.3f}")
    
    # Correlation between synergy and redundancy
    corr_val = None
    if has_syn_red:
        r, p = stats.pearsonr(df['avg_synergy'], df['avg_redundancy'])
        rho, p_rho = stats.spearmanr(df['avg_synergy'], df['avg_redundancy'])
        corr_val = r
        print(f"\n  Synergy-Redundancy correlation (per head):")
        print(f"    Pearson r  = {r:.4f}  (p = {p:.2e})")
        print(f"    Spearman rho = {rho:.4f}  (p = {p_rho:.2e})")
        
        print(f"\n  Per-layer Pearson r(syn, red):")
        for layer_idx in sorted(df['layer'].unique()):
            sub = df[df['layer'] == layer_idx]
            if len(sub) >= 3:
                r_l, p_l = stats.pearsonr(sub['avg_synergy'], sub['avg_redundancy'])
                print(f"    Layer {layer_idx:>2}: r={r_l:+.4f}  (p={p_l:.2e})")
    
    if has_syn_balance:
        print(f"\n  Syn balance [sts/(sts+rtr)] statistics:")
        print(f"    Mean:  {df['syn_balance'].mean():.4f}")
        print(f"    Std:   {df['syn_balance'].std():.4f}")
        print(f"    Range: [{df['syn_balance'].min():.4f}, {df['syn_balance'].max():.4f}]")
        
        layer_bal = df.groupby('layer')['syn_balance'].mean().values
        print(f"    Per-layer mean syn_balance:")
        for i, v in enumerate(layer_bal):
            print(f"      Layer {i:>2}: {v:.4f}")
    
    peak_layer = np.argmax(layer_means)
    trough_layer = np.argmin(layer_means)
    print(f"\n  Peak layer: {peak_layer} (score={layer_means[peak_layer]:.4f})")
    print(f"  Trough layer: {trough_layer} (score={layer_means[trough_layer]:.4f})")
    
    return {
        "label": label,
        "num_heads": len(df),
        "num_layers": num_layers,
        "heads_per_layer": heads_per_layer,
        "inverted_u": iu['inverted_u_thirds'],
        "neg_quad": iu['negative_quadratic'],
        "quad_coeff": iu['quad_coeff'],
        "peak_normalized": iu['peak_x_normalized'],
        "mid_minus_early": iu['mid_minus_early'],
        "mid_minus_late": iu['mid_minus_late'],
        "syn_red_corr": corr_val,
        "score_col": score_col,
        "layer_means": layer_means,
    }


def analyze_ablation(label, df):
    print(f"\n{'='*70}")
    print(f"  ABLATION: {label}")
    print(f"{'='*70}")
    
    print(f"  Columns: {list(df.columns)}")
    order_types = df['order_type'].unique()
    print(f"  Order types: {list(order_types)}")
    
    syn_red_orders = [o for o in order_types if o in ('syn_red', 'pair_balance', 'balanced')]
    random_orders = sorted([o for o in order_types if o.startswith('random')])
    
    if not syn_red_orders:
        print("  WARNING: No syn_red/balanced order found!")
        return None
    
    syn_red_label = syn_red_orders[0]
    syn_red_data = df[df['order_type'] == syn_red_label].sort_values('num_heads_removed').reset_index(drop=True)
    
    total_heads = syn_red_data['num_heads_removed'].max()
    print(f"  Total heads: {total_heads}")
    print(f"  Syn-red order label: '{syn_red_label}'")
    print(f"  Num random orderings: {len(random_orders)}")
    print(f"  Num data points (syn_red): {len(syn_red_data)}")
    
    # Compute random average
    random_merged = None
    if random_orders:
        random_dfs = []
        for ro in random_orders:
            rd = df[df['order_type'] == ro].sort_values('num_heads_removed').reset_index(drop=True)
            random_dfs.append(rd)
        
        random_merged = random_dfs[0][['num_heads_removed']].copy()
        for i, rd in enumerate(random_dfs):
            random_merged = random_merged.merge(
                rd[['num_heads_removed', 'mean_kl_div']].rename(columns={'mean_kl_div': f'kl_{i}'}),
                on='num_heads_removed', how='outer'
            )
        kl_cols = [c for c in random_merged.columns if c.startswith('kl_')]
        random_merged['random_mean'] = random_merged[kl_cols].mean(axis=1)
        random_merged['random_std'] = random_merged[kl_cols].std(axis=1)
        random_merged = random_merged.sort_values('num_heads_removed').reset_index(drop=True)
    
    # AUC computation
    syn_x = syn_red_data['num_heads_removed'].values / total_heads
    syn_y = syn_red_data['mean_kl_div'].values
    auc_syn = np.trapz(syn_y, syn_x)
    
    auc_rand = None
    auc_ratio = None
    if random_merged is not None:
        rand_x = random_merged['num_heads_removed'].values / total_heads
        rand_y = random_merged['random_mean'].values
        auc_rand = np.trapz(rand_y, rand_x)
        auc_ratio = auc_syn / auc_rand if auc_rand > 0 else float('inf')
    
    print(f"\n  AUC (area under KL curve):")
    print(f"    Syn-red order: {auc_syn:.6f}")
    if auc_rand is not None:
        print(f"    Random mean:   {auc_rand:.6f}")
        print(f"    AUC ratio (syn_red / random): {auc_ratio:.4f}")
        if auc_ratio > 1:
            print(f"    >>> SYN-RED CAUSES MORE DAMAGE (ratio > 1) -- matches paper <<<")
        else:
            print(f"    >>> Random causes more damage (ratio < 1) -- CONTRADICTS paper <<<")
    
    # KL at key percentages
    percentages = [0.10, 0.25, 0.50, 0.75]
    print(f"\n  KL divergence at key ablation fractions:")
    print(f"  {'Frac':>6}  {'#Heads':>7}  {'Syn-red KL':>12}  {'Random KL':>12}  {'Ratio(S/R)':>10}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*10}")
    
    kl_ratios_at_pcts = {}
    for pct in percentages:
        target_heads = int(round(pct * total_heads))
        
        syn_closest_idx = np.argmin(np.abs(syn_red_data['num_heads_removed'].values - target_heads))
        syn_closest = syn_red_data.iloc[syn_closest_idx]
        actual_heads = int(syn_closest['num_heads_removed'])
        syn_kl = syn_closest['mean_kl_div']
        
        rand_kl = None
        ratio = None
        if random_merged is not None:
            rand_closest_idx = np.argmin(np.abs(random_merged['num_heads_removed'].values - target_heads))
            rand_closest = random_merged.iloc[rand_closest_idx]
            rand_kl = rand_closest['random_mean']
            ratio = syn_kl / rand_kl if rand_kl > 1e-12 else float('inf')
        
        kl_ratios_at_pcts[pct] = ratio
        
        rand_str = f"{rand_kl:>12.6f}" if rand_kl is not None else f"{'N/A':>12}"
        ratio_str = f"{ratio:>10.3f}" if ratio is not None else f"{'N/A':>10}"
        print(f"  {pct:>5.0%}  {actual_heads:>7}  {syn_kl:>12.6f}  {rand_str}  {ratio_str}")
    
    # Max KL
    max_syn_kl = syn_y[-1] if len(syn_y) > 0 else 0
    max_rand_kl = rand_y[-1] if random_merged is not None and len(rand_y) > 0 else None
    print(f"\n  Max KL (all heads ablated):")
    print(f"    Syn-red: {max_syn_kl:.6f}")
    if max_rand_kl is not None:
        print(f"    Random:  {max_rand_kl:.6f}")
    
    # Crossover analysis
    if random_merged is not None:
        common_heads = np.intersect1d(
            syn_red_data['num_heads_removed'].values,
            random_merged['num_heads_removed'].values
        )
        if len(common_heads) > 1:
            syn_interp = np.interp(common_heads, syn_red_data['num_heads_removed'].values, syn_y)
            rand_interp = np.interp(common_heads, random_merged['num_heads_removed'].values, rand_y)
            diff = syn_interp - rand_interp
            
            crossovers = []
            for i in range(len(diff) - 1):
                if diff[i] * diff[i+1] < 0:
                    frac = common_heads[i] + (common_heads[i+1] - common_heads[i]) * abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
                    crossovers.append(frac / total_heads)
            
            if crossovers:
                print(f"\n  Crossover points (syn_red crosses random):")
                for c in crossovers:
                    print(f"    At {c:.1%} of heads ablated")
            else:
                if np.all(diff >= 0):
                    print(f"\n  Syn-red ALWAYS above random (no crossover) -- matches paper")
                elif np.all(diff <= 0):
                    print(f"\n  Syn-red ALWAYS below random (no crossover) -- CONTRADICTS paper")
                else:
                    print(f"\n  Mixed: syn-red sometimes above, sometimes below random")
            
            quarter = max(1, len(common_heads) // 4)
            early_advantage = np.mean(diff[:quarter])
            late_advantage = np.mean(diff[-quarter:])
            print(f"  Early-ablation advantage (syn-red minus random, first 25%): {early_advantage:+.6f}")
            print(f"  Late-ablation advantage (syn-red minus random, last 25%):   {late_advantage:+.6f}")
    
    return {
        "label": label,
        "total_heads": total_heads,
        "auc_syn": auc_syn,
        "auc_rand": auc_rand,
        "auc_ratio": auc_ratio,
        "kl_ratios": kl_ratios_at_pcts,
        "max_syn_kl": max_syn_kl,
    }


def main():
    print("=" * 70)
    print("  COMPREHENSIVE PhiID + ABLATION ANALYSIS")
    print("  Reproducing: 'A Brain-like Synergistic Core in LLMs'")
    print("=" * 70)
    
    ranking_results = []
    ablation_results = []
    
    for label, rank_path, abl_path in MODELS:
        if rank_path:
            df_rank = load_csv(rank_path)
            if df_rank is not None:
                res = analyze_rankings(label, df_rank)
                if res:
                    ranking_results.append(res)
            else:
                print(f"\n  [SKIP] Rankings not found: {rank_path}")
        
        if abl_path:
            df_abl = load_csv(abl_path)
            if df_abl is not None:
                res = analyze_ablation(label, df_abl)
                if res:
                    ablation_results.append(res)
            else:
                print(f"\n  [SKIP] Ablation not found: {abl_path}")
    
    # ============================================================
    # SUMMARY TABLE: Rankings
    # ============================================================
    print("\n\n")
    print("=" * 100)
    print("  SUMMARY: HEAD RANKINGS ACROSS ALL MODELS")
    print("=" * 100)
    
    print(f"\n  {'Model':<35} {'Heads':>5} {'Lyrs':>4} {'Inv-U':>6} {'NegQ':>5} "
          f"{'QCoeff':>8} {'Peak':>5} {'r(S,R)':>7} {'M-E':>7} {'M-L':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*4} {'-'*6} {'-'*5} {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    
    for r in ranking_results:
        corr_str = f"{r['syn_red_corr']:.3f}" if r['syn_red_corr'] is not None else "N/A"
        print(f"  {r['label']:<35} {r['num_heads']:>5} {r['num_layers']:>4} "
              f"{'YES' if r['inverted_u'] else 'no':>6} "
              f"{'YES' if r['neg_quad'] else 'no':>5} "
              f"{r['quad_coeff']:>8.4f} "
              f"{r['peak_normalized']:>5.2f} "
              f"{corr_str:>7} "
              f"{r['mid_minus_early']:>+7.4f} "
              f"{r['mid_minus_late']:>+7.4f}")
    
    # ============================================================
    # SUMMARY TABLE: Ablation
    # ============================================================
    print("\n\n")
    print("=" * 110)
    print("  SUMMARY: ABLATION RESULTS ACROSS ALL MODELS")
    print("=" * 110)
    
    print(f"\n  {'Model':<35} {'Heads':>5} {'AUC syn':>10} {'AUC rand':>10} {'Ratio':>7} "
          f"{'R@10%':>7} {'R@25%':>7} {'R@50%':>7} {'R@75%':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    
    for a in ablation_results:
        ratios = a['kl_ratios']
        
        def fmt_ratio(r):
            if r is None: return "N/A"
            if r == float('inf'): return "inf"
            return f"{r:.3f}"
        
        auc_rand_str = f"{a['auc_rand']:.6f}" if a['auc_rand'] is not None else "N/A"
        auc_ratio_str = f"{a['auc_ratio']:.3f}" if a['auc_ratio'] is not None else "N/A"
        
        print(f"  {a['label']:<35} {a['total_heads']:>5} {a['auc_syn']:>10.6f} "
              f"{auc_rand_str:>10} {auc_ratio_str:>7} "
              f"{fmt_ratio(ratios.get(0.10)):>7} {fmt_ratio(ratios.get(0.25)):>7} "
              f"{fmt_ratio(ratios.get(0.50)):>7} {fmt_ratio(ratios.get(0.75)):>7}")
    
    # ============================================================
    # INTERPRETATION
    # ============================================================
    print("\n\n")
    print("=" * 90)
    print("  INTERPRETATION")
    print("=" * 90)
    
    print("""
  Paper claims:
    1. Middle layers have higher synergy (inverted-U profile in syn_red_score)
    2. Synergy and redundancy are complementary (not highly correlated)
    3. Ablating synergistic heads first causes MORE behavior divergence than random
       (AUC ratio > 1, KL ratios > 1 at most ablation fractions)
""")
    
    # 1. Inverted-U
    has_iu = [r for r in ranking_results if r['inverted_u']]
    no_iu = [r for r in ranking_results if not r['inverted_u']]
    print(f"  1. INVERTED-U PROFILE:")
    print(f"     Models showing inverted-U (thirds test): {len(has_iu)}/{len(ranking_results)}")
    for r in has_iu:
        print(f"       + {r['label']} (peak@{r['peak_normalized']:.2f}, quad={r['quad_coeff']:.4f})")
    if no_iu:
        print(f"     Models WITHOUT inverted-U:")
        for r in no_iu:
            print(f"       - {r['label']} (peak@{r['peak_normalized']:.2f}, quad={r['quad_coeff']:.4f})")
    
    # 2. Correlation
    corr_models = [r for r in ranking_results if r['syn_red_corr'] is not None]
    if corr_models:
        print(f"\n  2. SYNERGY-REDUNDANCY CORRELATION:")
        print(f"     Paper expects low/negative correlation (as in brain data, r ~ -0.4)")
        for r in corr_models:
            level = 'HIGH positive' if r['syn_red_corr'] > 0.5 else 'moderate' if r['syn_red_corr'] > 0.3 else 'low/negative'
            print(f"       {r['label']}: r = {r['syn_red_corr']:.3f} ({level})")
    
    # 3. Ablation
    if ablation_results:
        print(f"\n  3. ABLATION (syn-red first vs random):")
        print(f"     Paper expects AUC ratio > 1 (synergistic heads cause more damage)")
        above_1 = [a for a in ablation_results if a['auc_ratio'] is not None and a['auc_ratio'] > 1]
        below_1 = [a for a in ablation_results if a['auc_ratio'] is not None and a['auc_ratio'] <= 1]
        print(f"     Models with AUC ratio > 1 (matches paper): {len(above_1)}/{len(ablation_results)}")
        for a in above_1:
            print(f"       + {a['label']}: ratio = {a['auc_ratio']:.3f}")
        if below_1:
            print(f"     Models with AUC ratio <= 1 (CONTRADICTS paper): {len(below_1)}/{len(ablation_results)}")
            for a in below_1:
                print(f"       - {a['label']}: ratio = {a['auc_ratio']:.3f}")
    
    # Best model
    if ablation_results:
        best = max(ablation_results, key=lambda a: a['auc_ratio'] if a['auc_ratio'] is not None else 0)
        print(f"\n     BEST ablation result: {best['label']} (AUC ratio = {best['auc_ratio']:.3f})")
    
    # Overall
    print(f"\n  OVERALL ASSESSMENT:")
    total_iu = len(has_iu)
    total_abl_match = len([a for a in ablation_results if a['auc_ratio'] is not None and a['auc_ratio'] > 1])
    total_high_corr = len([r for r in corr_models if r['syn_red_corr'] > 0.5])
    
    print(f"     Inverted-U profile: {total_iu}/{len(ranking_results)} models")
    print(f"     Ablation match (AUC ratio > 1): {total_abl_match}/{len(ablation_results)} models")
    if total_high_corr > 0:
        print(f"     WARNING: {total_high_corr}/{len(corr_models)} models show high syn-red correlation (r > 0.5)")
        print(f"     This means synergy and redundancy per head are NOT complementary,")
        print(f"     unlike brain data. The standard rank_diff method may not meaningfully")
        print(f"     separate synergistic from redundant heads.")
    
    if total_abl_match > 0 and total_iu > 0:
        print(f"\n     CONCLUSION: PARTIAL REPRODUCTION. Some models/methods show the")
        print(f"     expected patterns, but not consistently across all models.")
    elif total_abl_match == 0 and total_iu == 0:
        print(f"\n     CONCLUSION: FAILED REPRODUCTION. No models show the expected patterns.")
    elif total_abl_match > len(ablation_results) // 2:
        print(f"\n     CONCLUSION: MOSTLY SUCCESSFUL REPRODUCTION.")
    
    # Compare trained vs random baseline
    trained_pythia = [r for r in ranking_results if r['label'] == 'Pythia-1B (standard)']
    random_pythia = [r for r in ranking_results if r['label'] == 'Pythia-1B (random weights)']
    if trained_pythia and random_pythia:
        print(f"\n  4. TRAINED vs RANDOM BASELINE (Pythia-1B):")
        t = trained_pythia[0]
        r = random_pythia[0]
        print(f"     Trained: inverted-U={t['inverted_u']}, quad_coeff={t['quad_coeff']:.4f}, "
              f"layer-score std={np.std(t['layer_means']):.4f}")
        print(f"     Random:  inverted-U={r['inverted_u']}, quad_coeff={r['quad_coeff']:.4f}, "
              f"layer-score std={np.std(r['layer_means']):.4f}")
        print(f"     Paper expects: trained=inverted-U, random=flat")
    
    print("\n" + "=" * 70)
    print("  END OF ANALYSIS")
    print("=" * 70)


if __name__ == "__main__":
    main()
