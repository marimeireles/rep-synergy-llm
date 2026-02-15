#!/usr/bin/env python
"""
Generate all missing figures for individual models + combined multi-model figures.

Generates:
  Per-model (missing ones):
    - Gemma balanced: profile, heatmap, ablation
    - Pythia balanced (concat): profile, heatmap

  Combined multi-model:
    - All 3 models PhiID profiles (standard rank_diff)
    - All 3 models PhiID profiles (balanced sts/(sts+rtr))
    - All 3 models ablation curves (standard) — subplots
    - All 3 models ablation curves (balanced where available) — subplots
    - All 3 models ablation ratio curves (syn/random ratio vs fraction)
    - Summary bar chart: AUC ratios across models and methods
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS = "results"
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── Data loading ─────────────────────────────────────────────────────────
def load_rankings(path):
    return pd.read_csv(path)

def load_ablation(path):
    return pd.read_csv(path)

# All ranking files
RANKINGS = {
    "Pythia-1B": load_rankings(f"{RESULTS}/phiid_scores/pythia1b_head_rankings.csv"),
    "Qwen 3 8B": load_rankings(f"{RESULTS}/phiid_scores/qwen3_8b_head_rankings.csv"),
    "Gemma 3 4B": load_rankings(f"{RESULTS}/phiid_scores/gemma3_4b_head_rankings.csv"),
}

RANKINGS_BALANCED = {
    "Pythia-1B": load_rankings(f"{RESULTS}/phiid_scores/pythia1b_concat_head_rankings_balanced.csv"),
    "Qwen 3 8B": load_rankings(f"{RESULTS}/phiid_scores/qwen3_8b_head_rankings_balanced.csv"),
    "Gemma 3 4B": load_rankings(f"{RESULTS}/phiid_scores/gemma3_4b_head_rankings_balanced.csv"),
}

ABLATIONS = {
    "Pythia-1B": load_ablation(f"{RESULTS}/ablation/pythia1b_ablation.csv"),
    "Qwen 3 8B": load_ablation(f"{RESULTS}/ablation/qwen3_8b_ablation.csv"),
    "Gemma 3 4B": load_ablation(f"{RESULTS}/ablation/gemma3_4b_ablation.csv"),
}

ABLATIONS_BALANCED = {
    "Qwen 3 8B": load_ablation(f"{RESULTS}/ablation/qwen3_8b_ablation_balanced.csv"),
    "Gemma 3 4B": load_ablation(f"{RESULTS}/ablation/gemma3_4b_ablation_balanced.csv"),
}

MODEL_INFO = {
    "Pythia-1B":  {"layers": 16, "heads": 8,  "total": 128,  "color": "#1f77b4", "marker": "o"},
    "Qwen 3 8B":  {"layers": 36, "heads": 32, "total": 1152, "color": "#2ca02c", "marker": "^"},
    "Gemma 3 4B": {"layers": 34, "heads": 8,  "total": 272,  "color": "#d62728", "marker": "s"},
}

# ── Helper functions ─────────────────────────────────────────────────────
def get_layer_profile(df, score_col='syn_red_score'):
    """Return (normalized_x, layer_means)."""
    layer_means = df.groupby('layer')[score_col].mean()
    num_layers = len(layer_means)
    x = np.arange(num_layers) / (num_layers - 1)
    return x, layer_means.values

def get_ablation_curves(ablation_df):
    """Return (frac, syn_kl, rand_mean, rand_std)."""
    total = ablation_df['num_heads_removed'].max()
    syn = ablation_df[ablation_df['order_type'] == 'syn_red'].sort_values('num_heads_removed')
    rand = ablation_df[ablation_df['order_type'].str.startswith('random')]

    frac_syn = syn['num_heads_removed'].values / total
    kl_syn = syn['mean_kl_div'].values

    if not rand.empty:
        rand_g = rand.groupby('num_heads_removed')['mean_kl_div']
        frac_rand = np.array(sorted(rand['num_heads_removed'].unique())) / total
        kl_rand_mean = rand_g.mean().values
        kl_rand_std = rand_g.std().fillna(0).values
    else:
        frac_rand = frac_syn
        kl_rand_mean = np.zeros_like(kl_syn)
        kl_rand_std = np.zeros_like(kl_syn)

    return frac_syn, kl_syn, frac_rand, kl_rand_mean, kl_rand_std

def compute_auc(frac, kl):
    """Trapezoidal AUC."""
    return np.trapz(kl, frac)

# ── 1. Individual missing plots: Gemma balanced ─────────────────────────

# Gemma balanced PhiID profile
print("Generating Gemma balanced profile...")
df = RANKINGS_BALANCED["Gemma 3 4B"]
x, y = get_layer_profile(df)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, 'o-', color='darkred', linewidth=2, markersize=6)
ax.set_xlabel('Normalized Layer Depth', fontsize=12)
ax.set_ylabel('Average Syn-Red Score', fontsize=12)
ax.set_title('Gemma 3 4B: Syn-Red Score by Layer (Balanced)', fontsize=14)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGURES}/gemma3_4b_phiid_profile_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# Gemma balanced heatmap
print("Generating Gemma balanced heatmap...")
grid = np.zeros((34, 8))
for _, row in df.iterrows():
    grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']
fig, ax = plt.subplots(figsize=(10, 8))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel('Head Index', fontsize=12)
ax.set_ylabel('Layer', fontsize=12)
ax.set_title('Gemma 3 4B: Syn-Red Score per Head (Balanced)', fontsize=14)
ax.set_xticks(range(8))
ax.set_yticks(range(34))
fig.colorbar(im, ax=ax, label='Syn-Red Score')
fig.tight_layout()
fig.savefig(f"{FIGURES}/gemma3_4b_head_ranking_heatmap_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# Gemma balanced ablation
print("Generating Gemma balanced ablation curves...")
abl = ABLATIONS_BALANCED["Gemma 3 4B"]
frac_s, kl_s, frac_r, kl_rm, kl_rs = get_ablation_curves(abl)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(frac_s, kl_s, '-', color='red', linewidth=2, label='Synergistic order')
ax.plot(frac_r, kl_rm, '--', color='gray', linewidth=2, label='Random order')
ax.fill_between(frac_r, kl_rm - kl_rs, kl_rm + kl_rs, color='gray', alpha=0.2)
ax.set_xlabel('Fraction of Heads Deactivated', fontsize=12)
ax.set_ylabel('Behaviour Divergence (KL)', fontsize=12)
ax.set_title('Gemma 3 4B: Ablation — Balanced Ranking', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGURES}/gemma3_4b_ablation_curves_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 2. Individual: Gemma standard vs balanced comparison ─────────────────
print("Generating Gemma standard vs balanced comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Profile comparison
for method, rankings, ls in [("Standard (rank_diff)", RANKINGS["Gemma 3 4B"], '-'),
                              ("Balanced (sts/(sts+rtr))", RANKINGS_BALANCED["Gemma 3 4B"], '--')]:
    x, y = get_layer_profile(rankings)
    axes[0].plot(x, y, f'o{ls}', linewidth=2, markersize=5, label=method)
axes[0].set_xlabel('Normalized Layer Depth', fontsize=12)
axes[0].set_ylabel('Average Syn-Red Score', fontsize=12)
axes[0].set_title('Gemma 3 4B: PhiID Profile Comparison', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].set_xlim(-0.05, 1.05)
axes[0].grid(True, alpha=0.3)

# Ablation comparison
for method, abl_df, color in [("Standard", ABLATIONS["Gemma 3 4B"], '#1f77b4'),
                                ("Balanced", ABLATIONS_BALANCED["Gemma 3 4B"], '#d62728')]:
    fs, ks, fr, krm, krs = get_ablation_curves(abl_df)
    axes[1].plot(fs, ks, '-', color=color, linewidth=2, label=f'{method} (syn)')
    axes[1].plot(fr, krm, '--', color=color, linewidth=1.5, alpha=0.6, label=f'{method} (random)')
    axes[1].fill_between(fr, krm - krs, krm + krs, color=color, alpha=0.1)
axes[1].set_xlabel('Fraction of Heads Deactivated', fontsize=12)
axes[1].set_ylabel('Behaviour Divergence (KL)', fontsize=12)
axes[1].set_title('Gemma 3 4B: Ablation Comparison', fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Gemma 3 4B: Standard vs Balanced Methods', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGURES}/gemma3_4b_standard_vs_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 3. Pythia balanced profile + heatmap ─────────────────────────────────
print("Generating Pythia balanced profile...")
df_pyth = RANKINGS_BALANCED["Pythia-1B"]
x, y = get_layer_profile(df_pyth)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, 'o-', color='darkred', linewidth=2, markersize=6)
ax.set_xlabel('Normalized Layer Depth', fontsize=12)
ax.set_ylabel('Average Syn-Red Score', fontsize=12)
ax.set_title('Pythia-1B: Syn-Red Score by Layer (Balanced)', fontsize=14)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGURES}/pythia1b_phiid_profile_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("Generating Pythia balanced heatmap...")
grid = np.zeros((16, 8))
for _, row in df_pyth.iterrows():
    grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']
fig, ax = plt.subplots(figsize=(10, 6))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=1)
ax.set_xlabel('Head Index', fontsize=12)
ax.set_ylabel('Layer', fontsize=12)
ax.set_title('Pythia-1B: Syn-Red Score per Head (Balanced)', fontsize=14)
ax.set_xticks(range(8))
ax.set_yticks(range(16))
fig.colorbar(im, ax=ax, label='Syn-Red Score')
fig.tight_layout()
fig.savefig(f"{FIGURES}/pythia1b_head_ranking_heatmap_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 4. Combined: PhiID profiles — all 3 models (standard) ───────────────
print("Generating combined profile (standard)...")
fig, ax = plt.subplots(figsize=(10, 6))
for name, df in RANKINGS.items():
    x, y = get_layer_profile(df)
    info = MODEL_INFO[name]
    ax.plot(x, y, f"{info['marker']}-", color=info['color'], linewidth=2,
            markersize=5, label=f"{name} ({info['total']} heads)", alpha=0.9)
ax.set_xlabel('Normalized Layer Depth', fontsize=12)
ax.set_ylabel('Average Syn-Red Score', fontsize=12)
ax.set_title('PhiID Syn-Red Profile by Layer — All Models (Standard Rank-Diff)', fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_profiles_standard.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 5. Combined: PhiID profiles — all 3 models (balanced) ───────────────
print("Generating combined profile (balanced)...")
fig, ax = plt.subplots(figsize=(10, 6))
for name, df in RANKINGS_BALANCED.items():
    x, y = get_layer_profile(df)
    info = MODEL_INFO[name]
    ax.plot(x, y, f"{info['marker']}-", color=info['color'], linewidth=2,
            markersize=5, label=f"{name} ({info['total']} heads)", alpha=0.9)
ax.set_xlabel('Normalized Layer Depth', fontsize=12)
ax.set_ylabel('Average Syn-Red Score', fontsize=12)
ax.set_title('PhiID Syn-Red Profile by Layer — All Models (Balanced sts/(sts+rtr))', fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_profiles_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 6. Combined: profiles side-by-side (standard vs balanced) ────────────
print("Generating combined profile comparison (std vs balanced)...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

for ax, title_sfx, rank_dict in [
    (axes[0], "Standard (Rank-Diff)", RANKINGS),
    (axes[1], "Balanced (sts/(sts+rtr))", RANKINGS_BALANCED),
]:
    for name, df in rank_dict.items():
        x, y = get_layer_profile(df)
        info = MODEL_INFO[name]
        ax.plot(x, y, f"{info['marker']}-", color=info['color'], linewidth=2,
                markersize=5, label=name, alpha=0.9)
    ax.set_xlabel('Normalized Layer Depth', fontsize=12)
    ax.set_ylabel('Average Syn-Red Score', fontsize=12)
    ax.set_title(title_sfx, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

fig.suptitle('PhiID Syn-Red Layer Profiles — All Models', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_profiles_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 7. Combined ablation: 3 subplots (one per model) standard ───────────
print("Generating combined ablation subplots (standard)...")
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for ax, (name, abl_df) in zip(axes, ABLATIONS.items()):
    info = MODEL_INFO[name]
    fs, ks, fr, krm, krs = get_ablation_curves(abl_df)
    ax.plot(fs, ks, '-', color=info['color'], linewidth=2, label='Synergistic order')
    ax.plot(fr, krm, '--', color='gray', linewidth=2, label='Random order')
    ax.fill_between(fr, krm - krs, krm + krs, color='gray', alpha=0.2)

    auc_syn = compute_auc(fs, ks)
    auc_rand = compute_auc(fr, krm)
    ratio = auc_syn / auc_rand if auc_rand > 0 else float('inf')

    ax.set_xlabel('Fraction Deactivated', fontsize=11)
    ax.set_ylabel('Behaviour Divergence (KL)', fontsize=11)
    ax.set_title(f'{name} ({info["total"]} heads)\nAUC ratio: {ratio:.2f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Ablation Curves — Standard Rank-Diff Ranking', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_ablation_standard.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 8. Combined ablation: balanced (Qwen + Gemma only) ──────────────────
print("Generating combined ablation subplots (balanced)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (name, abl_df) in zip(axes, ABLATIONS_BALANCED.items()):
    info = MODEL_INFO[name]
    fs, ks, fr, krm, krs = get_ablation_curves(abl_df)
    ax.plot(fs, ks, '-', color=info['color'], linewidth=2, label='Synergistic order')
    ax.plot(fr, krm, '--', color='gray', linewidth=2, label='Random order')
    ax.fill_between(fr, krm - krs, krm + krs, color='gray', alpha=0.2)

    auc_syn = compute_auc(fs, ks)
    auc_rand = compute_auc(fr, krm)
    ratio = auc_syn / auc_rand if auc_rand > 0 else float('inf')

    ax.set_xlabel('Fraction Deactivated', fontsize=11)
    ax.set_ylabel('Behaviour Divergence (KL)', fontsize=11)
    ax.set_title(f'{name} ({info["total"]} heads)\nAUC ratio: {ratio:.2f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Ablation Curves — Balanced sts/(sts+rtr) Ranking', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_ablation_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 9. Ablation ratio curves (syn KL / random KL vs fraction) ───────────
print("Generating ablation ratio curves...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

# Standard
ax = axes[0]
for name, abl_df in ABLATIONS.items():
    info = MODEL_INFO[name]
    fs, ks, fr, krm, krs = get_ablation_curves(abl_df)
    # Interpolate random onto syn fractions
    krm_interp = np.interp(fs, fr, krm)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(krm_interp > 0.01, ks / krm_interp, np.nan)
    ax.plot(fs, ratio, f"{info['marker']}-", color=info['color'], linewidth=2,
            markersize=4, label=name, alpha=0.9)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Equal damage')
ax.set_xlabel('Fraction Deactivated', fontsize=11)
ax.set_ylabel('KL Ratio (Synergistic / Random)', fontsize=11)
ax.set_title('Standard Rank-Diff', fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 4)
ax.grid(True, alpha=0.3)

# Balanced
ax = axes[1]
for name, abl_df in ABLATIONS_BALANCED.items():
    info = MODEL_INFO[name]
    fs, ks, fr, krm, krs = get_ablation_curves(abl_df)
    krm_interp = np.interp(fs, fr, krm)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(krm_interp > 0.01, ks / krm_interp, np.nan)
    ax.plot(fs, ratio, f"{info['marker']}-", color=info['color'], linewidth=2,
            markersize=4, label=name, alpha=0.9)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Equal damage')
ax.set_xlabel('Fraction Deactivated', fontsize=11)
ax.set_ylabel('KL Ratio (Synergistic / Random)', fontsize=11)
ax.set_title('Balanced sts/(sts+rtr)', fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 4)
ax.grid(True, alpha=0.3)

fig.suptitle('Ablation KL Ratio: Synergistic vs Random Head Removal', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_ablation_ratio.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 10. Summary bar chart: AUC ratios ───────────────────────────────────
print("Generating AUC ratio summary bar chart...")

auc_data = []
for name, abl_df in ABLATIONS.items():
    fs, ks, fr, krm, krs = get_ablation_curves(abl_df)
    auc_syn = compute_auc(fs, ks)
    auc_rand = compute_auc(fr, krm)
    ratio = auc_syn / auc_rand if auc_rand > 0 else 0
    auc_data.append({"Model": name, "Method": "Standard", "AUC Ratio": ratio})

for name, abl_df in ABLATIONS_BALANCED.items():
    fs, ks, fr, krm, krs = get_ablation_curves(abl_df)
    auc_syn = compute_auc(fs, ks)
    auc_rand = compute_auc(fr, krm)
    ratio = auc_syn / auc_rand if auc_rand > 0 else 0
    auc_data.append({"Model": name, "Method": "Balanced", "AUC Ratio": ratio})

auc_df = pd.DataFrame(auc_data)

fig, ax = plt.subplots(figsize=(10, 5))
models_order = ["Pythia-1B", "Qwen 3 8B", "Gemma 3 4B"]
methods = ["Standard", "Balanced"]
x = np.arange(len(models_order))
width = 0.35

for i, method in enumerate(methods):
    vals = []
    for m in models_order:
        sub = auc_df[(auc_df['Model'] == m) & (auc_df['Method'] == method)]
        vals.append(sub['AUC Ratio'].values[0] if len(sub) > 0 else 0)
    bars = ax.bar(x + i * width - width / 2, vals, width, label=method,
                  color=['#4a86c8', '#d64545'][i], alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No difference (1.0)')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('AUC Ratio (Synergistic / Random)', fontsize=12)
ax.set_title('Ablation AUC Ratio Summary\n(>1 means synergistic heads cause more damage)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_order, fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.5)
fig.tight_layout()
fig.savefig(f"{FIGURES}/auc_ratio_summary.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 11. Grand combined: all 3 models, standard, on single ablation plot ──
print("Generating single-panel combined ablation (standard)...")
fig, ax = plt.subplots(figsize=(10, 6))
for name, abl_df in ABLATIONS.items():
    info = MODEL_INFO[name]
    fs, ks, fr, krm, krs = get_ablation_curves(abl_df)
    # Normalize KL by max to make models comparable on same axis
    kl_max = max(ks.max(), krm.max())
    ax.plot(fs, ks / kl_max, '-', color=info['color'], linewidth=2,
            label=f'{name} (syn)', alpha=0.9)
    ax.plot(fr, krm / kl_max, '--', color=info['color'], linewidth=1.5,
            label=f'{name} (random)', alpha=0.5)
ax.set_xlabel('Fraction of Heads Deactivated', fontsize=12)
ax.set_ylabel('Normalized KL Divergence', fontsize=12)
ax.set_title('All Models: Normalized Ablation Curves (Standard)', fontsize=14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_ablation_normalized.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 12. Head ranking heatmaps: all 3 models side by side (standard) ─────
print("Generating combined heatmaps (standard)...")
fig, axes = plt.subplots(1, 3, figsize=(22, 8),
                          gridspec_kw={'width_ratios': [8, 32, 8]})
cmap = sns.diverging_palette(240, 10, as_cmap=True)

for ax, (name, df) in zip(axes, RANKINGS.items()):
    info = MODEL_INFO[name]
    nl, nh = info['layers'], info['heads']
    grid = np.zeros((nl, nh))
    for _, row in df.iterrows():
        grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']
    im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel('Head', fontsize=10)
    ax.set_ylabel('Layer', fontsize=10)
    ax.set_title(f'{name}\n({nl}L x {nh}H)', fontsize=11)
    ax.set_xticks(range(0, nh, max(1, nh // 8)))
    ax.set_yticks(range(0, nl, max(1, nl // 8)))

fig.colorbar(im, ax=axes, label='Syn-Red Score', fraction=0.02, pad=0.04)
fig.suptitle('Head Ranking Heatmaps — Standard Rank-Diff', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_heatmaps_standard.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 13. Head ranking heatmaps: all 3 models side by side (balanced) ─────
print("Generating combined heatmaps (balanced)...")
fig, axes = plt.subplots(1, 3, figsize=(22, 8),
                          gridspec_kw={'width_ratios': [8, 32, 8]})

for ax, (name, df) in zip(axes, RANKINGS_BALANCED.items()):
    info = MODEL_INFO[name]
    nl, nh = info['layers'], info['heads']
    grid = np.zeros((nl, nh))
    for _, row in df.iterrows():
        grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']
    im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel('Head', fontsize=10)
    ax.set_ylabel('Layer', fontsize=10)
    ax.set_title(f'{name}\n({nl}L x {nh}H)', fontsize=11)
    ax.set_xticks(range(0, nh, max(1, nh // 8)))
    ax.set_yticks(range(0, nl, max(1, nl // 8)))

fig.colorbar(im, ax=axes, label='Syn-Red Score', fraction=0.02, pad=0.04)
fig.suptitle('Head Ranking Heatmaps — Balanced sts/(sts+rtr)', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_heatmaps_balanced.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 14. Smoothed profiles with confidence bands ─────────────────────────
print("Generating smoothed profiles with per-layer spread...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax, title_sfx, rank_dict in [
    (axes[0], "Standard (Rank-Diff)", RANKINGS),
    (axes[1], "Balanced (sts/(sts+rtr))", RANKINGS_BALANCED),
]:
    for name, df in rank_dict.items():
        info = MODEL_INFO[name]
        layer_g = df.groupby('layer')['syn_red_score']
        means = layer_g.mean()
        stds = layer_g.std()
        nl = len(means)
        x = np.arange(nl) / (nl - 1)
        ax.plot(x, means.values, f"{info['marker']}-", color=info['color'],
                linewidth=2, markersize=4, label=name, alpha=0.9)
        ax.fill_between(x, (means - stds).values, (means + stds).values,
                        color=info['color'], alpha=0.1)
    ax.set_xlabel('Normalized Layer Depth', fontsize=12)
    ax.set_ylabel('Avg Syn-Red Score (+/- std)', fontsize=12)
    ax.set_title(title_sfx, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

fig.suptitle('PhiID Layer Profiles with Per-Layer Spread', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGURES}/combined_profiles_with_spread.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n=== All figures generated! ===")
print(f"Output directory: {FIGURES}/")
for f in sorted(os.listdir(FIGURES)):
    if f.endswith('.png'):
        print(f"  {f}")
