# Investigation Plan: Fixing PhiID Results

## Problem Summary
Our syn-red rankings are noise (r=0.90 correlation between per-head synergy & redundancy).
Ablation curves overlap. PhiID profile doesn't show inverted-U. Need to find the root cause.

## Root Cause Analysis (COMPLETED)

**Core issue**: Per-head synergy and redundancy are highly positively correlated:
- Pythia-1B: Spearman r=0.90, Pearson r=0.82
- Qwen 3 8B: Spearman r=0.87

**Why**: Both sts and rtr scale with total TDMI (Time-Delayed Mutual Information).
Heads with higher lag-1 autocorrelation have higher TDMI with all other heads,
inflating both synergy and redundancy proportionally. This autocorrelation is
almost perfectly correlated with per-head synergy (r=0.97 for both models).

**Autocorrelation itself has an inverted-U shape** (middle layers ~0.5, early/late ~0.2
for Pythia; similar pattern for Qwen). This is the signal we want to capture,
but the rank-difference method (rank(syn) - rank(red)) cancels it out because
both syn and red track autocorrelation equally.

**In brain data** (Luppi et al. 2022), syn-red correlation is NEGATIVE (r≈-0.4),
so the rank-difference method works directly. In LLMs, the positive correlation
requires normalization before the ranking step.

---

## Experiments and Results

### Experiment 1: Verify activation extraction ✅ PASSED
Hook captures correct per-head L2 norms. Values match manual computation
within RoPE-expected differences (0.04-0.39 on values of 2-7).

### Experiment 2: Check Luppi et al. methodology ✅ COMPLETED
Luppi uses raw sts/rtr values → node strength → rank difference. In brain data
this works because syn-red correlation is negative. No per-pair normalization
in the original brain paper. However, in LLMs the positive correlation breaks this.

### Experiment 3: Alternative atom definitions ⏭️ SKIPPED (too slow)
Killed due to OMP thread exhaustion and excessive runtime.

### Experiment 4: Discrete PhiID ⏭️ SKIPPED
Not attempted due to runtime concerns.

### Experiment 5: Z-scored time series ✅ NO EFFECT
Z-scoring is a no-op for Gaussian PhiID because the library already normalizes
by std (line 193 of calculate.py), and np.cov subtracts the mean internally.
Verified: Raw and z-scored give identical sts/rtr values.

### Experiment 6: Concatenated prompts ✅ HELPS
Concatenating 60 prompts × 100 tokens = 6000-point time series per pair:
- Reduces per-head syn-red correlation from 0.90 to 0.64 (Pythia)
- Reduces pairwise-level correlation from 0.74 to 0.26
- Only 1 PhiID call per pair (vs 60), so ~60x faster
- Produces mild inverted-U for Pythia, but still noisy

### Additional: TDMI normalization ✅ HELPS
Normalizing sts and rtr by total TDMI (sum of all 16 atoms) per pair:
- Reduces per-head syn-red correlation from 0.87 to 0.51
- Tested on 500-pair subset, promising

### KEY FINDING: Per-pair balance sts/(sts+rtr) ✅ BEST APPROACH
Computing sts/(sts+rtr) per pair before averaging per head:
- **Qwen 3 8B**: Produces CLEAR inverted-U profile matching paper Figure 2c
  - Layers 0-1: ~0.47 (low, redundant)
  - Layers 18-21: ~0.72-0.74 (peak, synergistic)
  - Layers 33-35: ~0.50 (low, redundant)
- **Pythia-1B**: Peaks at layers 2-3 instead of middle layers (less clear)
  - Might be genuine: Pythia-1B is a simpler model with weaker synergistic core
  - Paper's Fig 3a shows Pythia's inverted-U is noisier than larger models

---

## Solution: Updated Ranking Method

**For large models (Qwen, Gemma, Llama)**: Use per-pair balance `sts/(sts+rtr)`
- Compute balance(i,j) = sts(i,j) / (sts(i,j) + rtr(i,j)) for each pair
- Per-head score = mean of balance across all pairs involving that head
- Min-max normalize to [0, 1]

**For Pythia-1B**: Use concatenated PhiID (6000pts) + rank difference
- Better covariance estimation reduces the syn-red correlation
- Rank difference then captures a mild inverted-U signal

## Files Created
- `src/head_ranking.py` — updated with `compute_pair_balance_scores()` method
- `results/phiid_scores/qwen3_8b_head_rankings_balanced.csv` — Qwen rankings
- `results/phiid_scores/pythia1b_concat_phiid.npz` — Pythia concatenated PhiID
- `results/phiid_scores/pythia1b_concat_head_rankings.csv` — Pythia concat rankings
- `scripts/rerun_ablation_balanced.py` — ablation script with flexible ranking input
- `scripts/slurm_rerun_qwen_ablation.sh` — SLURM job for Qwen ablation
- `scripts/slurm_rerun_pythia_ablation.sh` — SLURM job for Pythia ablation

---

## Ablation Results with Improved Rankings (COMPLETED)

SLURM jobs ran successfully on A100 GPUs:
- Pythia-1B: Job 56525751, step_size=1, 5 random seeds (all complete)
- Qwen 3 8B: Job 56525753, step_size=16, 3 of 5 random seeds completed before timeout

### Pythia-1B Ablation (Concatenated PhiID + Rank Difference)

| Heads Removed | Fraction | syn_red KL | Random Mean KL | syn_red / random |
|:---:|:---:|:---:|:---:|:---:|
| 16  | 12.5% | 0.136 | 0.291 | 0.47x |
| 32  | 25.0% | 0.284 | 0.609 | 0.47x |
| 48  | 37.5% | 0.630 | 1.302 | 0.48x |
| 64  | 50.0% | 1.061 | 1.639 | 0.65x |
| 80  | 62.5% | 2.017 | 2.428 | 0.83x |
| 96  | 75.0% | 3.717 | 3.179 | 1.17x |
| 112 | 87.5% | 3.992 | 4.001 | 1.00x |
| 128 | 100%  | 4.600 | 4.600 | 1.00x |

**Result**: Synergistic ordering causes LESS damage than random at all early checkpoints
(0-62.5%). syn_red is above random at only 1/9 interpolation points. Mean diff = -0.24.
This is the **OPPOSITE** of the paper's prediction.

### Qwen 3 8B Ablation (Per-Pair Balance)

| Heads Removed | Fraction | syn_red KL | Random Mean KL | syn_red / random |
|:---:|:---:|:---:|:---:|:---:|
| 144  | 12.5% | 0.249 | 0.225 | 1.11x |
| 288  | 25.0% | 0.670 | 0.462 | 1.45x |
| 432  | 37.5% | 1.001 | 0.834 | 1.20x |
| 576  | 50.0% | 1.541 | 1.648 | 0.94x |
| 720  | 62.5% | 2.779 | 3.944 | 0.70x |
| 864  | 75.0% | 5.374 | 6.781 | 0.79x |
| 1024 | 88.9% | 7.607 | 10.811 | 0.70x |
| 1152 | 100%  | 15.042 | 15.042 | 1.00x |

**Result**: Two-phase pattern. syn_red causes MORE damage in the 8-39% range (matching
paper partially), but LESS damage from 40-100%. syn_red above random at 25/73 points.
Overall AUC ratio = 0.86, meaning synergistic ordering causes 14% less total damage.

### Key Observation

Neither model reproduces the paper's Figure 4a where synergistic-first ablation
causes a persistently steeper KL curve than random. Instead:

1. **Pythia-1B**: The most synergistic heads (by our ranking) are the LEAST
   critical for behavior. Removing them first causes ~50% less damage at 12.5-37.5%.

2. **Qwen 3 8B**: Partial match in the 8-39% range, but the redundant heads
   (removed last in synergistic ordering) turn out to be much more critical,
   causing the random curve to rise faster in the 40-100% range.

### Possible Explanations

1. **PhiID profile ≠ behavioral importance**: The inverted-U profile captures
   an information-theoretic structural property of the representations, but this
   doesn't necessarily translate to causal importance for output behavior. Synergistic
   heads may perform higher-order integration that has redundant fallbacks.

2. **Implementation differences**: The paper may use a subtly different ablation
   approach, ranking method, or activation extraction that we haven't identified.
   The paper includes Mediano (phyid creator) and Luppi as co-authors.

3. **Model-specific effects**: The paper primarily shows results for Gemma 3 4B.
   Pythia-1B may be too small. Qwen 3 8B may have a different attention structure
   (GQA with 8 KV heads) that affects the PhiID analysis.

4. **Missing methodological detail**: The paper may apply additional normalization,
   filtering, or post-processing steps not described in the methods section.

---

## Final Summary

### What worked:
- **Per-pair balance normalization** produces a clear inverted-U PhiID profile
  for Qwen 3 8B, matching the paper's Figure 2c
- **Concatenated PhiID** (6000-point time series) improves covariance estimation
  and reduces syn-red correlation
- **Root cause identified**: positive syn-red correlation in LLMs (vs negative in
  brain data) breaks the simple rank-difference method
- **Pipeline infrastructure** works correctly (activation extraction, PhiID
  computation, ablation engine, visualization)

### What didn't work:
- **Ablation curves** do not reproduce Figure 4a — synergistic-first removal does
  NOT cause faster KL divergence rise than random ordering
- **Pythia-1B** results are the opposite of the paper's prediction
- **Qwen 3 8B** shows a partial match only in the first third of ablation

### Files Created/Modified
- `scripts/plot_ablation_from_logs.py` — plot ablation curves from SLURM logs
- `scripts/rank_heads_balanced.py` — compute per-pair balance for Pythia
- `results/phiid_scores/pythia1b_concat_head_rankings_balanced.csv` — Pythia balanced rankings
- `results/ablation/pythia1b_ablation_concat.csv` — full Pythia ablation data (step_size=1)
- `results/figures/ablation_curves_improved.png` — side-by-side ablation plots

### Visualizations
- `results/figures/ablation_curves_improved.png` — Fig 4a comparison
- `results/figures/qwen3_8b_phiid_profile_balanced.png` — Qwen inverted-U profile
- `results/figures/pythia1b_phiid_profile_concat.png` — Pythia PhiID profile
- `results/figures/combined_phiid_profiles_improved.png` — both models overlaid
