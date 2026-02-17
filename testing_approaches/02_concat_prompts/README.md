# Approach 02: Concatenated Prompts for PhiID

## Problem
Our standard pipeline computes PhiID separately for each of the 60 prompts (100 timesteps each), then averages the resulting sts and rtr values across prompts. This produces:
- **Positive** syn-red correlation (r=0.5-0.84) instead of the negative correlation (r~-0.4) seen in brain data
- **Flat** syn-red layer profile instead of the inverted-U shape reported in the paper

The root cause is **high lag-1 autocorrelation (~0.97)** in the per-prompt activation time series. With only 100 timesteps, PhiID can't distinguish real synergy/redundancy structure from variance scaling — both sts and rtr just track overall coupling magnitude.

## What we're doing differently
Instead of computing PhiID per prompt and averaging:
- **Concatenate** all 60 prompts' activation time series into one long series per head: 60 x 100 = **6000 timesteps**
- Run PhiID **once** per head pair on the 6000-point series

This should help because:
1. **Breaks autocorrelation**: At prompt boundaries, there's a discontinuity that disrupts the smooth lag-1 autocorrelation. Measured effect: pairwise correlation drops from 0.74 to 0.26.
2. **More statistical power**: 6000 points gives PhiID ~60x more data to estimate the covariance structure, allowing it to better separate synergistic from redundant interactions.
3. **Matches ambiguous paper wording**: The paper says "average synergy and redundancy across time and prompts" — concatenation followed by temporal averaging is one valid interpretation.

## How to run
```bash
# CPU-only job (no GPU needed)
sbatch slurm/concat_phiid_gemma3_4b_it.sh

# Or directly:
python scripts/test_concat_phiid.py --model gemma3-4b-it --max-workers 32
```

## What to look for in results
- **Pairwise syn-red correlation**: Should decrease from ~0.8 toward 0 or negative
- **Per-head syn-red correlation**: Same — closer to 0 or negative
- **Inverted-U correlation**: Positive r with quadratic template means middle layers are more synergistic
- **Layer profile plot**: Should peak around normalized depth 0.4-0.6

## Output files
All output goes to `testing_approaches/02_concat_prompts/`:
- `gemma3_4b_it_concat_pairwise_phiid.npz` — full pairwise sts/rtr matrices
- `gemma3_4b_it_head_rankings_standard.csv` — rankings using rank_diff method
- `gemma3_4b_it_head_rankings_balanced.csv` — rankings using sts/(sts+rtr) method
- `gemma3_4b_it_concat_phiid_analysis.png` — 3-panel diagnostic figure

## Comparison with Approach 01
| Aspect | 01: Per-pair balance | 02: Concat prompts |
|--------|---------------------|-------------------|
| Changes PhiID input? | No | Yes (6000 pts) |
| Changes ranking? | Yes (sts/(sts+rtr)) | Tests both methods |
| Addresses autocorrelation? | Indirectly (ratio) | Directly (breaks it) |
| Recomputes PhiID? | No | Yes (CPU, ~2 hrs) |
