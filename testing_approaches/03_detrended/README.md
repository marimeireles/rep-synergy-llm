# Approach 03: Detrended PhiID

## What's different here

Previous approaches (01, 02) used a "balanced" metric `sts/(sts+rtr)` per pair
as a workaround for the positive syn-red correlation. This is **not** what the
paper does. The paper uses **rank_diff**: rank heads by synergy, rank by
redundancy, subtract ranks, min-max normalize to [0,1].

This experiment drops the balanced metric entirely and tests whether
**detrending the activation time series** fixes the underlying problem so that
the paper's rank_diff method works on its own.

## Motivation

Our LLM activation time series show a clear downward drift during generation
(~40% decrease over 100 tokens). This non-stationarity inflates TDMI equally
for sts and rtr, creating a strong positive syn-red correlation (r=0.82-0.90)
that makes rank_diff produce noise. Brain fMRI papers (Luppi 2022, ref [8])
detrend before PhiID — the LLM paper doesn't mention it.

## Methods tested

| Method | Description |
|--------|-------------|
| `raw` | No preprocessing (baseline, same as original pipeline) |
| `linear` | `scipy.signal.detrend(type='linear')` per prompt per head |
| `first_diff` | `x[t] - x[t-1]` — measures changes, removes trend + autocorrelation |
| `poly2` | Remove quadratic (degree-2) polynomial trend per prompt per head |

## What stays the same as the paper

- PhiID atoms: `sts` (Syn→Syn) and `rtr` (Red→Red)
- Per-prompt computation: PhiID on each 100-step series, averaged across 60 prompts
- All pairs: C(272,2) = 36,856 pairs for Gemma 4B (no sampling)
- Per-head averaging: mean of all pairs involving each head
- Ranking: `rank(synergy) - rank(redundancy)`, min-max normalized (rank_diff)
- PhiID params: `tau=1, kind="gaussian", redundancy="MMI"`

## How to run

```bash
cd /home/marimeir/scratch/rep-synergy-llm
mkdir -p slurm/logs
sbatch slurm/detrended_phiid_gemma3_4b_it.sh
```

Or directly (needs 32 CPU cores):
```bash
python scripts/test_detrended_phiid.py \
    --model gemma3-4b-it \
    --max-workers 32 \
    --methods raw linear first_diff poly2
```

## Key metrics to check

1. **Pairwise syn-red correlation** — does detrending reduce it? (target: low or negative)
2. **Per-head syn-red correlation** — same question at head level
3. **Inverted-U correlation** — does the layer profile match a quadratic peaking at mid-layers?
4. **Lag-1 autocorrelation** — how much does each method reduce it?
