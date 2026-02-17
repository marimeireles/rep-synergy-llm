# Testing Approaches

Systematic evaluation of methods to address the positive synergy-redundancy correlation
problem discovered during reproduction of "A Brain-like Synergistic Core in LLMs."

## The Problem

In brain data, per-region synergy and redundancy are *negatively* correlated (r ~ -0.4),
meaning regions specialize as either synergistic or redundant. In LLMs, we find a strong
*positive* correlation (r = 0.5-0.84): heads with high synergy also have high redundancy.
This is driven by overall coupling magnitude — strongly connected head pairs score high
on all PhiID atoms. The paper's rank_diff method (syn_rank - red_rank) is noisy under
this condition.

## Approaches

### 01: Per-Pair Balance Normalization [COMPLETED — no recomputation needed]
- **Method**: `sts/(sts+rtr)` per pair before averaging per head
- **Rationale**: Removes scale effect; measures synergy *fraction* not magnitude
- **Data needed**: Existing pairwise PhiID matrices only
- **Status**: Done. Results in `01_per_pair_balance/`

### 02: Different Time Series Representation [SKIPPED — requires full GPU re-extraction]
- **Method**: Use full attention vector (not L2 norm), attention entropy, or residual stream
- **Rationale**: Scalar L2 norm discards directional information
- **Data needed**: New activation extraction (GPU) + new PhiID computation (CPU)
- **Status**: Not run. Too expensive for a hypothesis test.

### 03a: Larger Tau Values [TODO — CPU only, uses existing activations]
- **Method**: Re-run PhiID with tau=2, 3, 5 instead of tau=1
- **Rationale**: Larger lag reduces autocorrelation influence on PhiID atoms
- **Data needed**: Existing activations + new PhiID computation (CPU, hours)

### 03b: Concatenated Prompts [TODO — CPU only, uses existing activations]
- **Method**: Concatenate all 60 prompts into one long time series per head (6000 points)
- **Rationale**: Reduces autocorrelation; tested on Pythia (reduced r from 0.84 to 0.54)
- **Data needed**: Existing activations + new PhiID computation (CPU, hours)

### 03c: Shuffled Temporal Order [TODO — CPU only, uses existing activations]
- **Method**: Randomly shuffle time steps before computing PhiID (destroys temporal structure)
- **Rationale**: Control experiment — if results are similar, temporal structure doesn't matter
- **Data needed**: Existing activations + new PhiID computation (CPU, hours)

### 04: Non-Gaussian PhiID (Discrete) [TODO — CPU only, uses existing activations]
- **Method**: `kind="discrete"` instead of `kind="gaussian"` in calc_PhiID
- **Rationale**: Captures nonlinear dependencies; may separate syn/red differently
- **Data needed**: Existing activations + new PhiID computation (CPU, slower than Gaussian)

### 05: Different Redundancy Measure (CCS) [TODO — CPU only, uses existing activations]
- **Method**: `redundancy="CCS"` instead of `redundancy="MMI"` in calc_PhiID
- **Rationale**: Different mathematical definition of redundancy changes the decomposition
- **Data needed**: Existing activations + new PhiID computation (CPU, hours)

### 06: Group-Level Analysis [TODO — CPU only, uses existing activations]
- **Method**: Compute PhiID between groups of heads (e.g., all heads in layer i vs layer j)
- **Rationale**: Reduces noise from individual head pairs; fewer pairs = faster computation
- **Data needed**: Existing activations + new PhiID computation (CPU, fast — few pairs)

### 07: PID Instead of PhiID [SKIPPED — needs new code/library]
- **Method**: Partial Information Decomposition with output logits as target
- **Rationale**: Directly measures synergistic/redundant contribution to predictions
- **Data needed**: Existing activations + logits + new PID code

## Models Analyzed

| Model | Heads | Layers x Heads/Layer |
|-------|-------|---------------------|
| Pythia-1B | 128 | 16 x 8 |
| Qwen 3 8B | 1152 | 36 x 32 |
| Gemma 3 4B (pretrained) | 272 | 34 x 8 |
| Gemma 3 4B (instruct) | 272 | 34 x 8 |
| Gemma 3 12B (instruct) | 768 | 48 x 16 |
