# CLAUDE.md — Autonomous Execution Plan

## Project: Reproduce "A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning"

Paper: arxiv.org/abs/2601.06851v1

This file instructs Claude Code to autonomously implement and run the full PhiID
analysis + ablation pipeline for Pythia-1B on a GPU server. No user interaction required.

---

## Key Methodology (from the paper — follow exactly)

**Note**: The paper tests Gemma 3 (4B/1B), Llama 3, Qwen 3, DeepSeek V2 Lite, and Pythia-1B.
We reproduce on **Pythia-1B** (`EleutherAI/pythia-1b`) — the paper directly analyzes this model
for training progression (Fig 3a). Fully open, no license needed. 16 layers × 16 heads = 256 heads.
Training checkpoints available on HuggingFace for the emergence analysis.

- **Framework**: PhiID (Integrated Information Decomposition) [Mediano et al., ref 14 in paper], NOT static PID
- **Unit of analysis**: Attention HEADS, not neurons/hidden dims
- **Activations**: L2 norm of per-head attention output: `a(h,t) = ||softmax(Q K^T / sqrt(d_k)) V||_2`
- **Data**: 60 cognitive task prompts (6 categories × 10), generate 100 tokens autoregressively per prompt
- **PhiID library**: `phyid` from github.com/Imperial-MIND-lab/integrated-info-decomp (ref [20] in paper)
- **PhiID atoms**: `sts` (Syn→Syn, temporally persistent synergy) and `rtr` (Red→Red, temporally persistent redundancy)
- **PhiID call**: `calc_PhiID(src, trg, tau=1, kind="gaussian", redundancy="MMI")` — returns LOCAL arrays, must np.mean() them
- **Head pairs**: ALL pairs of attention heads (not random subsets). Pythia-1B: 16 heads × 16 layers = 256 heads → C(256,2) = 32,640 pairs. Average across all 60 prompts per pair.
- **Ranking** (from paper p.4): Per-head avg synergy and redundancy across all pairs involving that head. Rank heads by synergy, rank by redundancy. `syn_red_score = synergy_rank - redundancy_rank`. Min-max normalize to [0,1].
- **Ablation** (from paper p.13-14): Iteratively ablate heads one at a time in decreasing order of syn-red rank (most synergistic first). At each step, measure KL divergence between original and ablated output distributions, conditioned on the non-ablated model's token sequence. Compare against random ordering (5 random seeds, report mean ± std). This is NOT percentage-based — it's cumulative iterative removal.
- **KL formula**: `Behaviour_divergence(q) = (1/T) * sum_t KL(p_original(x_t | x_<t) || p_ablated(x_t | x_<t))` where x_<t is the non-ablated model's generated sequence.
- **Expected result**: Middle layers are synergistic (inverted-U profile), early/late layers are redundant

---

## Server Environment

- **Cluster**: Narval2 (Alliance Canada — docs.alliancecan.ca)
- **User**: `marimeir`
- **Working directory**: `/home/marimeir/scratch/rep-synergy-llm`
- **Python**: 3.10.19 via micromamba
- **Env name**: `syn`
- **Activate**: `micromamba activate syn` (or already active if launched from the env)
- **Python binary**: `~/micromamba/envs/syn/bin/python`
- **MAMBA_ROOT_PREFIX**: `/home/marimeir/micromamba`
- **GPU**: A100-40GB (Narval nodes have 4× A100 per node)

## Execution Mode

- **Autonomy**: Full — do not ask for confirmation, execute each phase sequentially
- **Error handling**: If a step fails, debug and retry up to 3 times before logging failure and moving on
- **All output**: Goes to `results/` directory
- **Python command**: Always use `python` (not `python3`) — the micromamba env provides it
- **Package install**: Use `pip install` within the active `syn` conda env (NOT micromamba install for pip-only packages like phyid/transformers)

---

## Project Structure

```
rep-synergy-llm/
├── configs/
│   └── config.py                 # Simple Python config dict (no Hydra)
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Seeding, device, logging, model loading
│   ├── prompts.py                # The 60 cognitive task prompts
│   ├── activation_extraction.py  # Hook-based per-head activation capture
│   ├── phiid_computation.py      # PhiID wrapper around phyid library
│   ├── head_ranking.py           # Rank heads by syn-red score
│   ├── ablation.py               # Head ablation + KL divergence measurement
│   └── visualization.py          # All plotting functions
├── scripts/
│   ├── 01_extract_activations.py
│   ├── 02_compute_phiid.py
│   ├── 03_rank_heads.py
│   ├── 04_run_ablation.py
│   ├── 05_random_baseline.py
│   └── 06_visualize.py
├── tests/
│   ├── test_phiid.py
│   └── test_ablation.py
├── results/
│   ├── activations/
│   ├── phiid_scores/
│   ├── ablation/
│   └── figures/
├── requirements.txt
└── CLAUDE.md
```

---

## Phase 0: Environment Setup

The `syn` micromamba environment (Python 3.10.19) should already be active.
If not, activate it: `micromamba activate syn`

1. Install dependencies into the existing `syn` env:
```bash
# PyTorch with CUDA (use conda channel for CUDA-aware build on Narval)
micromamba install -y -n syn pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Everything else via pip (within the active conda env)
pip install transformers accelerate
pip install -e /home/marimeir/scratch/integrated-info-decomp  # phyid — already cloned locally
pip install numpy scipy
pip install matplotlib seaborn
pip install tqdm pandas pytest
```

2. Pre-download Pythia-1B (will cache to ~/.cache/huggingface/):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b')
AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
```

3. Create directories: `results/activations/`, `results/phiid_scores/`, `results/ablation/`, `results/figures/`

**Note on Narval**: Compute nodes may not have internet access. Run Phase 0 on a login node first, then submit compute jobs via SLURM. Alternatively, download models to scratch before submitting GPU jobs.

---

## Phase 1: Config, Utils, Prompts, and Activation Extraction

### 1a. Create `configs/config.py`
Simple Python dict — no Hydra/omegaconf:
```python
CONFIG = {
    "model_name": "EleutherAI/pythia-1b",  # paper uses this for Fig 3a (training progression)
    "num_tokens_to_generate": 100,
    "seed": 42,
    "device": "auto",  # auto-detect cuda/cpu
    "results_dir": "results",
    # Generation: greedy decoding (deterministic, reproducible)
    "do_sample": False,       # greedy, not stochastic
    "temperature": 1.0,       # only matters if do_sample=True
    "top_k": 0,
    "top_p": 1.0,
    # Ablation
    "num_random_ablation_seeds": 5,  # paper uses 5 random orderings
}
```

### 1b. Create `src/prompts.py`
All 60 prompts from the paper's Appendix A (Tables 2 and 3). These are the EXACT prompts:

**Syntax and Grammar Correction** (10):
1. "Correct the error: He go to school every day."
2. "Correct the error: She have two cats and a dogs."
3. "Correct the error: I eats breakfast at 8:00 in the morning."
4. "Correct the error: Every students in the classroom has their own laptop."
5. "Correct the error: She don't like going to the park on weekends."
6. "Correct the error: We was happy to see the rainbow after the storm."
7. "Correct the error: There is many reasons to celebrate today."
8. "Correct the error: Him and I went to the market yesterday."
9. "Correct the error: The books is on the table."
10. "Correct the error: They walks to school together every morning."

**Part of Speech Tagging** (10):
1. "Identify the parts of speech in the sentence: Quickly, the agile cat climbed the tall tree."
2. "Identify the parts of speech in the sentence: She whispered a secret to her friend during the boring lecture."
3. "Identify the parts of speech in the sentence: The sun sets in the west."
4. "Identify the parts of speech in the sentence: Can you believe this amazing view?"
5. "Identify the parts of speech in the sentence: He quickly finished his homework."
6. "Identify the parts of speech in the sentence: The beautifully decorated cake was a sight to behold."
7. "Identify the parts of speech in the sentence: They will travel to Japan next month."
8. "Identify the parts of speech in the sentence: My favorite book was lost."
9. "Identify the parts of speech in the sentence: The loud music could be heard from miles away."
10. "Identify the parts of speech in the sentence: She sold all of her paintings at the art show."

**Basic Numerical Reasoning** (10):
1. "If you have 15 apples and you give away 5, how many do you have left?"
2. "A rectangle's length is twice its width. If the rectangle's perimeter is 36 meters, what are its length and width?"
3. "You read 45 pages of a book each day. How many pages will you have read after 7 days?"
4. "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?"
5. "There are 8 slices in a pizza. If you eat 2 slices, what fraction of the pizza is left?"
6. "If one pencil costs 50 cents, how much do 12 pencils cost?"
7. "You have a 2-liter bottle of soda. If you pour out 500 milliliters, how much soda is left?"
8. "A marathon is 42 kilometers long. If you have run 10 kilometers, how much further do you have to run?"
9. "If you divide 24 by 3, then multiply by 2, what is the result?"
10. "A car travels 150 miles on 10 gallons of gas. How many miles per gallon does the car get?"

**Basic Common Sense Reasoning** (10):
1. "If it starts raining while the sun is shining, what weather phenomenon might you expect to see?"
2. "Why do people wear sunglasses?"
3. "What might you use to write on a chalkboard?"
4. "Why would you put a letter in an envelope?"
5. "If you're cold, what might you do to get warm?"
6. "What is the purpose of a refrigerator?"
7. "Why might someone plant a tree?"
8. "What happens to ice when it's left out in the sun?"
9. "Why do people shake hands when they meet?"
10. "What can you use to measure the length of a desk?"

**Abstract Reasoning and Creative Thinking** (10):
1. "Imagine a future where humans have evolved to live underwater. Describe the adaptations they might develop."
2. "Invent a sport that could be played on Mars considering its lower gravity compared to Earth. Describe the rules."
3. "Describe a world where water is scarce, and every drop counts."
4. "Write a story about a child who discovers they can speak to animals."
5. "Imagine a city that floats in the sky. What does it look like, and how do people live?"
6. "Create a dialogue between a human and an alien meeting for the first time."
7. "Design a vehicle that can travel on land, water, and air. Describe its features."
8. "Imagine a new holiday and explain how people celebrate it."
9. "Write a poem about a journey through a desert."
10. "Describe a device that allows you to experience other people's dreams."

**Emotional Intelligence and Social Cognition** (10):
1. "Write a dialogue between two characters where one comforts the other after a loss, demonstrating empathy."
2. "Describe a situation where someone misinterprets a friend's actions as hostile, and how they resolve the misunderstanding."
3. "Compose a letter from a character apologising for a mistake they made."
4. "Describe a scene where a character realizes they are in love."
5. "Write a conversation between two old friends who haven't seen each other in years."
6. "Imagine a character facing a moral dilemma. What do they choose and why?"
7. "Describe a character who is trying to make amends for past actions."
8. "Write about a character who overcomes a fear with the help of a friend."
9. "Create a story about a misunderstanding between characters from different cultures."
10. "Imagine a scenario where a character has to forgive someone who wronged them."

### 1c. Create `src/utils.py`
- `set_seed(seed)` — sets torch, numpy, random seeds
- `get_device()` — returns cuda if available, else cpu
- `load_model_and_tokenizer(model_name)` — loads from HuggingFace, returns model in eval mode on device

### 1d. Create `src/activation_extraction.py`

This is the critical file. Must extract **per-head** attention activations during **autoregressive generation**.

```python
class HeadActivationExtractor:
    """
    Extracts per-attention-head activation norms during autoregressive generation.

    For each attention layer, hooks capture the output of each individual head
    BEFORE the output projection combines them. The activation for head h at
    generation step t is:

        a(h, t) = ||softmax(Q_h K_h^T / sqrt(d_k)) V_h||_2

    This is the L2 norm of the attention-weighted value vector for that head.
    """

    def __init__(self, model, tokenizer):
        # Register hooks on each attention layer to capture per-head outputs
        # Pythia uses GPT-NeoX architecture: self.query_key_value projects to Q,K,V
        # After splitting into heads and computing attention, we need the
        # per-head output BEFORE c_proj (the output projection)
        pass

    def generate_and_extract(self, prompt, num_tokens=100):
        """
        Autoregressively generate num_tokens from prompt.
        At each generation step, record the L2 norm of each head's output.

        Returns:
            activations: np.ndarray of shape (num_heads_total, num_tokens)
                where num_heads_total = num_layers * num_heads_per_layer
                Each value is the L2 norm of that head's attention output at that step.
            generated_tokens: list of token ids generated
        """
        pass

    def extract_all_prompts(self, prompts, num_tokens=100):
        """
        Run generate_and_extract for all prompts.
        Returns:
            all_activations: np.ndarray of shape (num_prompts, num_heads_total, num_tokens)
            all_tokens: list of lists of token ids
        """
        pass
```

**Implementation detail for Pythia hooks**: Pythia uses the GPT-NeoX architecture (`GPTNeoXAttention`). The attention computation uses `self.query_key_value` (fused QKV projection), then splits into heads. We need to hook to capture per-head outputs BEFORE the output projection (`self.dense`) merges them. The cleanest approach:

1. Hook into each `GPTNeoXAttention` module
2. Inside the hook, intercept after the attention-weighted values are computed but before `self.dense` merges heads
3. For each head, compute `||attn_output[:, :, head_idx, :]||_2` (L2 norm across head_dim)
4. Since we generate token-by-token, at each step the sequence length for the NEW token is 1 (with KV cache), so we take the last position

**Pythia-1B architecture reference**:
- 16 layers, 16 heads per layer = 256 total heads
- Hidden dim = 2048, head dim = 128
- Attention module path: `model.gpt_neox.layers[i].attention`
- Uses rotary position embeddings (RoPE), but this doesn't affect our L2 norm extraction

### 1e. Create `scripts/01_extract_activations.py`
- Load Pythia-1B (`EleutherAI/pythia-1b`)
- Load all 60 prompts from `src/prompts.py`
- For each prompt: generate 100 tokens, record per-head L2 norms at each step
- Save activations to `results/activations/pythia1b_activations.npz`
  - Shape: `(60, 256, 100)` — (prompts, total_heads, generation_steps)
- Also save generated token sequences (needed for ablation KL computation)
- Also save original logits at each generation step (needed for ablation KL computation)
  - Shape: `(60, 100, vocab_size)` — these are the p_original distributions
- **Run this script and verify output shapes**

### 1f. Verification
- Load saved activations, assert shape is `(60, 256, 100)`
- Check values are finite, positive (L2 norms), non-zero
- Print min/max/mean per layer to sanity check

---

## Phase 2: PhiID Computation

### 2a. Create `src/phiid_computation.py`

Wrapper around the `phyid` library:

```python
from phyid.calculate import calc_PhiID

def compute_pairwise_phiid(activations, head_i, head_j, tau=1):
    """
    Compute PhiID between two attention heads' activation time series.

    Args:
        activations: np.ndarray of shape (num_prompts, num_heads, num_steps)
        head_i, head_j: indices of the two heads
        tau: time lag (default 1)

    For each prompt:
        src = activations[prompt, head_i, :]  # shape (100,) — 1D time series
        trg = activations[prompt, head_j, :]  # shape (100,) — 1D time series
        atoms, _ = calc_PhiID(src, trg, tau=tau, kind="gaussian", redundancy="MMI")

        IMPORTANT: Each atom value (atoms['sts'], atoms['rtr'], etc.) is a
        LOCAL array of shape (N-tau,), NOT a scalar. You must take np.mean()
        to get the average atom value for that prompt:
            sts_for_prompt = np.mean(atoms['sts'])
            rtr_for_prompt = np.mean(atoms['rtr'])

    Average the mean sts and mean rtr across all 60 prompts.

    Returns:
        dict with keys: 'sts' (float, grand mean synergy), 'rtr' (float, grand mean redundancy)
    """
    pass

def compute_all_pairs_phiid(activations, num_heads, tau=1):
    """
    Compute PhiID for ALL pairs of attention heads.

    Pythia-1B: 256 heads → C(256,2) = 32,640 pairs
    Each pair averaged over 60 prompts.

    Returns:
        pair_results: dict mapping (i,j) -> {'sts': float, 'rtr': float}
    """
    pass
```

**Critical implementation detail from phyid source code**:
- `calc_PhiID(src, trg, tau)` takes two 1D arrays and an integer tau
- It internally constructs: `[src_past, trg_past, src_future, trg_future]` where past=`[:-tau]`, future=`[tau:]`
- For `kind="gaussian"`, it standardizes to unit variance internally — no pre-normalization needed
- Return: `atoms_res` is a dict where each value is an array of shape `(N-tau,)` (local/per-sample values), NOT a scalar. **You must `np.mean(atoms['sts'])` to get the average.**
- The 16 atoms are: `rtr, rtx, rty, rts, xtr, xtx, xty, xts, ytr, ytx, yty, yts, str, stx, sty, sts`
- We only need `sts` (Syn→Syn) and `rtr` (Red→Red)
- Only depends on numpy + scipy — purely CPU-bound, no GPU needed for PhiID
- **VERIFY at test time**: The return type may be arrays (local values) or scalars depending on mode. `np.mean()` is safe either way (returns scalar on scalar input). But check the actual return shape in tests and log it.

### 2b. Create `tests/test_phiid.py`
- Test that `calc_PhiID` runs without error on synthetic time series (two 1D arrays, tau=1)
- Test that return values are dicts with correct keys (`sts`, `rtr`, etc.)
- Test that each atom value is an array (not scalar) and np.mean() gives a finite number
- Test with correlated signals (should have higher mean rtr than sts)
- Test with more complex signals (verify sts and rtr are both non-negative after averaging)
- Reference: phyid's own tests download .mat data from OSF — our tests should use synthetic data only (no network needed on compute nodes)

### 2c. Create `scripts/02_compute_phiid.py`
- Load activations from Phase 1
- Compute PhiID for all 32,640 head pairs (averaged over 60 prompts each)
- Save pairwise results to `results/phiid_scores/pythia1b_pairwise_phiid.npz`
- Log progress (this will take a while — ~10K PhiID computations)
- **Run tests first, then run this script**

**Computational note**: 32,640 pairs × 60 prompts = 1,958,400 individual `calc_PhiID` calls. Each is fast (Gaussian closed-form, numpy only, CPU-bound — no GPU needed for this phase). Use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor` to parallelize across pairs. Narval nodes have many CPU cores. Estimate ~1-4 hours depending on parallelism.

---

## Phase 3: Head Ranking

### 3a. Create `src/head_ranking.py`

```python
def compute_head_scores(pair_results, num_heads):
    """
    For each head, compute its average synergy and redundancy
    across all pairs that include it.

    For head h:
        avg_synergy[h] = mean of sts for all pairs (h, j) and (i, h)
        avg_redundancy[h] = mean of rtr for all pairs (h, j) and (i, h)

    Returns:
        synergy_per_head: np.ndarray of shape (num_heads,)
        redundancy_per_head: np.ndarray of shape (num_heads,)
    """
    pass

def compute_syn_red_rank(synergy_per_head, redundancy_per_head):
    """
    Paper's ranking method:
    1. Rank all heads by synergy (ordinal rank, 1 = lowest synergy)
    2. Rank all heads by redundancy (ordinal rank, 1 = lowest redundancy)
    3. syn_red_score = synergy_rank - redundancy_rank
    4. Min-max normalize: (score - min) / (max - min) → range [0, 1]

    High score = more synergistic, Low score = more redundant.

    Returns:
        syn_red_rank: np.ndarray of shape (num_heads,) in [0, 1]
        head_layer_map: list mapping head index to (layer, head_within_layer)
    """
    pass

def get_head_layer_mapping(num_layers, num_heads_per_layer):
    """
    Map flat head index to (layer_idx, head_within_layer_idx).
    E.g., for Pythia-1B: head 0-15 → layer 0, head 16-31 → layer 1, etc.
    """
    pass
```

### 3b. Create `scripts/03_rank_heads.py`
- Load pairwise PhiID results
- Compute per-head synergy and redundancy averages
- Compute syn-red rank scores
- Save to `results/phiid_scores/pythia1b_head_rankings.csv`
  - Columns: head_idx, layer, head_in_layer, avg_synergy, avg_redundancy, syn_rank, red_rank, syn_red_score
- **Run this script**
- Print per-layer average syn_red_score → should show inverted-U (middle layers higher)

---

## Phase 4: Ablation Experiments

### 4a. Create `src/ablation.py`

```python
class HeadAblationEngine:
    """
    Ablates attention heads and measures behavior divergence via KL divergence.

    Key methodological detail from paper:
    - The ablated model is conditioned on the NON-ABLATED model's token sequence
    - We compare output distributions, not generated text
    - KL(P_original || P_ablated) at each generation step
    """

    def __init__(self, model, tokenizer, device):
        pass

    def zero_out_heads(self, head_indices):
        """
        Register forward hooks that zero out the specified attention heads.
        head_indices: list of (layer_idx, head_within_layer_idx) tuples
        """
        pass

    def compute_behaviour_divergence(self, prompt_tokens, original_token_sequence,
                                      original_logits, head_indices_to_ablate):
        """
        From paper p.13-14:

        1. Use the pre-computed original model's generated token sequence x_<t
           and its logits (probability distributions) at each step
        2. Register hooks to zero out the specified attention heads
        3. Feed the SAME token sequence to the ablated model at each step
        4. At each step t, compute:
           KL(p_original(x_t | x_<t) || p_ablated(x_t | x_<t))
           where p_original and p_ablated are full vocabulary distributions
        5. Average KL over all T generation steps

        Returns: float — behaviour divergence for this prompt
        """
        pass

    def run_iterative_ablation(self, prompts, original_tokens_per_prompt,
                               original_logits_per_prompt, head_rankings, order='syn_red'):
        """
        Iteratively remove heads ONE AT A TIME in decreasing order of syn_red_score.

        Paper method (p.13-14):
        - Start with all heads active
        - At step k, ablate the k-th highest syn-red ranked head (cumulative)
        - For each prompt q:
            - Feed the non-ablated model's token sequence to the ablated model
            - At each generation step t, compute KL(p_original || p_ablated)
            - Average KL over all T tokens = behaviour_divergence(q)
        - Average behaviour_divergence across all prompts
        - Record (num_heads_removed, mean_behaviour_divergence)

        For random baseline: repeat with 5 different random orderings, report mean ± std.

        Args:
            order: 'syn_red' (most synergistic first) or 'random'

        Returns:
            DataFrame with columns: [num_heads_removed, mean_kl_div, std_kl_div, order_type]
        """
        pass
```

### 4b. Create `scripts/04_run_ablation.py`
- Load model, head rankings, and original generated token sequences + logits
- Run iterative ablation in syn-red rank order (most synergistic first)
- Run iterative ablation in random order (5 random seeds, report mean ± std)
- Save to `results/ablation/pythia1b_ablation.csv`
- **Run this script**

### 4c. Verification (matches paper Fig. 4a)
- Synergistic-first ablation (solid line) should cause KL divergence to rise MUCH faster than random (dashed line with shaded std)
- At 0 heads removed, KL should be 0
- X-axis: fraction of nodes deactivated, Y-axis: KL divergence

---

## Phase 5: Random Baseline Comparison

### 5a. Create `scripts/05_random_baseline.py`
- Initialize Pythia-1B with random weights: load architecture config, initialize fresh model with `AutoModelForCausalLM.from_config(config)`
- Run the same pipeline: generate 100 tokens per prompt, extract per-head activations
- Compute PhiID for all head pairs
- Compute head rankings
- Save to `results/phiid_scores/pythia1b_random_phiid.npz` and `pythia1b_random_head_rankings.csv`
- **Run this script**

### 5b. Expected result
- Random model should show FLAT syn-red profile across layers (no inverted-U)
- This confirms the synergistic core emerges from training, not architecture

---

## Phase 6: Visualization

### 6a. Create `src/visualization.py`

```python
def plot_phiid_profile(head_rankings_df, title, save_path):
    """
    Reproduces paper Fig. 2c:
    X-axis: normalized layer depth (0 to 1)
    Y-axis: average syn_red_score per layer (min-max normalized)
    Expected: inverted-U shape (middle layers more synergistic)
    """

def plot_synergy_redundancy_heatmaps(pairwise_results, num_heads, title, save_path):
    """
    Reproduces paper Fig. 2a:
    Two heatmaps (synergy matrix and redundancy matrix) showing
    pairwise values between all attention heads. Axes are source/target head indices.
    """

def plot_head_ranking_heatmap(head_rankings_df, num_layers, num_heads_per_layer, title, save_path):
    """
    Reproduces paper Fig. 2b:
    Heatmap: layers × heads_per_layer, colored by syn_red_score.
    Red = synergistic, Blue = redundant.
    """

def plot_ablation_curves(ablation_df, title, save_path):
    """
    Reproduces paper Fig. 4a:
    X-axis: fraction of nodes deactivated
    Y-axis: KL divergence (behaviour divergence)
    Solid line: synergistic order. Dashed line: random order with shaded std region.
    """

def plot_trained_vs_random(trained_df, random_df, title, save_path):
    """
    Reproduces paper Fig. 3a concept:
    Overlaid PhiID profiles. Trained: inverted-U. Random: flat/absent pattern.
    """
```

### 6b. Create `scripts/06_visualize.py`
- Load all results from previous phases
- Generate all four figure types
- Save to `results/figures/`
- **Run this script**

---

## Execution Sequence

```bash
# Phase 0: Setup (run on login node — needs internet)
micromamba activate syn
micromamba install -y -n syn pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers accelerate numpy scipy matplotlib seaborn tqdm pandas pytest
pip install -e /home/marimeir/scratch/integrated-info-decomp
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b'); AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')"

# Phase 1: Extract activations
python scripts/01_extract_activations.py

# Phase 2: Compute PhiID (HEAVY — hours)
python -m pytest tests/test_phiid.py -v
python scripts/02_compute_phiid.py

# Phase 3: Rank heads
python scripts/03_rank_heads.py

# Phase 4: Ablation experiments (HEAVY)
python scripts/04_run_ablation.py

# Phase 5: Random baseline
python scripts/05_random_baseline.py

# Phase 6: Visualize
python scripts/06_visualize.py
```

---

## Git Discipline

**Commit after each completed phase.** Do not wait until the end.
- After creating all source files for a phase: `git add src/... scripts/... && git commit -m "phase N: description"`
- After a successful script run with verified output: `git commit -m "phase N: verified — description of results"`
- After fixing bugs: `git commit -m "fix: description of what was wrong"`
- Stage specific files — never use `git add .` or `git add -A`
- Never commit files in `results/` (gitignored)

---

## Implementation Notes

- **Use Pythia-1B** (`EleutherAI/pythia-1b`, 16 layers, 16 heads = 256 total). This is the model the paper analyzes for Fig 3a.
- **Device**: Always use CUDA if available. All torch ops under `torch.no_grad()`.
- **Progress logging**: Print timestamps and progress bars for long-running computations.
- **Idempotency**: Check if output files exist before rerunning. Skip completed phases.
- **Error recovery**: Wrap each phase in try/except. Log to `results/errors.log`.
- **PhiID computation parallelism**: Use Python `multiprocessing` or `concurrent.futures` to parallelize across head pairs. The phyid Gaussian method is CPU-bound and embarrassingly parallel.
- **Memory**: Per-head activations for Pythia-1B are tiny: 60 × 256 × 100 × 4 bytes ≈ 6.1 MB.
- **KV cache**: When generating tokens, use HuggingFace's `model.generate()` with `use_cache=True` for efficiency. But we need hooks to fire at each step, so may need manual generation loop.

## Narval-Specific Notes

- **Scratch directory**: Use `/home/marimeir/scratch/` for all large outputs. Scratch is purged after 60 days of inactivity.
- **Internet on compute nodes**: Narval compute nodes may NOT have internet. All model downloads and pip installs must happen on the login node before submitting jobs.
- **SLURM**: If running interactively via `salloc`, all scripts can be called directly. For batch jobs, wrap in sbatch scripts.
- **GPU allocation**: Request with `--gres=gpu:1` for single-GPU jobs. A100-40GB is more than enough for Pythia-1B (~4GB model).
- **Module system**: Narval uses `module load` for system software, but we use micromamba so this is generally not needed. If CUDA issues arise, try `module load cuda/12.1`.
- **Troubleshooting**: If anything weird happens (CUDA errors, SLURM issues, filesystem quirks), consult https://docs.alliancecan.ca/wiki/Narval/en

## Permission Notes for Claude Code on Server

Claude Code needs these permissions in `.claude/settings.json`:
- `Bash(pip:*)` — install packages
- `Bash(python:*)` — run scripts
- `Bash(python -m pytest:*)` — run tests
- `Bash(micromamba:*)` — manage conda env
- `Bash(mkdir:*)` — create directories
- `Bash(nvidia-smi:*)` — check GPU
- All file read/write/edit/glob/grep tools
