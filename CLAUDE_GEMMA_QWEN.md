# CLAUDE_GEMMA_QWEN.md — Multi-Model Experiments (Gemma 3 4B + Qwen 3 8B)

Paper: arxiv.org/abs/2601.06851v1

Instructions for reproducing figures on **Gemma 3 4B** and **Qwen 3 8B**.
Builds on top of the existing Pythia-1B pipeline.

---

## Target Figures

| Figure | Description | Status | Code Location |
|--------|------------|--------|---------------|
| **2a** | Pairwise synergy/redundancy heatmaps | READY | `src/visualization.py:plot_synergy_redundancy_heatmaps` |
| **2b** | Per-head syn-red ranking heatmap (layers x heads) | READY | `src/visualization.py:plot_head_ranking_heatmap` |
| **3b** | Graph representation of synergistic/redundant cores (top 10% connections) | READY | `src/graph_analysis.py` + `src/visualization.py:plot_graph_cores` |
| **4a** | Ablation KL divergence curves (syn order vs random) | READY | `src/visualization.py:plot_ablation_curves` |
| **4b** | MATH benchmark accuracy under Gaussian noise perturbation | READY | `src/perturbation.py` + `src/math_eval.py` + `src/visualization.py:plot_math_perturbation` |

---

## Model Specifications

| | Gemma 3 4B | Qwen 3 8B |
|---|---|---|
| **HuggingFace ID** | `google/gemma-3-4b-pt` | `Qwen/Qwen3-8B` |
| **Config key** | `gemma3-4b` | `qwen3-8b` |
| **Layers** | 26 | 36 |
| **Attention heads/layer** | 8 (4 KV heads, GQA) | 32 (8 KV heads, GQA) |
| **Total heads** | 208 | 1,152 |
| **Head dim** | 256 | 128 |
| **Hidden dim** | 2560 | 4096 |
| **C(n,2) pairs** | 21,528 | 663,176 |
| **dtype** | float16 | float16 |
| **Gated?** | YES (needs HF token) | No |
| **VRAM** | ~8 GB (fp16) | ~16 GB (fp16) |

**Unit of analysis**: Query heads (not KV heads). In GQA models, each query head produces its own attention pattern and weighted output, even when KV heads are shared. This is what the paper analyzes.

---

## Phase 0: Pre-Requisites (Login Node — needs internet)

### 0a. HuggingFace Token for Gemma

Gemma 3 is a gated model. You need:
1. Accept the license at https://huggingface.co/google/gemma-3-4b-pt
2. Create `.env` file in the project root:
   ```bash
   cp .env.example .env
   # Edit .env and add your token:
   # HF_TOKEN=hf_your_actual_token_here
   ```
3. Alternatively, set `HF_TOKEN` in your shell before running.

### 0b. Pre-Cache Models (MUST be done on login node)

Compute nodes have NO internet. All model weights must be cached first.

```bash
# On the login node:
micromamba activate syn

# Gemma 3 4B (needs HF_TOKEN)
export HF_TOKEN=hf_your_token_here
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Downloading Gemma 3 4B...')
AutoModelForCausalLM.from_pretrained('google/gemma-3-4b-pt', torch_dtype=torch.float16)
AutoTokenizer.from_pretrained('google/gemma-3-4b-pt')
print('Done.')
"

# Qwen 3 8B (public, no token needed)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Downloading Qwen 3 8B...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', torch_dtype=torch.float16)
AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
print('Done.')
"
```

### 0c. Install Extra Dependencies (for Fig 3b and 4b)

```bash
pip install networkx python-louvain   # Fig 3b: graph analysis
pip install datasets                   # Fig 4b: MATH benchmark
```

### 0d. Pre-Cache MATH Dataset (for Fig 4b)

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('hendrycks/competition_math', split='test')
print(f'MATH test set: {len(ds)} problems')
"
```

---

## Phase 1-4, 6: Running the Existing Pipeline (Fig 2a, 2b, 4a)

The unified pipeline script handles everything. For each model:

### Interactive (salloc)

```bash
# Request a GPU node
salloc --account=def-zhijing_gpu --gres=gpu:1 --cpus-per-task=32 --mem=48G --time=12:00:00

# Activate environment
micromamba activate syn
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Source HF token
source .env 2>/dev/null || true

# --- Gemma 3 4B ---
python scripts/run_pipeline.py --model gemma3-4b --phases 1 2 3 4 6 --max-workers 32

# --- Qwen 3 8B ---
# NOTE: Qwen has 1,152 heads = 663K pairs. Phase 2 alone takes ~24 hours.
# Phase 4 uses step_size=8 automatically (measuring KL every 8 heads removed).
# Total: ~30-40 hours. Use SLURM batch instead for this one.
python scripts/run_pipeline.py --model qwen3-8b --phases 1 2 3 4 6 --max-workers 32
```

### SLURM Batch (recommended for Qwen)

```bash
# Create logs directory
mkdir -p slurm/logs

# Submit jobs
sbatch slurm/gemma3.sh    # ~6-10 hrs, 1 GPU
sbatch slurm/qwen3.sh     # ~30-40 hrs, 1 GPU (has checkpoint/resume for PhiID)
```

**Note**: The SLURM scripts currently use `--account=def-bhrett`. Change to `def-zhijing_gpu` if needed:
```bash
sed -i 's/def-bhrett/def-zhijing_gpu/' slurm/gemma3.sh slurm/qwen3.sh
```

### What Each Phase Produces

| Phase | Output | Gemma Time | Qwen Time |
|-------|--------|------------|-----------|
| 1 | `results/activations/{model}_activations.npz` + `logits.npz` + `tokens.json` | ~10 min | ~15 min |
| 2 | `results/phiid_scores/{model}_pairwise_phiid.npz` | ~80 min | ~24 hrs |
| 3 | `results/phiid_scores/{model}_head_rankings.csv` | seconds | seconds |
| 4 | `results/ablation/{model}_ablation.csv` | ~4-8 hrs | ~4-8 hrs |
| 6 | `results/figures/{model}_*.png` (profile, heatmaps, ablation) | seconds | seconds |

Phases are **idempotent** — if output files exist, they're skipped. Delete output files to force re-run.

### Phase 6 Generates These Figures

For each model, phase 6 creates:
- `{model}_phiid_profile.png` — Fig 2c (inverted-U layer profile)
- `{model}_synergy_redundancy.png` — **Fig 2a** (pairwise heatmaps)
- `{model}_head_ranking_heatmap.png` — **Fig 2b** (layers x heads grid)
- `{model}_ablation_curves.png` — **Fig 4a** (syn-red vs random ablation)

### Multi-Model Comparison (after both models are done)

```bash
python scripts/08_compare_models.py
```

Generates:
- `results/figures/multi_model_profiles.png` — overlaid Fig 2c for all models
- `results/figures/multi_model_ablation.png` — overlaid Fig 4a for all models

---

## Phase 7: Figure 3b — Graph Representation of Synergistic/Redundant Cores

**NEEDS IMPLEMENTATION**

### What the Paper Shows (Fig 3b)

The paper represents the synergy and redundancy matrices as **undirected weighted graphs**:
- Each attention head is a **node**
- Edge weight between heads i and j = pairwise synergy (or redundancy)
- Only the **top 10% strongest connections** are shown for clarity
- Synergistic core graph and redundant core graph shown side by side
- Paper does this for Gemma 3 1B Instruct, but we do it for Gemma 3 4B and Qwen 3 8B

### What Fig 3c Shows (bonus)

Graph-theoretic metrics across models:
- **Global efficiency**: inverse of average shortest path length — high in synergistic core = integrated processing
- **Modularity**: community structure measure — high in redundant core = compartmentalized processing

### Implementation Plan

#### 7a. New function in `src/visualization.py`: `plot_graph_cores`

```python
import networkx as nx
from community import community_louvain  # python-louvain package

def plot_graph_cores(sts_matrix, rtr_matrix, num_layers, num_heads_per_layer,
                     top_pct=0.1, title_prefix="", save_path=None):
    """
    Fig 3b: Visualize synergistic and redundant cores as undirected graphs.

    Args:
        sts_matrix: (N, N) pairwise synergy matrix
        rtr_matrix: (N, N) pairwise redundancy matrix
        num_layers: number of layers
        num_heads_per_layer: heads per layer
        top_pct: fraction of strongest edges to keep (paper uses 0.1 = top 10%)
        title_prefix: model name for title
        save_path: output file path
    """
    # For each matrix (sts, rtr):
    #   1. Create a graph from the upper triangle (undirected)
    #   2. Threshold to keep only top 10% edges by weight
    #   3. Color nodes by layer (gradient from early=light to late=dark)
    #   4. Layout: spring layout or spectral layout
    #   5. Draw with matplotlib
    pass
```

#### 7b. New function: `compute_graph_metrics`

```python
def compute_graph_metrics(sts_matrix, rtr_matrix, top_pct=0.1):
    """
    Fig 3c: Compute graph-theoretic properties of synergy and redundancy networks.

    Returns dict with:
        - 'syn_global_efficiency': float
        - 'red_global_efficiency': float
        - 'syn_modularity': float (Louvain community detection)
        - 'red_modularity': float
    """
    # For each matrix:
    #   1. Build thresholded graph (top 10% edges)
    #   2. Global efficiency = nx.global_efficiency(G)
    #   3. Modularity via community_louvain.best_partition(G)
    #      then community_louvain.modularity(partition, G)
    pass
```

#### 7c. New script: `scripts/07_graph_analysis.py`

```bash
python scripts/07_graph_analysis.py --model gemma3-4b
python scripts/07_graph_analysis.py --model qwen3-8b
```

This script:
1. Loads pairwise PhiID matrices from `results/phiid_scores/{model}_pairwise_phiid.npz`
2. Calls `plot_graph_cores()` to generate the graph figure
3. Calls `compute_graph_metrics()` to compute global efficiency and modularity
4. Saves figure to `results/figures/{model}_graph_cores.png`
5. Saves metrics to `results/phiid_scores/{model}_graph_metrics.json`

#### 7d. Integration into `run_pipeline.py`

Add as phase 7 in the pipeline runner:
```python
# Add to the phase_map:
7: lambda: phase7_graph_analysis(config, args.revision),
```

### Key Implementation Notes

- For Qwen 3 8B (1,152 nodes), the graph will be very dense even at 10%. Consider:
  - Using a circular layout grouped by layer
  - Or force-directed layout with `nx.spring_layout(G, k=0.5, iterations=50)`
  - Edge alpha scaled by weight for visual clarity
- The paper uses the FULL matrix (not just upper triangle) for graph construction, but since it's symmetric, this is equivalent to an undirected graph
- Node color should encode layer depth (e.g., viridis colormap)
- Edge thickness/alpha should encode weight
- For large graphs (Qwen), consider showing only a subset of nodes or using a different layout

---

## Phase 8: Figure 4b — MATH Benchmark Perturbation

**NEEDS IMPLEMENTATION**

### What the Paper Shows (Fig 4b)

Accuracy on the MATH benchmark when perturbing different subsets of heads:
- **Synergistic core** (top 25% most synergistic heads)
- **Redundant core** (top 25% most redundant heads)
- **Random subset** (25% of heads, multiple seeds)
- Perturbation = **Gaussian noise** injection (NOT zeroing)

### Key Differences from Fig 4a (Ablation)

| | Fig 4a (Ablation) | Fig 4b (Perturbation) |
|---|---|---|
| **Method** | Zero out head outputs | Inject Gaussian noise into weights |
| **Target** | All heads iteratively | Fixed subset (top 25%) |
| **Metric** | KL divergence on 60 prompts | Accuracy on MATH benchmark |
| **Noise location** | N/A | Q-projection rows + O-projection columns |

### Implementation Plan

#### 8a. New file: `src/perturbation.py`

```python
class GaussianNoisePerturbation:
    """
    Injects Gaussian noise into attention head weight matrices.

    For each targeted head h in layer l:
    - Q-projection: Add N(0, sigma) to the rows of W_Q corresponding to head h
      (rows h*head_dim to (h+1)*head_dim)
    - O-projection: Add N(0, sigma) to the columns of W_O corresponding to head h
      (columns h*head_dim to (h+1)*head_dim)

    This is DIFFERENT from ablation (zeroing). Noise allows partial function
    while disrupting precise computation.
    """

    def __init__(self, model, model_spec, sigma=0.1):
        """
        Args:
            model: the HuggingFace model
            model_spec: ModelSpec from model_registry
            sigma: std of Gaussian noise (fraction of param std)
                   The paper doesn't specify the exact sigma. We use sigma
                   as a multiplier of each parameter's own std:
                   noise_std = sigma * param.std()
        """
        self.model = model
        self.spec = model_spec
        self.sigma = sigma
        self._original_weights = {}  # for restoration

    def perturb_heads(self, head_indices):
        """
        Inject noise into Q-projection and O-projection weights for specified heads.

        Args:
            head_indices: list of flat head indices (will be converted to layer, head pairs)
        """
        # For each head:
        #   layer, head_in_layer = divmod(head_idx, spec.num_heads)
        #
        # For GPT-NeoX (Pythia):
        #   QKV is fused in model.gpt_neox.layers[l].attention.query_key_value
        #   O-proj is model.gpt_neox.layers[l].attention.dense
        #
        # For Gemma 3 / Qwen 3:
        #   Q-proj: model.model.layers[l].self_attn.q_proj
        #   O-proj: model.model.layers[l].self_attn.o_proj
        #
        # Save original weights before perturbation for restoration.
        pass

    def restore_weights(self):
        """Restore all perturbed weights to their original values."""
        pass
```

**Architecture-specific weight locations:**

| Component | Pythia (GPT-NeoX) | Gemma 3 / Qwen 3 |
|-----------|-------------------|-------------------|
| Q-proj | `layers[l].attention.query_key_value` (fused, rows 0:num_heads*head_dim) | `layers[l].self_attn.q_proj` (rows h*head_dim:(h+1)*head_dim) |
| O-proj | `layers[l].attention.dense` (columns h*head_dim:(h+1)*head_dim) | `layers[l].self_attn.o_proj` (columns h*head_dim:(h+1)*head_dim) |

For Pythia's fused QKV projection, the rows are laid out as: [Q_head0, Q_head1, ..., K_head0, K_head1, ..., V_head0, V_head1, ...]. To perturb Q for head h, modify rows `h*head_dim` to `(h+1)*head_dim`.

#### 8b. New file: `src/math_eval.py`

```python
def evaluate_math_accuracy(model, tokenizer, device, num_problems=500,
                           max_new_tokens=512):
    """
    Evaluate accuracy on the MATH (competition_math) benchmark.

    Uses the 'hendrycks/competition_math' dataset from HuggingFace.
    Problems are formatted as:
        "Problem: {problem}\nAnswer:"
    Model generates a response, and we extract the final answer.

    The answer is inside \\boxed{...} in the ground truth. We check if the
    model's output contains the correct answer (exact match after normalization).

    Args:
        model: HuggingFace causal LM (may have perturbed weights)
        tokenizer: corresponding tokenizer
        device: torch device
        num_problems: number of MATH problems to evaluate on (default 500)
        max_new_tokens: max tokens for generation

    Returns:
        accuracy: float (0 to 1)
        results: list of dicts with problem, prediction, ground_truth, correct
    """
    pass

def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} in the text."""
    # Regex to find \\boxed{...}, handling nested braces
    pass

def normalize_answer(answer):
    """Normalize mathematical answer for comparison."""
    # Strip whitespace, normalize fractions, etc.
    pass
```

#### 8c. New script: `scripts/09_math_perturbation.py`

```bash
python scripts/09_math_perturbation.py --model gemma3-4b --sigma 0.1 --num-problems 500
python scripts/09_math_perturbation.py --model qwen3-8b --sigma 0.1 --num-problems 500
```

This script:
1. Loads the model and head rankings
2. Defines three perturbation conditions:
   - **Synergistic core**: top 25% heads by syn_red_score
   - **Redundant core**: bottom 25% heads by syn_red_score
   - **Random**: 25% of heads (3 random seeds for mean +/- std)
3. For each condition:
   a. Load fresh model
   b. Apply Gaussian noise perturbation
   c. Evaluate on MATH benchmark
   d. Record accuracy
4. Also evaluate the **unperturbed** baseline
5. Save results to `results/ablation/{model}_math_perturbation.csv`
6. Generate Fig 4b bar chart

#### 8d. New visualization function: `plot_math_perturbation`

```python
def plot_math_perturbation(results_df, title, save_path):
    """
    Fig 4b: Bar chart of MATH accuracy under different perturbation conditions.

    X-axis: Perturbation condition (Baseline, Synergistic, Redundant, Random)
    Y-axis: MATH accuracy (%)
    Random bars show error bars (std across seeds)

    Expected: Synergistic perturbation causes the largest accuracy drop.
    """
    pass
```

### Key Implementation Notes for Fig 4b

1. **Sigma selection**: The paper doesn't specify the exact noise magnitude. Start with `sigma = 0.1` (noise std = 10% of each parameter's std). If results are too weak or too strong, try 0.05, 0.2, 0.5.

2. **MATH evaluation is expensive**: Each evaluation requires generating responses for hundreds of problems. Budget ~1 hour per condition per model on A100.

3. **Gemma 3 4B may struggle on MATH**: This is a pretrained (not instruction-tuned) model. Expect low baseline accuracy. The key is the *relative* drop between conditions, not absolute accuracy.

4. **Qwen 3 8B should perform better**: Larger model, and Qwen is known for strong math capabilities.

5. **Alternative**: If MATH accuracy is too low for meaningful comparison, consider using a simpler benchmark (GSM8K, or even perplexity on a held-out set as a proxy).

6. **Memory**: Perturbing weights is done in-place (with backup for restoration), so no extra VRAM is needed.

---

## Execution Order (Full Pipeline)

```bash
# === Login Node (internet required) ===

# 0. Pre-cache models and datasets
# (see Phase 0 above)

# === Compute Node (GPU required) ===

# 1-4,6. Main pipeline for Gemma (ready to run)
python scripts/run_pipeline.py --model gemma3-4b --phases 1 2 3 4 6 --max-workers 32

# 1-4,6. Main pipeline for Qwen (ready to run)
python scripts/run_pipeline.py --model qwen3-8b --phases 1 2 3 4 6 --max-workers 32

# 7. Graph analysis — Fig 3b (NEEDS IMPLEMENTATION)
python scripts/07_graph_analysis.py --model gemma3-4b
python scripts/07_graph_analysis.py --model qwen3-8b

# 8. Multi-model comparison plots
python scripts/08_compare_models.py

# 9. MATH perturbation — Fig 4b (NEEDS IMPLEMENTATION)
python scripts/09_math_perturbation.py --model gemma3-4b --sigma 0.1
python scripts/09_math_perturbation.py --model qwen3-8b --sigma 0.1
```

### SLURM Strategy

For Qwen 3 8B, the full pipeline exceeds typical wall time limits. Split into jobs:

```bash
# Job 1: Phases 1-3 (extraction + PhiID + ranking) — ~25 hrs
sbatch slurm/qwen3.sh  # modify to run --phases 1 2 3

# Job 2: Phase 4 (ablation) — ~8 hrs, AFTER Job 1 completes
sbatch --dependency=afterok:<JOB1_ID> slurm/qwen3_ablation.sh  # --phases 4 6

# Job 3: Phase 7 (graph analysis) + Phase 9 (MATH) — ~4 hrs, AFTER Job 1
sbatch --dependency=afterok:<JOB1_ID> slurm/qwen3_extra.sh  # graph + MATH
```

---

## Expected Results

### Figure 2a (Pairwise Heatmaps)
- Synergy heatmap: should show block-diagonal structure (within-layer pairs have higher synergy) with some cross-layer hot spots
- Redundancy heatmap: should show broader structure, especially in early and late layers

### Figure 2b (Head Ranking Grid)
- Middle layers should be predominantly red (synergistic)
- Early and late layers should be predominantly blue (redundant)
- This is the "inverted-U" pattern in 2D form

### Figure 3b (Graph Cores)
- Synergistic graph: dense, integrated connections across middle-layer heads
- Redundant graph: more modular/clustered structure in early/late layers
- Global efficiency should be higher for synergistic network
- Modularity should be higher for redundant network

### Figure 4a (Ablation Curves)
- Synergistic-order removal (solid line) should cause KL divergence to rise FASTER than random (dashed line)
- This demonstrates that synergistic heads disproportionately drive model behavior

### Figure 4b (MATH Perturbation)
- Perturbing synergistic core should cause the LARGEST accuracy drop
- Perturbing redundant core should cause a smaller drop
- Random perturbation should fall in between (closer to redundant)

---

## Troubleshooting

### "Model not found" on compute node
Models aren't cached. Go back to login node and run Phase 0b.

### Gemma "gated model" error
Need HF_TOKEN. See Phase 0a. Make sure `.env` file exists OR `export HF_TOKEN=...`.

### PhiID checkpoint resume
Phase 2 saves checkpoints every 5,000 pairs. If a job times out, just resubmit — it resumes automatically.

### Qwen phase 2 is too slow
663K pairs at ~30ms each = ~5.5 hours with perfect parallelism. With 32 workers it's ~24 hours. Consider:
- Running on a node with more CPUs (64+)
- Splitting into multiple jobs (the checkpoint system handles this)

### Out of memory (GPU)
- Gemma 3 4B in fp16: ~8 GB VRAM. Should fit on any A100.
- Qwen 3 8B in fp16: ~16 GB VRAM. Fits on A100-40GB.
- If OOM during ablation (full sequence forward pass), reduce batch size or sequence length.

### SLURM account
Scripts use `def-bhrett`. Change to your account:
```bash
sed -i 's/def-bhrett/YOUR_ACCOUNT/' slurm/gemma3.sh slurm/qwen3.sh
```

---

## File Inventory

### Existing (ready to use)
```
scripts/run_pipeline.py          # Unified pipeline runner (phases 1-6)
scripts/08_compare_models.py     # Multi-model comparison figures
src/model_registry.py            # Architecture detection for all 3 models
src/activation_extraction.py     # Hook-based per-head activation capture
src/phiid_computation.py         # PhiID computation with parallelism + checkpoints
src/head_ranking.py              # Syn-red ranking
src/ablation.py                  # Head ablation + KL divergence
src/visualization.py             # All current plotting functions
src/prompts.py                   # 60 cognitive task prompts
src/utils.py                     # Model loading, seeding, paths
configs/config.py                # Multi-model config system
slurm/gemma3.sh                  # SLURM script for Gemma 3 4B
slurm/qwen3.sh                  # SLURM script for Qwen 3 8B
.env.example                     # Template for HF_TOKEN
```

### Needs Implementation
```
src/graph_analysis.py            # NEW — graph construction + metrics (Fig 3b, 3c)
src/perturbation.py              # NEW — Gaussian noise injection (Fig 4b)
src/math_eval.py                 # NEW — MATH benchmark evaluation (Fig 4b)
scripts/07_graph_analysis.py     # NEW — graph visualization script
scripts/09_math_perturbation.py  # NEW — MATH perturbation script
src/visualization.py             # UPDATE — add plot_graph_cores, plot_math_perturbation
```
