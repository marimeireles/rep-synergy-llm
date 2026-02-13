# rep-synergy-llm

Reproducing ["A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning"](https://arxiv.org/abs/2601.06851v1) using PhiID (Integrated Information Decomposition) on Pythia-1B, Gemma 3 4B, and Qwen 3 8B.

## Setup on Narval

All steps below run on the **login node** (internet access required).

### 1. Clone the repo

```bash
cd /home/marimeir/scratch
git clone <repo-url> rep-synergy-llm
cd rep-synergy-llm
```

### 2. Create the conda environment

```bash
export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
micromamba create -n syn python=3.10 -y
micromamba activate syn
```

### 3. Install dependencies

```bash
# PyTorch with CUDA (conda channel for proper CUDA build)
micromamba install -y -n syn pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Python packages
pip install transformers accelerate numpy scipy matplotlib seaborn tqdm pandas pytest

# phyid (PhiID library) — local editable install, no internet needed on compute nodes
pip install -e /home/marimeir/scratch/integrated-info-decomp
```

### 4. Create `.env` with your HuggingFace token

Gemma 3 is a gated model — you need a [HuggingFace token](https://huggingface.co/settings/tokens) with access granted to `google/gemma-3-4b-pt`.

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder:

```
HF_TOKEN=hf_your_actual_token_here
```

This file is gitignored. Both the Python scripts and SLURM jobs read it automatically.

### 5. Pre-download models

Compute nodes have no internet. Download everything on the login node first.

```bash
micromamba activate syn
source .env  # makes HF_TOKEN available

# Pythia-1B (no token needed)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b')
AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
print('Pythia-1B downloaded.')
"

# Gemma 3 4B (needs HF_TOKEN with access)
python -c "
import os
token = os.environ['HF_TOKEN']
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('google/gemma-3-4b-pt', token=token)
AutoTokenizer.from_pretrained('google/gemma-3-4b-pt', token=token)
print('Gemma 3 4B downloaded.')
"

# Qwen 3 8B (no token needed)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B')
AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
print('Qwen 3 8B downloaded.')
"
```

### 6. Create output directories

```bash
mkdir -p results/{activations,phiid_scores,ablation,figures} slurm/logs
```

## Running

### Unified pipeline

```bash
# Full pipeline for a model (phases: 1=extract, 2=phiid, 3=rank, 4=ablation, 6=visualize)
python scripts/run_pipeline.py --model pythia-1b  --phases 1 2 3 4 6
python scripts/run_pipeline.py --model gemma3-4b  --phases 1 2 3 4 6
python scripts/run_pipeline.py --model qwen3-8b   --phases 1 2 3 4 6
```

### SLURM jobs

```bash
sbatch slurm/gemma3.sh    # ~12 hrs, 1 GPU + 32 CPU
sbatch slurm/qwen3.sh     # ~48 hrs, 1 GPU + 32 CPU (resumable)
```

### Multi-model comparison (after individual runs complete)

```bash
python scripts/08_compare_models.py
```

### Legacy single-model scripts (Pythia-1B only)

```bash
python scripts/01_extract_activations.py
python scripts/02_compute_phiid.py
python scripts/03_rank_heads.py
python scripts/04_run_ablation.py
python scripts/05_random_baseline.py
python scripts/06_visualize.py
```
