#!/bin/bash
#SBATCH --job-name=phiid-synergy
#SBATCH --account=def-zhijing_gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

# Reproduce "A Brain-like Synergistic Core in LLMs" on Pythia-1B
# Runs all 6 phases sequentially on a single GPU node.

set -e  # Exit on error

# Load CUDA/cuDNN modules (required on Narval compute nodes)
module load StdEnv/2023 cudacore/.12.2.2 cudnn/9.2.1.18 2>/dev/null || true

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate syn

# Fallback: ensure torch can find its bundled CUDA libraries
export LD_LIBRARY_PATH=/home/marimeir/micromamba/envs/syn/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

# Compute nodes have no internet — force HuggingFace to use only cached files
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /lustre07/scratch/marimeir/rep-synergy-llm

# Ensure results directories exist
mkdir -p results/activations results/phiid_scores results/ablation results/figures

echo "=== Starting pipeline at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"
python -c "import torch; print(f'torch {torch.__version__}, cuda: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}')"
echo ""

# Unbuffer Python output for real-time monitoring
export PYTHONUNBUFFERED=1

# Phase 1: Extract activations (GPU)
echo "=== Phase 1: Activation Extraction ==="
python scripts/01_extract_activations.py
echo "Phase 1 done at $(date)"
echo ""

# Phase 2: Compute PhiID (CPU-heavy, parallelized)
echo "=== Phase 2: PhiID Computation ==="
python scripts/02_compute_phiid.py
echo "Phase 2 done at $(date)"
echo ""

# Phase 3: Rank heads
echo "=== Phase 3: Head Ranking ==="
python scripts/03_rank_heads.py
echo "Phase 3 done at $(date)"
echo ""

# Phase 4: Ablation experiments (GPU)
echo "=== Phase 4: Ablation ==="
python scripts/04_run_ablation.py
echo "Phase 4 done at $(date)"
echo ""

# Phase 5: Random baseline (GPU + CPU)
echo "=== Phase 5: Random Baseline ==="
python scripts/05_random_baseline.py
echo "Phase 5 done at $(date)"
echo ""

# Phase 6: Visualization
echo "=== Phase 6: Visualization ==="
python scripts/06_visualize.py
echo "Phase 6 done at $(date)"
echo ""

echo "=== Pipeline complete at $(date) ==="
