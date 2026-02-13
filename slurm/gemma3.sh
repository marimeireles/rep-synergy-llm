#!/bin/bash
#SBATCH --job-name=gemma3-phiid
#SBATCH --account=def-bhrett
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/gemma3_%j.out
#SBATCH --error=slurm/logs/gemma3_%j.err

# Gemma 3 4B full pipeline (Figure 2c + 4a)
# Phase 1: ~10 min (GPU), Phase 2: ~80 min (CPU), Phase 3: seconds
# Phase 4: ~4-8 hrs (GPU), Phase 6: seconds
# Total: ~6-10 hrs

set -e

export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /home/marimeir/scratch/rep-synergy-llm
mkdir -p slurm/logs

echo "=== Gemma 3 4B Pipeline ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load secrets from .env (HF_TOKEN, etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

python scripts/run_pipeline.py --model gemma3-4b --phases 1 2 3 4 6 --max-workers 32

echo "=== Complete ==="
echo "End time: $(date)"
