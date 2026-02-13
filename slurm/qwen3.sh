#!/bin/bash
#SBATCH --job-name=qwen3-phiid
#SBATCH --account=def-bhrett
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/qwen3_%j.out
#SBATCH --error=slurm/logs/qwen3_%j.err

# Qwen 3 8B full pipeline
# Phase 1: ~15 min (GPU), Phase 2: ~24 hrs (CPU, 663K pairs)
# Phase 3: seconds, Phase 4: ~4-8 hrs (GPU, step_size=8)
# Phase 6: seconds. Total: ~30-40 hrs
#
# PhiID computation (phase 2) has checkpoint/resume support.
# If job times out, resubmit — it will resume from the last checkpoint.

set -e

export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /home/marimeir/scratch/rep-synergy-llm
mkdir -p slurm/logs

echo "=== Qwen 3 8B Pipeline ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load secrets from .env (HF_TOKEN, etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

python scripts/run_pipeline.py --model qwen3-8b --phases 1 2 3 4 6 --max-workers 32

echo "=== Complete ==="
echo "End time: $(date)"
