#!/bin/bash
#SBATCH --job-name=qwen_ablation_balanced
#SBATCH --account=def-zhijing_gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results/slurm_qwen_ablation_balanced_%j.out
#SBATCH --error=results/slurm_qwen_ablation_balanced_%j.err

# Activate environment
export MAMBA_ROOT_PREFIX=/home/marimeir/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate syn

# Offline mode (compute nodes have no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Torch lib path
export LD_LIBRARY_PATH=/home/marimeir/micromamba/envs/syn/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

cd /lustre07/scratch/marimeir/rep-synergy-llm

echo "Starting Qwen 3 8B ablation with balanced ranking..."
echo "Date: $(date)"
nvidia-smi

python scripts/rerun_ablation_balanced.py \
    --model qwen3-8b \
    --ranking-csv results/phiid_scores/qwen3_8b_head_rankings_balanced.csv \
    --step-size 16 \
    --num-random-seeds 5 \
    --output results/ablation/qwen3_8b_ablation_balanced.csv

echo "Done: $(date)"
