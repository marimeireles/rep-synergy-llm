#!/bin/bash
#SBATCH --job-name=pythia_ablation_concat
#SBATCH --account=def-zhijing_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=results/slurm_pythia_ablation_concat_%j.out
#SBATCH --error=results/slurm_pythia_ablation_concat_%j.err

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

echo "Starting Pythia-1B ablation with concatenated PhiID ranking..."
echo "Date: $(date)"
nvidia-smi

python scripts/rerun_ablation_balanced.py \
    --model pythia-1b \
    --ranking-csv results/phiid_scores/pythia1b_concat_head_rankings.csv \
    --step-size 1 \
    --num-random-seeds 5 \
    --output results/ablation/pythia1b_ablation_concat.csv

echo "Done: $(date)"
