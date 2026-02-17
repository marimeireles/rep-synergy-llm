"""
Re-run ablation using Gaussian noise injection instead of zeroing.

The paper states: "injecting Gaussian noise into the query-output projections
of selected attention heads." This script uses the noise ablation method
with per-head calibrated noise scaling.

Supports all models. Reuses existing activations, logits, and rankings.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ablation import HeadAblationEngine
from src.model_registry import detect_model_spec
from src.utils import set_seed, get_device, load_dotenv
from src.prompts import PROMPTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


MODEL_CONFIGS = {
    'pythia-1b': {
        'model_name': 'EleutherAI/pythia-1b',
        'prefix': 'pythia1b',
        'dtype': torch.float32,
    },
    'qwen3-8b': {
        'model_name': 'Qwen/Qwen3-8B',
        'prefix': 'qwen3_8b',
        'dtype': torch.float16,
    },
    'gemma3-4b': {
        'model_name': 'google/gemma-3-4b-pt',
        'prefix': 'gemma3_4b',
        'dtype': torch.bfloat16,
    },
    'gemma3-4b-it': {
        'model_name': 'google/gemma-3-4b-it',
        'prefix': 'gemma3_4b_it',
        'dtype': torch.bfloat16,
    },
    'gemma3-12b-it': {
        'model_name': 'google/gemma-3-12b-it',
        'prefix': 'gemma3_12b_it',
        'dtype': torch.bfloat16,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Noise injection ablation experiment")
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--ranking-csv', type=str, required=True,
                        help='Path to head rankings CSV (standard or balanced)')
    parser.add_argument('--ranking-method', type=str, default='standard',
                        choices=['standard', 'balanced'],
                        help='Label for the ranking method used')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                        help='Noise scale multiplier (default: 1.0 = match activation std)')
    parser.add_argument('--step-size', type=int, default=1,
                        help='Measure KL every N heads removed')
    parser.add_argument('--num-random-seeds', type=int, default=5)
    parser.add_argument('--num-calibration-prompts', type=int, default=10,
                        help='Number of prompts to use for noise calibration')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path')
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    cfg = MODEL_CONFIGS[args.model]
    prefix = cfg['prefix']

    load_dotenv()
    hf_token = os.environ.get('HF_TOKEN')
    local_only = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'

    # Load model
    logger.info(f"Loading model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model_name'], local_files_only=local_only, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model_name'], dtype=cfg['dtype'], device_map='auto',
        local_files_only=local_only, token=hf_token,
    )
    model.eval()

    spec = detect_model_spec(model)
    logger.info(f"Model spec: {spec}")

    # Load rankings
    logger.info(f"Loading rankings from {args.ranking_csv}")
    rankings_df = pd.read_csv(args.ranking_csv)

    # Load original tokens and logits
    act_dir = 'results/activations'
    with open(f'{act_dir}/{prefix}_tokens.json', 'r') as f:
        tokens_data = json.load(f)
    if isinstance(tokens_data, dict):
        original_tokens_per_prompt = tokens_data['generated_tokens']
    else:
        original_tokens_per_prompt = tokens_data

    with open(f'{act_dir}/{prefix}_prompt_token_ids.json', 'r') as f:
        prompt_token_ids = json.load(f)

    original_logits = np.load(f'{act_dir}/{prefix}_logits.npz')['logits'].astype(np.float32)
    logger.info(f"Loaded logits shape: {original_logits.shape}")

    # Create ablation engine with NOISE method
    engine = HeadAblationEngine(
        model, tokenizer, device, model_spec=spec,
        ablation_method='noise', noise_scale=args.noise_scale,
    )

    # Calibrate noise from activation statistics
    logger.info(f"Calibrating noise from {args.num_calibration_prompts} prompts...")
    engine.calibrate_noise(
        prompt_token_ids, original_tokens_per_prompt,
        num_calibration_prompts=args.num_calibration_prompts,
    )

    num_heads_per_layer = spec.num_heads
    num_total_heads = spec.num_total_heads

    # Auto step_size for large models
    if args.step_size == 1 and num_total_heads > 500:
        args.step_size = max(1, num_total_heads // 150)
        logger.info(f"Auto step_size={args.step_size} for {num_total_heads} heads")

    # Synergistic order: most synergistic first
    head_order_syn = rankings_df.sort_values(
        'syn_red_score', ascending=False
    )['head_idx'].tolist()

    method_label = f"noise_{args.ranking_method}"

    logger.info(f"Running synergistic-order noise ablation ({method_label})...")
    results_syn = engine.run_iterative_ablation(
        prompts=PROMPTS,
        prompt_token_ids=prompt_token_ids,
        original_tokens_per_prompt=original_tokens_per_prompt,
        original_logits_per_prompt=original_logits,
        head_order=head_order_syn,
        order_name=f'syn_red_{method_label}',
        num_layers=spec.num_layers,
        num_heads_per_layer=num_heads_per_layer,
        step_size=args.step_size,
    )

    # Random orderings
    all_heads = list(range(num_total_heads))
    for seed in range(args.num_random_seeds):
        rng = np.random.RandomState(seed)
        random_order = rng.permutation(all_heads).tolist()
        logger.info(f"Running random noise ablation (seed {seed})...")
        results_rand = engine.run_iterative_ablation(
            prompts=PROMPTS,
            prompt_token_ids=prompt_token_ids,
            original_tokens_per_prompt=original_tokens_per_prompt,
            original_logits_per_prompt=original_logits,
            head_order=random_order,
            order_name=f'random_{seed}_{method_label}',
            num_layers=spec.num_layers,
            num_heads_per_layer=num_heads_per_layer,
            step_size=args.step_size,
        )
        results_syn.extend(results_rand)

    # Save
    df_out = pd.DataFrame(results_syn)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    logger.info(f"Saved noise ablation results to {args.output}")
    logger.info(f"Total rows: {len(df_out)}")


if __name__ == '__main__':
    main()
