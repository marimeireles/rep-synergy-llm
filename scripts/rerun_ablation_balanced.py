"""
Re-run ablation for Qwen 3 8B using per-pair balance ranking.

This script uses the improved head rankings (per-pair balance sts/(sts+rtr))
which produce a clearer synergistic core signal than the raw rank difference method.
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
from src.utils import set_seed, get_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Model key: qwen3-8b or pythia-1b')
    parser.add_argument('--ranking-csv', type=str, required=True,
                        help='Path to head rankings CSV')
    parser.add_argument('--step-size', type=int, default=1,
                        help='Measure KL every N heads removed')
    parser.add_argument('--num-random-seeds', type=int, default=5)
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path')
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    # Model configs
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

    cfg = MODEL_CONFIGS[args.model]
    prefix = cfg['prefix']

    # Load .env for HF_TOKEN (gated models like Gemma)
    from src.utils import load_dotenv
    load_dotenv()
    hf_token = os.environ.get('HF_TOKEN')
    local_only = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'

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
    # Handle both formats: plain list or dict with 'generated_tokens' key
    if isinstance(tokens_data, dict):
        original_tokens_per_prompt = tokens_data['generated_tokens']
    else:
        original_tokens_per_prompt = tokens_data

    with open(f'{act_dir}/{prefix}_prompt_token_ids.json', 'r') as f:
        prompt_token_ids = json.load(f)

    original_logits = np.load(f'{act_dir}/{prefix}_logits.npz')['logits']
    logger.info(f"Loaded logits shape: {original_logits.shape}")

    from src.prompts import PROMPTS
    prompts = PROMPTS

    num_heads_per_layer = spec.num_heads

    # Create ablation engine
    engine = HeadAblationEngine(model, tokenizer, device, model_spec=spec)

    # Synergistic order (most synergistic first)
    head_order_syn = rankings_df.sort_values(
        'syn_red_score', ascending=False
    )['head_idx'].tolist()

    logger.info("Running synergistic-order ablation...")
    results_syn = engine.run_iterative_ablation(
        prompts=prompts,
        prompt_token_ids=prompt_token_ids,
        original_tokens_per_prompt=original_tokens_per_prompt,
        original_logits_per_prompt=original_logits,
        head_order=head_order_syn,
        order_name='syn_red',
        num_layers=spec.num_layers,
        num_heads_per_layer=num_heads_per_layer,
        step_size=args.step_size,
    )

    # Random orderings
    all_heads = list(range(spec.num_layers * num_heads_per_layer))
    for seed in range(args.num_random_seeds):
        rng = np.random.RandomState(seed)
        random_order = rng.permutation(all_heads).tolist()
        logger.info(f"Running random ablation (seed {seed})...")
        results_rand = engine.run_iterative_ablation(
            prompts=prompts,
            prompt_token_ids=prompt_token_ids,
            original_tokens_per_prompt=original_tokens_per_prompt,
            original_logits_per_prompt=original_logits,
            head_order=random_order,
            order_name=f'random_{seed}',
            num_layers=spec.num_layers,
            num_heads_per_layer=num_heads_per_layer,
            step_size=args.step_size,
        )
        results_syn.extend(results_rand)

    # Save
    df_out = pd.DataFrame(results_syn)
    df_out.to_csv(args.output, index=False)
    logger.info(f"Saved ablation results to {args.output}")


if __name__ == '__main__':
    main()
