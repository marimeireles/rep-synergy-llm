#!/usr/bin/env python
"""
Unified pipeline runner for multi-model PhiID analysis.

Usage:
  python scripts/run_pipeline.py --model pythia-1b --phases 1 2 3 4 6
  python scripts/run_pipeline.py --model gemma3-4b --phases 1 2 3 4 6
  python scripts/run_pipeline.py --model qwen3-8b --phases 1 2 3 6
  python scripts/run_pipeline.py --model pythia-1b --revision step1000 --phases 1 2 3

Phases:
  1 = Extract activations (GPU)
  2 = Compute PhiID (CPU-heavy)
  3 = Rank heads
  4 = Ablation experiments (GPU)
  5 = Random baseline (GPU + CPU)
  6 = Visualize
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from src.utils import (
    set_seed, get_device, load_model_and_tokenizer,
    setup_logging, ensure_dirs, get_result_path, load_dotenv,
)
from src.prompts import PROMPTS
from src.model_registry import detect_model_spec
from src.activation_extraction import HeadActivationExtractor
from src.phiid_computation import compute_all_pairs_phiid
from src.head_ranking import compute_head_scores, compute_syn_red_rank, build_ranking_dataframe
from src.ablation import HeadAblationEngine
from src.visualization import (
    plot_phiid_profile,
    plot_synergy_redundancy_heatmaps,
    plot_head_ranking_heatmap,
    plot_ablation_curves,
    plot_trained_vs_random,
)

logger = logging.getLogger(__name__)


def phase1_extract(config, device, revision=None, hf_token=None):
    """Phase 1: Extract per-head activations during autoregressive generation."""
    model_id = config["model_id"]
    results_dir = config["results_dir"]

    act_path = get_result_path(results_dir, "activations", model_id, "activations.npz", revision)
    logits_path = get_result_path(results_dir, "activations", model_id, "logits.npz", revision)
    tokens_path = get_result_path(results_dir, "activations", model_id, "tokens.json", revision)
    prompt_ids_path = get_result_path(results_dir, "activations", model_id, "prompt_token_ids.json", revision)

    if os.path.exists(act_path) and os.path.exists(logits_path):
        logger.info(f"Activations already exist at {act_path}. Verifying...")
        data = np.load(act_path)
        logger.info(f"  Shape: {data['activations'].shape}")
        assert np.all(np.isfinite(data['activations'])), "Non-finite values!"
        logger.info("  Verification passed. Skipping extraction.")
        return

    model, tokenizer = load_model_and_tokenizer(
        config["model_name"], device,
        revision=revision,
        torch_dtype=config["torch_dtype"],
        hf_token=hf_token,
    )
    spec = detect_model_spec(model)
    extractor = HeadActivationExtractor(model, tokenizer, device, model_spec=spec)

    logger.info(f"Extracting activations for {len(PROMPTS)} prompts, "
                f"{config['num_tokens_to_generate']} tokens each")

    all_activations, all_tokens, all_logits = extractor.extract_all_prompts(
        PROMPTS, num_tokens=config["num_tokens_to_generate"]
    )

    # Save activations
    np.savez(act_path, activations=all_activations)
    logger.info(f"Saved activations to {act_path} — shape: {all_activations.shape}")

    # Save logits — use float16 + compression for large-vocab models
    if all_logits.shape[-1] > 100000:
        logger.info(f"Large vocab ({all_logits.shape[-1]}), saving logits as float16 compressed")
        np.savez_compressed(logits_path, logits=all_logits.astype(np.float16))
    else:
        np.savez_compressed(logits_path, logits=all_logits)
    logger.info(f"Saved logits to {logits_path} — shape: {all_logits.shape}")

    # Save tokens
    with open(tokens_path, 'w') as f:
        json.dump(all_tokens, f)
    logger.info(f"Saved tokens to {tokens_path}")

    # Save prompt token IDs
    prompt_token_ids = [tokenizer.encode(p) for p in PROMPTS]
    with open(prompt_ids_path, 'w') as f:
        json.dump(prompt_token_ids, f)
    logger.info(f"Saved prompt token IDs to {prompt_ids_path}")

    # Verification
    assert np.all(np.isfinite(all_activations)), "Non-finite values!"
    assert np.all(all_activations >= 0), "Negative L2 norms!"

    # Per-layer stats
    for layer in range(spec.num_layers):
        start = layer * spec.num_heads
        end = start + spec.num_heads
        layer_acts = all_activations[:, start:end, :]
        logger.info(f"Layer {layer:2d}: min={layer_acts.min():.4f}, "
                    f"max={layer_acts.max():.4f}, mean={layer_acts.mean():.4f}")

    # Free model
    del model, extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Phase 1 complete.")


def phase2_phiid(config, revision=None, max_workers=None):
    """Phase 2: Compute PhiID for all head pairs."""
    model_id = config["model_id"]
    results_dir = config["results_dir"]

    save_path = get_result_path(results_dir, "phiid_scores", model_id, "pairwise_phiid.npz", revision)

    if os.path.exists(save_path):
        logger.info(f"PhiID results already exist at {save_path}. Verifying...")
        data = np.load(save_path)
        logger.info(f"  sts shape: {data['sts_matrix'].shape}")
        logger.info("  Skipping computation.")
        return

    act_path = get_result_path(results_dir, "activations", model_id, "activations.npz", revision)
    if not os.path.exists(act_path):
        logger.error(f"Activations not found at {act_path}. Run phase 1 first.")
        sys.exit(1)

    activations = np.load(act_path)['activations']
    num_heads = activations.shape[1]
    logger.info(f"Loaded activations: shape {activations.shape}, {num_heads} heads")

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 32)
    logger.info(f"Using {max_workers} workers")

    sts_matrix, rtr_matrix = compute_all_pairs_phiid(
        activations=activations,
        num_heads=num_heads,
        tau=config["phiid_tau"],
        kind=config["phiid_kind"],
        redundancy=config["phiid_redundancy"],
        max_workers=max_workers,
        save_path=save_path,
        checkpoint_interval=5000,
    )

    logger.info(f"PhiID computation complete.")
    logger.info(f"  sts range: [{sts_matrix.min():.6f}, {sts_matrix.max():.6f}]")
    logger.info(f"  rtr range: [{rtr_matrix.min():.6f}, {rtr_matrix.max():.6f}]")


def phase3_rank(config, revision=None):
    """Phase 3: Rank heads by syn-red score."""
    model_id = config["model_id"]
    results_dir = config["results_dir"]

    save_path = get_result_path(results_dir, "phiid_scores", model_id, "head_rankings.csv", revision)

    if os.path.exists(save_path):
        logger.info(f"Head rankings already exist at {save_path}.")
        df = pd.read_csv(save_path)
        layer_means = df.groupby('layer')['syn_red_score'].mean()
        for layer, score in layer_means.items():
            logger.info(f"  Layer {layer:2d}: {score:.4f}")
        return

    phiid_path = get_result_path(results_dir, "phiid_scores", model_id, "pairwise_phiid.npz", revision)
    if not os.path.exists(phiid_path):
        logger.error(f"PhiID results not found at {phiid_path}. Run phase 2 first.")
        sys.exit(1)

    data = np.load(phiid_path)
    sts_matrix = data['sts_matrix']
    rtr_matrix = data['rtr_matrix']
    num_heads = sts_matrix.shape[0]

    # Infer num_layers and num_heads_per_layer from activations shape
    act_path = get_result_path(results_dir, "activations", model_id, "activations.npz", revision)
    if os.path.exists(act_path):
        act_data = np.load(act_path)
        total_heads = act_data['activations'].shape[1]
    else:
        total_heads = num_heads

    # We need to figure out num_layers and heads_per_layer.
    # Try common configurations.
    num_heads_per_layer, num_layers = _infer_architecture(total_heads, model_id)

    synergy_per_head, redundancy_per_head = compute_head_scores(
        sts_matrix, rtr_matrix, num_heads
    )
    syn_red_rank, syn_rank, red_rank = compute_syn_red_rank(
        synergy_per_head, redundancy_per_head
    )
    df = build_ranking_dataframe(
        synergy_per_head, redundancy_per_head,
        syn_red_rank, syn_rank, red_rank,
        num_layers, num_heads_per_layer,
    )
    df.to_csv(save_path, index=False)
    logger.info(f"Saved head rankings to {save_path}")

    # Print top-10 most synergistic and redundant
    logger.info("\nTop-10 most synergistic heads:")
    for _, row in df.nlargest(10, 'syn_red_score').iterrows():
        logger.info(f"  Head {int(row['head_idx']):3d} (L{int(row['layer']):2d}H{int(row['head_in_layer']):2d}): "
                    f"score={row['syn_red_score']:.4f}")


def phase4_ablation(config, device, revision=None, hf_token=None):
    """Phase 4: Iterative ablation experiments."""
    model_id = config["model_id"]
    results_dir = config["results_dir"]

    save_path = get_result_path(results_dir, "ablation", model_id, "ablation.csv", revision)

    if os.path.exists(save_path):
        logger.info(f"Ablation results already exist at {save_path}. Skipping.")
        return

    # Load prerequisites
    rankings_path = get_result_path(results_dir, "phiid_scores", model_id, "head_rankings.csv", revision)
    logits_path = get_result_path(results_dir, "activations", model_id, "logits.npz", revision)
    tokens_path = get_result_path(results_dir, "activations", model_id, "tokens.json", revision)
    prompt_ids_path = get_result_path(results_dir, "activations", model_id, "prompt_token_ids.json", revision)

    for path, name in [(rankings_path, "head rankings"), (logits_path, "logits"), (tokens_path, "tokens")]:
        if not os.path.exists(path):
            logger.error(f"{name} not found at {path}. Run previous phases first.")
            sys.exit(1)

    rankings_df = pd.read_csv(rankings_path)
    original_logits = np.load(logits_path)['logits'].astype(np.float32)
    with open(tokens_path, 'r') as f:
        original_tokens = json.load(f)

    # Load or compute prompt_token_ids
    if os.path.exists(prompt_ids_path):
        with open(prompt_ids_path, 'r') as f:
            prompt_token_ids = json.load(f)
    else:
        # Need tokenizer to encode prompts
        local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], revision=revision,
            local_files_only=local_only, token=hf_token,
        )
        prompt_token_ids = [tokenizer.encode(p) for p in PROMPTS]

    logger.info(f"Loaded logits shape: {original_logits.shape}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        config["model_name"], device,
        revision=revision,
        torch_dtype=config["torch_dtype"],
        hf_token=hf_token,
    )
    spec = detect_model_spec(model)
    engine = HeadAblationEngine(model, tokenizer, device, model_spec=spec)

    num_total_heads = spec.num_total_heads

    # Determine step_size for large models
    step_size = 1
    if num_total_heads > 500:
        step_size = max(1, num_total_heads // 150)
        logger.info(f"Large model ({num_total_heads} heads), using step_size={step_size}")

    # Synergistic order: most synergistic first
    sorted_heads = rankings_df.sort_values('syn_red_score', ascending=False)
    syn_red_order = sorted_heads['head_idx'].tolist()

    logger.info("Running synergistic-order ablation...")
    syn_results = engine.run_iterative_ablation(
        prompts=PROMPTS,
        prompt_token_ids=prompt_token_ids,
        original_tokens_per_prompt=original_tokens,
        original_logits_per_prompt=original_logits,
        head_order=syn_red_order,
        order_name='syn_red',
        num_layers=spec.num_layers,
        num_heads_per_layer=spec.num_heads,
        step_size=step_size,
    )

    all_results = syn_results

    # Random orderings
    num_seeds = config["num_random_ablation_seeds"]
    for seed_idx in range(num_seeds):
        rng = np.random.RandomState(config["seed"] + seed_idx + 1)
        random_order = list(range(num_total_heads))
        rng.shuffle(random_order)

        logger.info(f"Running random ablation (seed {seed_idx + 1}/{num_seeds})...")
        rand_results = engine.run_iterative_ablation(
            prompts=PROMPTS,
            prompt_token_ids=prompt_token_ids,
            original_tokens_per_prompt=original_tokens,
            original_logits_per_prompt=original_logits,
            head_order=random_order,
            order_name=f'random_{seed_idx}',
            num_layers=spec.num_layers,
            num_heads_per_layer=spec.num_heads,
            step_size=step_size,
        )
        all_results.extend(rand_results)

    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved ablation results to {save_path} ({len(df)} rows)")

    # Free model
    del model, engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase5_random_baseline(config, device, revision=None):
    """Phase 5: Random baseline — random weights, same architecture."""
    model_id = config["model_id"]
    results_dir = config["results_dir"]

    rankings_path = get_result_path(results_dir, "phiid_scores", model_id, "random_head_rankings.csv")

    if os.path.exists(rankings_path):
        logger.info("Random baseline rankings already exist. Skipping.")
        return

    logger.info(f"Initializing {config['model_name']} with random weights...")
    local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    model_config = AutoConfig.from_pretrained(config["model_name"], local_files_only=local_only)
    model = AutoModelForCausalLM.from_config(model_config)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    spec = detect_model_spec(model)
    extractor = HeadActivationExtractor(model, tokenizer, device, model_spec=spec)

    logger.info("Extracting activations from random model...")
    all_activations, _, _ = extractor.extract_all_prompts(
        PROMPTS, num_tokens=config["num_tokens_to_generate"]
    )
    logger.info(f"Random activations shape: {all_activations.shape}")

    # Save random activations
    rand_act_path = get_result_path(results_dir, "activations", model_id, "random_activations.npz")
    np.savez(rand_act_path, activations=all_activations)

    # Free model
    del model, extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute PhiID
    phiid_path = get_result_path(results_dir, "phiid_scores", model_id, "random_phiid.npz")
    num_heads = all_activations.shape[1]

    if os.path.exists(phiid_path):
        logger.info("Random PhiID already computed. Loading...")
        data = np.load(phiid_path)
        sts_matrix = data['sts_matrix']
        rtr_matrix = data['rtr_matrix']
    else:
        max_workers = min(multiprocessing.cpu_count(), 32)
        sts_matrix, rtr_matrix = compute_all_pairs_phiid(
            activations=all_activations,
            num_heads=num_heads,
            tau=config["phiid_tau"],
            kind=config["phiid_kind"],
            redundancy=config["phiid_redundancy"],
            max_workers=max_workers,
            save_path=phiid_path,
            checkpoint_interval=5000,
        )

    # Rank heads
    num_heads_per_layer, num_layers = _infer_architecture(num_heads, model_id)
    synergy_per_head, redundancy_per_head = compute_head_scores(sts_matrix, rtr_matrix, num_heads)
    syn_red_rank, syn_rank, red_rank = compute_syn_red_rank(synergy_per_head, redundancy_per_head)
    df = build_ranking_dataframe(
        synergy_per_head, redundancy_per_head,
        syn_red_rank, syn_rank, red_rank,
        num_layers, num_heads_per_layer,
    )
    df.to_csv(rankings_path, index=False)
    logger.info(f"Saved random head rankings to {rankings_path}")


def phase6_visualize(config, revision=None):
    """Phase 6: Generate visualization figures."""
    model_id = config["model_id"]
    results_dir = config["results_dir"]
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rev_suffix = f"_{revision}" if revision else ""
    title_prefix = f"{model_id}{rev_suffix}"

    rankings_path = get_result_path(results_dir, "phiid_scores", model_id, "head_rankings.csv", revision)
    if not os.path.exists(rankings_path):
        logger.error(f"Head rankings not found at {rankings_path}. Run phase 3 first.")
        return

    trained_df = pd.read_csv(rankings_path)
    num_layers = trained_df['layer'].nunique()
    num_heads_per_layer = trained_df['head_in_layer'].nunique()

    # Fig 2c: PhiID profile
    plot_phiid_profile(
        trained_df,
        title=f"{title_prefix}: Syn-Red Score by Layer Depth",
        save_path=os.path.join(fig_dir, f"{model_id}{rev_suffix}_phiid_profile.png"),
    )

    # Fig 2a: Synergy/Redundancy heatmaps
    phiid_path = get_result_path(results_dir, "phiid_scores", model_id, "pairwise_phiid.npz", revision)
    if os.path.exists(phiid_path):
        data = np.load(phiid_path)
        plot_synergy_redundancy_heatmaps(
            data['sts_matrix'], data['rtr_matrix'],
            title_prefix=title_prefix,
            save_path=os.path.join(fig_dir, f"{model_id}{rev_suffix}_synergy_redundancy.png"),
        )

    # Fig 2b: Head ranking heatmap
    plot_head_ranking_heatmap(
        trained_df,
        num_layers=num_layers,
        num_heads_per_layer=num_heads_per_layer,
        title=f"{title_prefix}: Syn-Red Score per Head",
        save_path=os.path.join(fig_dir, f"{model_id}{rev_suffix}_head_ranking_heatmap.png"),
    )

    # Fig 4a: Ablation curves
    ablation_path = get_result_path(results_dir, "ablation", model_id, "ablation.csv", revision)
    if os.path.exists(ablation_path):
        ablation_df = pd.read_csv(ablation_path)
        plot_ablation_curves(
            ablation_df,
            title=f"{title_prefix}: Ablation — Synergistic vs Random Order",
            save_path=os.path.join(fig_dir, f"{model_id}{rev_suffix}_ablation_curves.png"),
        )

    # Fig 3a: Trained vs Random
    random_rankings_path = get_result_path(results_dir, "phiid_scores", model_id, "random_head_rankings.csv")
    if os.path.exists(random_rankings_path):
        random_df = pd.read_csv(random_rankings_path)
        plot_trained_vs_random(
            trained_df, random_df,
            title=f"{title_prefix}: Trained vs Random Initialization",
            save_path=os.path.join(fig_dir, f"{model_id}{rev_suffix}_trained_vs_random.png"),
        )

    logger.info(f"All figures saved to {fig_dir}/")


def _infer_architecture(total_heads, model_id):
    """Infer (num_heads_per_layer, num_layers) from total_heads and model_id."""
    known = {
        "pythia1b": (8, 16),
        "gemma3_4b": (8, 34),
        "qwen3_8b": (32, 36),
    }
    if model_id in known:
        return known[model_id]

    # Fallback: try common factors
    for nhl in [32, 16, 12, 8, 6, 4]:
        if total_heads % nhl == 0:
            return (nhl, total_heads // nhl)
    return (1, total_heads)


def main():
    parser = argparse.ArgumentParser(description="Multi-model PhiID pipeline runner")
    parser.add_argument("--model", required=True,
                        choices=["pythia-1b", "gemma3-4b", "qwen3-8b"],
                        help="Model to analyze")
    parser.add_argument("--revision", default=None,
                        help="Model revision (e.g., step1000 for Pythia checkpoints)")
    parser.add_argument("--phases", nargs='+', type=int, required=True,
                        help="Phases to run (1-6)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers for PhiID computation")
    args = parser.parse_args()

    setup_logging()

    # Load .env file (HF_TOKEN, etc.) — won't overwrite existing env vars
    load_dotenv()

    config = get_config(args.model)
    set_seed(config["seed"])
    ensure_dirs(config["results_dir"])

    hf_token = os.environ.get("HF_TOKEN")
    device = get_device(config["device"])

    logger.info(f"Pipeline: model={args.model}, revision={args.revision}, "
                f"phases={args.phases}, device={device}")

    phase_map = {
        1: lambda: phase1_extract(config, device, args.revision, hf_token),
        2: lambda: phase2_phiid(config, args.revision, args.max_workers),
        3: lambda: phase3_rank(config, args.revision),
        4: lambda: phase4_ablation(config, device, args.revision, hf_token),
        5: lambda: phase5_random_baseline(config, device, args.revision),
        6: lambda: phase6_visualize(config, args.revision),
    }

    for phase_num in args.phases:
        if phase_num not in phase_map:
            logger.error(f"Unknown phase: {phase_num}. Valid phases: 1-6")
            continue
        logger.info(f"{'='*60}")
        logger.info(f"Starting Phase {phase_num}")
        logger.info(f"{'='*60}")
        try:
            phase_map[phase_num]()
        except Exception as e:
            logger.error(f"Phase {phase_num} failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    main()
