#!/usr/bin/env python
"""
Test concatenated-prompts PhiID computation.

Instead of computing PhiID per prompt (100 timepoints) and averaging,
concatenate all 60 prompts into one long time series (6000 points) per head
and run PhiID once per pair. This should:
- Reduce lag-1 autocorrelation effects
- Give PhiID more statistical power
- Potentially produce the inverted-U profile and negative syn-red correlation

Usage:
  python scripts/test_concat_phiid.py --model gemma3-4b-it --max-workers 32
"""

import argparse
import logging
import os
import sys
import time
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.head_ranking import (
    compute_head_scores, compute_syn_red_rank, compute_pair_balance_scores,
    get_head_layer_mapping, build_ranking_dataframe,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Model architectures: model_id -> (num_heads_per_layer, num_layers)
ARCHITECTURES = {
    "pythia1b": (8, 16),
    "gemma3_4b": (8, 34),
    "gemma3_4b_it": (8, 34),
    "gemma3_12b_it": (16, 48),
    "qwen3_8b": (32, 36),
}

MODEL_ID_MAP = {
    "pythia-1b": "pythia1b",
    "gemma3-4b": "gemma3_4b",
    "gemma3-4b-it": "gemma3_4b_it",
    "gemma3-12b-it": "gemma3_12b_it",
    "qwen3-8b": "qwen3_8b",
}


def _compute_single_pair_concat(args):
    """Compute PhiID for a single pair on concatenated time series."""
    from phyid.calculate import calc_PhiID

    series_i, series_j, tau, kind, redundancy = args

    try:
        atoms, _ = calc_PhiID(series_i, series_j, tau=tau, kind=kind, redundancy=redundancy)
        sts_val = float(np.mean(atoms['sts']))
        rtr_val = float(np.mean(atoms['rtr']))
    except Exception as e:
        logger.warning(f"PhiID failed: {e}")
        sts_val = 0.0
        rtr_val = 0.0

    return sts_val, rtr_val


def main():
    parser = argparse.ArgumentParser(description="Test concatenated-prompts PhiID")
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_ID_MAP.keys()))
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Parallel workers for PhiID (default: auto)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Where to read activations from')
    parser.add_argument('--output-dir', type=str,
                        default='testing_approaches/02_concat_prompts',
                        help='Where to write output files')
    args = parser.parse_args()

    model_id = MODEL_ID_MAP[args.model]
    num_heads_per_layer, num_layers = ARCHITECTURES[model_id]
    num_heads = num_layers * num_heads_per_layer

    # Load activations
    act_path = os.path.join(args.results_dir, 'activations', f'{model_id}_activations.npz')
    logger.info(f"Loading activations from {act_path}")
    data = np.load(act_path)
    activations = data['activations']  # (num_prompts, num_heads, num_steps)
    logger.info(f"Activations shape: {activations.shape}")

    num_prompts, n_heads_check, num_steps = activations.shape
    assert n_heads_check == num_heads, f"Expected {num_heads} heads, got {n_heads_check}"
    logger.info(f"Model: {args.model} ({num_layers}L x {num_heads_per_layer}H = {num_heads} heads)")
    logger.info(f"Prompts: {num_prompts}, Steps per prompt: {num_steps}")
    logger.info(f"Concatenated length: {num_prompts * num_steps}")

    # Concatenate all prompts per head: (num_heads, num_prompts * num_steps)
    concat_activations = activations.transpose(1, 0, 2).reshape(num_heads, -1)
    concat_length = concat_activations.shape[1]
    logger.info(f"Concatenated activations shape: {concat_activations.shape}")

    # Quick diagnostic: check autocorrelation before and after concatenation
    logger.info("\n=== Autocorrelation diagnostics ===")
    sample_heads = np.random.choice(num_heads, min(10, num_heads), replace=False)
    per_prompt_autocorrs = []
    concat_autocorrs = []
    for h in sample_heads:
        # Per-prompt autocorrelation (average across prompts)
        prompt_acs = []
        for p in range(num_prompts):
            ts = activations[p, h, :]
            if np.std(ts) > 1e-10:
                ac = np.corrcoef(ts[:-1], ts[1:])[0, 1]
                prompt_acs.append(ac)
        per_prompt_autocorrs.append(np.mean(prompt_acs))

        # Concatenated autocorrelation
        cts = concat_activations[h]
        if np.std(cts) > 1e-10:
            ac = np.corrcoef(cts[:-1], cts[1:])[0, 1]
            concat_autocorrs.append(ac)

    logger.info(f"Per-prompt lag-1 autocorrelation: {np.mean(per_prompt_autocorrs):.4f}")
    logger.info(f"Concatenated lag-1 autocorrelation: {np.mean(concat_autocorrs):.4f}")

    # Compute PhiID for all pairs on concatenated series
    all_pairs = list(combinations(range(num_heads), 2))
    total_pairs = len(all_pairs)
    logger.info(f"\nComputing PhiID for {total_pairs} pairs on {concat_length}-point series...")

    # Output paths — write to testing_approaches dir, not results/
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{model_id}_concat_pairwise_phiid.npz')
    ckpt_path = save_path.replace('.npz', '_checkpoint.npz')

    # Check for checkpoint
    sts_matrix = np.zeros((num_heads, num_heads))
    rtr_matrix = np.zeros((num_heads, num_heads))
    completed = 0

    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        sts_matrix = ckpt['sts_matrix']
        rtr_matrix = ckpt['rtr_matrix']
        completed = int(ckpt['completed'])
        logger.info(f"Resuming from checkpoint: {completed}/{total_pairs} pairs done")

    remaining_pairs = all_pairs[completed:]

    # Build work items
    work_items = []
    for (i, j) in remaining_pairs:
        work_items.append((
            concat_activations[i],  # shape (concat_length,)
            concat_activations[j],
            1,  # tau
            "gaussian",
            "MMI",
        ))

    start_time = time.time()
    checkpoint_interval = 5000

    if args.max_workers == 1:
        for idx, (i, j) in enumerate(tqdm(remaining_pairs, desc="PhiID (concat)",
                                           initial=completed, total=total_pairs)):
            sts_val, rtr_val = _compute_single_pair_concat(work_items[idx])
            sts_matrix[i, j] = sts_val
            sts_matrix[j, i] = sts_val
            rtr_matrix[i, j] = rtr_val
            rtr_matrix[j, i] = rtr_val

            if (completed + idx + 1) % checkpoint_interval == 0:
                np.savez(ckpt_path, sts_matrix=sts_matrix, rtr_matrix=rtr_matrix,
                         completed=np.array(completed + idx + 1))
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(remaining_pairs) - idx - 1) / rate
                logger.info(f"Checkpoint: {completed + idx + 1}/{total_pairs}, "
                            f"rate: {rate:.1f} pairs/s, ETA: {remaining/60:.0f} min")
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_pair = {}
            for idx, item in enumerate(work_items):
                future = executor.submit(_compute_single_pair_concat, item)
                future_to_pair[future] = (remaining_pairs[idx], completed + idx)

            pbar = tqdm(total=total_pairs, initial=completed, desc="PhiID (concat)")
            done_count = completed
            for future in as_completed(future_to_pair):
                (i, j), pair_idx = future_to_pair[future]
                try:
                    sts_val, rtr_val = future.result()
                    sts_matrix[i, j] = sts_val
                    sts_matrix[j, i] = sts_val
                    rtr_matrix[i, j] = rtr_val
                    rtr_matrix[j, i] = rtr_val
                except Exception as e:
                    logger.warning(f"Pair ({i},{j}) failed: {e}")

                done_count += 1
                pbar.update(1)

                if done_count % checkpoint_interval == 0:
                    np.savez(ckpt_path, sts_matrix=sts_matrix, rtr_matrix=rtr_matrix,
                             completed=np.array(done_count))
                    elapsed = time.time() - start_time
                    rate = (done_count - completed) / elapsed
                    remaining_est = (total_pairs - done_count) / rate if rate > 0 else 0
                    logger.info(f"Checkpoint: {done_count}/{total_pairs}, "
                                f"rate: {rate:.1f} pairs/s, ETA: {remaining_est/60:.0f} min")

            pbar.close()

    # Save final results
    np.savez(save_path, sts_matrix=sts_matrix, rtr_matrix=rtr_matrix)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    elapsed = time.time() - start_time
    logger.info(f"PhiID computation done in {elapsed/60:.1f} minutes")

    # === Analysis ===
    logger.info("\n=== Analysis of concatenated PhiID results ===")

    # 1. Syn-red correlation
    upper_tri = np.triu_indices(num_heads, k=1)
    sts_vals = sts_matrix[upper_tri]
    rtr_vals = rtr_matrix[upper_tri]
    corr, pval = pearsonr(sts_vals, rtr_vals)
    logger.info(f"Pairwise syn-red correlation: r={corr:.4f}, p={pval:.2e}")

    # 2. Standard ranking (rank_diff)
    synergy_per_head, redundancy_per_head = compute_head_scores(
        sts_matrix, rtr_matrix, num_heads)
    syn_red_rank, syn_rank, red_rank = compute_syn_red_rank(
        synergy_per_head, redundancy_per_head)

    head_corr, _ = pearsonr(synergy_per_head, redundancy_per_head)
    logger.info(f"Per-head syn-red correlation: r={head_corr:.4f}")

    # 3. Balanced ranking
    syn_balance, syn_red_balanced = compute_pair_balance_scores(
        sts_matrix, rtr_matrix, num_heads)

    # 4. Per-layer profiles
    mapping = get_head_layer_mapping(num_layers, num_heads_per_layer)

    logger.info(f"\n{'Layer':>5}  {'Std Score':>9}  {'Bal Score':>9}")
    logger.info("-" * 30)
    for layer in range(num_layers):
        heads_in_layer = [h for h in range(num_heads) if mapping[h][0] == layer]
        std_mean = np.mean([syn_red_rank[h] for h in heads_in_layer])
        bal_mean = np.mean([syn_red_balanced[h] for h in heads_in_layer])
        logger.info(f"{layer:5d}  {std_mean:9.4f}  {bal_mean:9.4f}")

    # 5. Inverted-U test: correlation with quadratic
    layer_depth_norm = np.array([mapping[h][0] / (num_layers - 1) for h in range(num_heads)])
    quadratic = -(layer_depth_norm - 0.5)**2 + 0.25  # peaks at 0.5

    std_quad_corr, _ = pearsonr(syn_red_rank, quadratic)
    bal_quad_corr, _ = pearsonr(syn_red_balanced, quadratic)
    logger.info(f"\nCorrelation with inverted-U quadratic:")
    logger.info(f"  Standard (rank_diff): r={std_quad_corr:.4f}")
    logger.info(f"  Balanced (pair_balance): r={bal_quad_corr:.4f}")

    # 6. Save rankings
    df_std = build_ranking_dataframe(
        synergy_per_head, redundancy_per_head, syn_red_rank, syn_rank, red_rank,
        num_layers, num_heads_per_layer)
    std_csv = os.path.join(output_dir, f'{model_id}_head_rankings_standard.csv')
    df_std.to_csv(std_csv, index=False)
    logger.info(f"Saved standard rankings to {std_csv}")

    # Balanced rankings
    rows = []
    for h in range(num_heads):
        layer, head_in_layer = mapping[h]
        rows.append({
            "head_idx": h,
            "layer": layer,
            "head_in_layer": head_in_layer,
            "avg_synergy": synergy_per_head[h],
            "avg_redundancy": redundancy_per_head[h],
            "syn_balance": syn_balance[h],
            "syn_red_score": syn_red_balanced[h],
        })
    df_bal = pd.DataFrame(rows)
    bal_csv = os.path.join(output_dir, f'{model_id}_head_rankings_balanced.csv')
    df_bal.to_csv(bal_csv, index=False)
    logger.info(f"Saved balanced rankings to {bal_csv}")

    # 7. Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Profile plot (both methods)
        layer_std = df_std.groupby('layer')['syn_red_score'].mean()
        layer_bal = df_bal.groupby('layer')['syn_red_score'].mean()
        x = np.array(layer_std.index) / (num_layers - 1)

        axes[0].plot(x, layer_std.values, 'b-o', label=f'Standard (r_quad={std_quad_corr:.3f})', markersize=3)
        axes[0].plot(x, layer_bal.values, 'r-s', label=f'Balanced (r_quad={bal_quad_corr:.3f})', markersize=3)
        axes[0].set_xlabel('Normalized Layer Depth')
        axes[0].set_ylabel('Mean Syn-Red Score')
        axes[0].set_title(f'{args.model} — Concatenated PhiID Profile')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Syn vs Red scatter
        axes[1].scatter(synergy_per_head, redundancy_per_head, alpha=0.5, s=10)
        axes[1].set_xlabel('Avg Synergy (sts)')
        axes[1].set_ylabel('Avg Redundancy (rtr)')
        axes[1].set_title(f'Per-head Syn vs Red (r={head_corr:.3f})')
        axes[1].grid(True, alpha=0.3)

        # Pairwise syn vs red (subsample)
        n_show = min(5000, len(sts_vals))
        idx = np.random.choice(len(sts_vals), n_show, replace=False)
        axes[2].scatter(sts_vals[idx], rtr_vals[idx], alpha=0.1, s=5)
        axes[2].set_xlabel('Pairwise Synergy (sts)')
        axes[2].set_ylabel('Pairwise Redundancy (rtr)')
        axes[2].set_title(f'Pairwise Syn vs Red (r={corr:.3f})')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'{model_id}_concat_phiid_analysis.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {fig_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    # 8. Graph visualization — Fig 3b (top 10% edges)
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        logger.info("\n=== Graph Visualization (Fig 3b) ===")

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        _, node_colors = zip(*[(h // num_heads_per_layer, (h // num_heads_per_layer) / max(num_layers - 1, 1))
                                for h in range(num_heads)])
        node_colors = np.array(node_colors)

        for ax, matrix, label, edge_cmap in [
            (axes[0], sts_matrix, "Synergistic Core (top 10% sts)", "Reds"),
            (axes[1], rtr_matrix, "Redundant Core (top 10% rtr)", "Blues"),
        ]:
            # Build thresholded graph
            triu_idx = np.triu_indices(num_heads, k=1)
            weights = matrix[triu_idx]
            num_keep = max(1, int(len(weights) * 0.10))
            threshold = np.sort(weights)[-num_keep]

            G = nx.Graph()
            G.add_nodes_from(range(num_heads))
            for idx_e in range(len(weights)):
                if weights[idx_e] >= threshold:
                    i_n, j_n = triu_idx[0][idx_e], triu_idx[1][idx_e]
                    G.add_edge(i_n, j_n, weight=float(weights[idx_e]))

            logger.info(f"{label}: {G.number_of_edges()} edges, "
                        f"{nx.number_connected_components(G)} components")

            # Layout: circular, ordered by layer
            pos = {}
            for node in range(num_heads):
                angle = 2 * np.pi * node / num_heads
                pos[node] = (np.cos(angle), np.sin(angle))

            # Draw edges
            if G.number_of_edges() > 0:
                edge_weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
                w_min, w_max = edge_weights.min(), edge_weights.max()
                if w_max > w_min:
                    edge_alphas = 0.05 + 0.5 * (edge_weights - w_min) / (w_max - w_min)
                    edge_widths = 0.2 + 1.2 * (edge_weights - w_min) / (w_max - w_min)
                else:
                    edge_alphas = np.full_like(edge_weights, 0.3)
                    edge_widths = np.full_like(edge_weights, 0.5)

                cmap_edge = plt.cm.get_cmap(edge_cmap)
                for idx_e, (u, v, d) in enumerate(G.edges(data=True)):
                    ax.plot(
                        [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color=cmap_edge(0.5),
                        alpha=float(edge_alphas[idx_e]),
                        linewidth=float(edge_widths[idx_e]),
                        zorder=1,
                    )

            # Draw nodes
            node_x = [pos[n][0] for n in range(num_heads)]
            node_y = [pos[n][1] for n in range(num_heads)]
            sc = ax.scatter(
                node_x, node_y, c=node_colors, cmap='viridis',
                s=30, zorder=2, edgecolors='white', linewidths=0.3,
            )
            ax.set_title(f"{args.model}: {label}", fontsize=12)
            ax.set_aspect('equal')
            ax.axis('off')

            # Log degree distribution by layer
            degrees = dict(G.degree())
            for layer in range(num_layers):
                heads_in_l = [h for h in range(num_heads) if h // num_heads_per_layer == layer]
                mean_deg = np.mean([degrees[h] for h in heads_in_l])
                if mean_deg > 0:
                    logger.info(f"  Layer {layer:2d}: mean degree = {mean_deg:.1f}")

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap='viridis',
                                  norm=mcolors.Normalize(0, num_layers - 1)),
            ax=axes, fraction=0.02, pad=0.04,
        )
        cbar.set_label('Layer', fontsize=11)

        fig.suptitle(f"{args.model}: Concatenated PhiID — Synergistic & Redundant Cores (top 10%)",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        graph_path = os.path.join(output_dir, f'{model_id}_graph_cores.png')
        fig.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved graph cores figure to {graph_path}")

    except ImportError:
        logger.warning("networkx not installed — skipping graph visualization. "
                       "Install with: pip install networkx")
    except Exception as e:
        logger.warning(f"Graph visualization failed: {e}")

    logger.info("\n=== SUMMARY ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Method: Concatenated prompts ({num_prompts} x {num_steps} = {concat_length} points)")
    logger.info(f"Pairwise syn-red correlation: r={corr:.4f}")
    logger.info(f"Per-head syn-red correlation: r={head_corr:.4f}")
    logger.info(f"Inverted-U correlation (standard): r={std_quad_corr:.4f}")
    logger.info(f"Inverted-U correlation (balanced): r={bal_quad_corr:.4f}")


if __name__ == '__main__':
    main()
