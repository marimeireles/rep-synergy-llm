"""
Visualization functions for the PhiID analysis.

Reproduces figures from the paper:
- Fig 2a: Synergy and redundancy heatmaps
- Fig 2b: Head ranking heatmap (layers x heads)
- Fig 2c: PhiID profile across layers (inverted-U)
- Fig 3a: Trained vs random comparison / checkpoint progression
- Fig 3b: Graph representation of synergistic/redundant cores
- Fig 4a: Ablation curves
- Fig 4b: MATH benchmark perturbation bar chart
- Multi-model overlays
"""

import logging
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_phiid_profile(head_rankings_df, title, save_path):
    """
    Fig 2c: Average syn_red_score per layer vs normalized layer depth.
    Expected: inverted-U shape.
    """
    layer_means = head_rankings_df.groupby('layer')['syn_red_score'].mean()
    num_layers = len(layer_means)
    x = np.arange(num_layers) / (num_layers - 1)  # normalized to [0, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, layer_means.values, 'o-', color='darkred', linewidth=2, markersize=6)
    ax.set_xlabel('Normalized Layer Depth', fontsize=12)
    ax.set_ylabel('Average Syn-Red Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved PhiID profile to {save_path}")


def plot_synergy_redundancy_heatmaps(sts_matrix, rtr_matrix, title_prefix, save_path):
    """
    Fig 2a: Two heatmaps showing pairwise synergy and redundancy between all heads.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Synergy heatmap
    im1 = ax1.imshow(sts_matrix, aspect='auto', cmap='Reds')
    ax1.set_title(f'{title_prefix} — Synergy (sts)', fontsize=12)
    ax1.set_xlabel('Head Index')
    ax1.set_ylabel('Head Index')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Redundancy heatmap
    im2 = ax2.imshow(rtr_matrix, aspect='auto', cmap='Blues')
    ax2.set_title(f'{title_prefix} — Redundancy (rtr)', fontsize=12)
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Head Index')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved synergy/redundancy heatmaps to {save_path}")


def plot_head_ranking_heatmap(head_rankings_df, num_layers, num_heads_per_layer,
                               title, save_path):
    """
    Fig 2b: Heatmap of layers x heads_per_layer colored by syn_red_score.
    Red = synergistic, Blue = redundant.
    """
    grid = np.zeros((num_layers, num_heads_per_layer))
    for _, row in head_rankings_df.iterrows():
        grid[int(row['layer']), int(row['head_in_layer'])] = row['syn_red_score']

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Blue to Red
    im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(num_heads_per_layer))
    ax.set_yticks(range(num_layers))
    fig.colorbar(im, ax=ax, label='Syn-Red Score')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved head ranking heatmap to {save_path}")


def plot_ablation_curves(ablation_df, title, save_path):
    """
    Fig 4a: Ablation curves.
    X-axis: fraction of heads deactivated.
    Y-axis: KL divergence (behaviour divergence).
    Solid line: synergistic order. Dashed line: random order with shaded std.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Total heads for fraction computation
    total_heads = ablation_df['num_heads_removed'].max()

    # Synergistic order
    syn_data = ablation_df[ablation_df['order_type'] == 'syn_red']
    if not syn_data.empty:
        x_syn = syn_data['num_heads_removed'].values / total_heads
        y_syn = syn_data['mean_kl_div'].values
        ax.plot(x_syn, y_syn, '-', color='red', linewidth=2, label='Synergistic order')

    # Random order (may have multiple seeds -> compute mean +/- std)
    random_data = ablation_df[ablation_df['order_type'].str.startswith('random')]
    if not random_data.empty:
        # Group by num_heads_removed across random seeds
        random_grouped = random_data.groupby('num_heads_removed')['mean_kl_div']
        x_rand = np.array(sorted(random_data['num_heads_removed'].unique())) / total_heads
        y_mean = random_grouped.mean().values
        y_std = random_grouped.std().values

        ax.plot(x_rand, y_mean, '--', color='gray', linewidth=2, label='Random order')
        ax.fill_between(x_rand, y_mean - y_std, y_mean + y_std,
                        color='gray', alpha=0.2)

    ax.set_xlabel('Fraction of Heads Deactivated', fontsize=12)
    ax.set_ylabel('Behaviour Divergence (KL)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved ablation curves to {save_path}")


def plot_trained_vs_random(trained_df, random_df, title, save_path):
    """
    Fig 3a: Overlaid PhiID profiles for trained vs random model.
    Trained: inverted-U. Random: flat.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, label, color, marker in [
        (trained_df, 'Trained', 'darkred', 'o'),
        (random_df, 'Random init', 'gray', 's'),
    ]:
        layer_means = df.groupby('layer')['syn_red_score'].mean()
        num_layers = len(layer_means)
        x = np.arange(num_layers) / (num_layers - 1)
        ax.plot(x, layer_means.values, f'{marker}-', color=color,
                linewidth=2, markersize=6, label=label)

    ax.set_xlabel('Normalized Layer Depth', fontsize=12)
    ax.set_ylabel('Average Syn-Red Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved trained vs random comparison to {save_path}")


def plot_checkpoint_progression(checkpoint_rankings, title, save_path):
    """
    Fig 3a (training progression): Color-coded lines from early (light) to
    late (dark) training checkpoints.

    Args:
        checkpoint_rankings: dict mapping checkpoint name -> DataFrame
            e.g., {"step1": df1, "step64": df2, ...}
        title: figure title
        save_path: output path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort checkpoints by step number
    def _step_num(name):
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else 0

    sorted_ckpts = sorted(checkpoint_rankings.keys(), key=_step_num)
    n_ckpts = len(sorted_ckpts)

    # Color gradient: light to dark red
    cmap = plt.cm.Reds
    colors = [cmap(0.2 + 0.7 * i / max(n_ckpts - 1, 1)) for i in range(n_ckpts)]

    for idx, ckpt_name in enumerate(sorted_ckpts):
        df = checkpoint_rankings[ckpt_name]
        layer_means = df.groupby('layer')['syn_red_score'].mean()
        num_layers = len(layer_means)
        x = np.arange(num_layers) / (num_layers - 1)

        step_num = _step_num(ckpt_name)
        label = f"Step {step_num:,}"

        ax.plot(x, layer_means.values, 'o-', color=colors[idx],
                linewidth=1.5, markersize=4, label=label, alpha=0.9)

    ax.set_xlabel('Normalized Layer Depth', fontsize=12)
    ax.set_ylabel('Average Syn-Red Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved checkpoint progression to {save_path}")


def plot_multi_model_profiles(model_rankings, title, save_path):
    """
    Fig 2c multi-model: Overlaid PhiID profiles for multiple models.

    Args:
        model_rankings: dict mapping model name -> DataFrame
            e.g., {"Pythia-1B": df1, "Gemma 3 4B": df2, "Qwen 3 8B": df3}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    model_styles = {
        'Pythia-1B':  {'color': '#1f77b4', 'marker': 'o'},
        'Gemma 3 4B': {'color': '#d62728', 'marker': 's'},
        'Qwen 3 8B':  {'color': '#2ca02c', 'marker': '^'},
    }

    for model_name, df in model_rankings.items():
        layer_means = df.groupby('layer')['syn_red_score'].mean()
        num_layers = len(layer_means)
        x = np.arange(num_layers) / (num_layers - 1)

        style = model_styles.get(model_name, {'color': 'gray', 'marker': 'D'})
        ax.plot(x, layer_means.values, f"{style['marker']}-",
                color=style['color'], linewidth=2, markersize=5,
                label=model_name, alpha=0.9)

    ax.set_xlabel('Normalized Layer Depth', fontsize=12)
    ax.set_ylabel('Average Syn-Red Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved multi-model profiles to {save_path}")


def plot_multi_model_ablation(model_ablations, title, save_path):
    """
    Fig 4a multi-model: Ablation curves for multiple models.

    Args:
        model_ablations: dict mapping model name -> DataFrame
            Each DataFrame has columns: num_heads_removed, mean_kl_div, order_type
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    model_colors = {
        'Pythia-1B':  '#1f77b4',
        'Gemma 3 4B': '#d62728',
        'Qwen 3 8B':  '#2ca02c',
    }

    for model_name, ablation_df in model_ablations.items():
        color = model_colors.get(model_name, 'gray')
        total_heads = ablation_df['num_heads_removed'].max()

        # Synergistic order
        syn_data = ablation_df[ablation_df['order_type'] == 'syn_red']
        if not syn_data.empty:
            x = syn_data['num_heads_removed'].values / total_heads
            y = syn_data['mean_kl_div'].values
            ax.plot(x, y, '-', color=color, linewidth=2,
                    label=f'{model_name} (syn)', alpha=0.9)

        # Random order mean
        random_data = ablation_df[ablation_df['order_type'].str.startswith('random')]
        if not random_data.empty:
            grouped = random_data.groupby('num_heads_removed')['mean_kl_div']
            x = np.array(sorted(random_data['num_heads_removed'].unique())) / total_heads
            y_mean = grouped.mean().values
            ax.plot(x, y_mean, '--', color=color, linewidth=1.5,
                    label=f'{model_name} (random)', alpha=0.6)

    ax.set_xlabel('Fraction of Heads Deactivated', fontsize=12)
    ax.set_ylabel('Behaviour Divergence (KL)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved multi-model ablation to {save_path}")


def plot_graph_cores(
    sts_matrix, rtr_matrix,
    num_layers, num_heads_per_layer,
    top_pct=0.1, title_prefix="", save_path=None,
):
    """
    Fig 3b: Visualize synergistic and redundant cores as undirected graphs.
    Only the top `top_pct` strongest connections are shown.

    Nodes are colored by layer depth (viridis: early=purple, late=yellow).
    Edge alpha/width scales with weight.

    Args:
        sts_matrix: (N, N) pairwise synergy matrix
        rtr_matrix: (N, N) pairwise redundancy matrix
        num_layers: number of layers
        num_heads_per_layer: heads per layer
        top_pct: fraction of strongest edges to keep (default 0.1)
        title_prefix: model name for titles
        save_path: output file path
    """
    try:
        import networkx as nx
        from src.graph_analysis import build_thresholded_graph, get_graph_layout, get_node_layer_colors
    except ImportError:
        logger.error("networkx not installed. Skipping graph core plot.")
        return

    N = sts_matrix.shape[0]
    _, node_colors = get_node_layer_colors(N, num_layers, num_heads_per_layer)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, matrix, label, edge_cmap in [
        (axes[0], sts_matrix, "Synergistic Core (top 10% synergy)", "Reds"),
        (axes[1], rtr_matrix, "Redundant Core (top 10% redundancy)", "Blues"),
    ]:
        G = build_thresholded_graph(matrix, top_pct=top_pct)
        pos = get_graph_layout(G, num_layers, num_heads_per_layer)

        # Edge weights for alpha scaling
        if G.number_of_edges() > 0:
            edge_weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
            w_min, w_max = edge_weights.min(), edge_weights.max()
            if w_max > w_min:
                edge_alphas = 0.1 + 0.7 * (edge_weights - w_min) / (w_max - w_min)
            else:
                edge_alphas = np.full_like(edge_weights, 0.4)
            edge_widths = 0.3 + 1.5 * (edge_weights - w_min) / max(w_max - w_min, 1e-10)
        else:
            edge_alphas = []
            edge_widths = []

        # Draw edges with individual alpha
        for idx, (u, v, d) in enumerate(G.edges(data=True)):
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color=plt.cm.get_cmap(edge_cmap)(0.5),
                alpha=float(edge_alphas[idx]) if len(edge_alphas) > 0 else 0.3,
                linewidth=float(edge_widths[idx]) if len(edge_widths) > 0 else 0.5,
                zorder=1,
            )

        # Draw nodes
        node_x = [pos[n][0] for n in range(N)]
        node_y = [pos[n][1] for n in range(N)]
        sc = ax.scatter(
            node_x, node_y,
            c=node_colors, cmap='viridis', s=30, zorder=2,
            edgecolors='white', linewidths=0.3,
        )

        ax.set_title(f"{title_prefix}: {label}", fontsize=12)
        ax.set_aspect('equal')
        ax.axis('off')

    # Shared colorbar for layer depth
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(0, num_layers - 1)),
        ax=axes, fraction=0.02, pad=0.04,
    )
    cbar.set_label('Layer', fontsize=11)

    fig.suptitle(f"{title_prefix}: Graph Representation of Synergistic & Redundant Cores",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved graph cores to {save_path}")


def plot_math_perturbation(results_df, title, save_path):
    """
    Fig 4b: Bar chart of MATH accuracy under different perturbation conditions.

    X-axis: Perturbation condition (Baseline, Synergistic, Redundant, Random)
    Y-axis: MATH accuracy (%)
    Random bars show error bars (std across seeds).

    Args:
        results_df: DataFrame with columns: condition, accuracy, seed (optional)
        title: figure title
        save_path: output file path
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions_order = ['baseline', 'synergistic', 'redundant', 'random']
    condition_labels = {
        'baseline': 'Baseline\n(no perturbation)',
        'synergistic': 'Synergistic\ncore (top 25%)',
        'redundant': 'Redundant\ncore (top 25%)',
        'random': 'Random\nsubset (25%)',
    }
    condition_colors = {
        'baseline': '#4a4a4a',
        'synergistic': '#d62728',
        'redundant': '#1f77b4',
        'random': '#7f7f7f',
    }

    x_positions = []
    bar_heights = []
    bar_errors = []
    bar_labels = []
    bar_colors = []

    for i, cond in enumerate(conditions_order):
        cond_data = results_df[results_df['condition'] == cond]
        if cond_data.empty:
            continue

        mean_acc = cond_data['accuracy'].mean() * 100
        std_acc = cond_data['accuracy'].std() * 100 if len(cond_data) > 1 else 0

        x_positions.append(i)
        bar_heights.append(mean_acc)
        bar_errors.append(std_acc)
        bar_labels.append(condition_labels.get(cond, cond))
        bar_colors.append(condition_colors.get(cond, 'gray'))

    bars = ax.bar(x_positions, bar_heights, yerr=bar_errors,
                  color=bar_colors, capsize=5, edgecolor='black', linewidth=0.5,
                  width=0.6, alpha=0.85)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel('MATH Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, h, err in zip(bars, bar_heights, bar_errors):
        label = f"{h:.1f}%"
        if err > 0:
            label += f"\n({err:.1f})"  # noqa: UP032
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.5,
                label, ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved MATH perturbation chart to {save_path}")
