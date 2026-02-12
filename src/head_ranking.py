"""
Head ranking by synergy-redundancy score.

For each head, average its synergy/redundancy across all pairs that include it,
then compute syn_red_score = synergy_rank - redundancy_rank, min-max normalized.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


def get_head_layer_mapping(num_layers, num_heads_per_layer):
    """
    Map flat head index to (layer_idx, head_within_layer_idx).
    E.g., head 0-15 -> layer 0, head 16-31 -> layer 1, etc.
    """
    mapping = []
    for layer in range(num_layers):
        for head in range(num_heads_per_layer):
            mapping.append((layer, head))
    return mapping


def compute_head_scores(sts_matrix, rtr_matrix, num_heads):
    """
    For each head, compute its average synergy and redundancy
    across all pairs that include it.

    Args:
        sts_matrix: np.ndarray (num_heads, num_heads) — pairwise synergy
        rtr_matrix: np.ndarray (num_heads, num_heads) — pairwise redundancy
        num_heads: total number of heads

    Returns:
        synergy_per_head: np.ndarray of shape (num_heads,)
        redundancy_per_head: np.ndarray of shape (num_heads,)
    """
    synergy_per_head = np.zeros(num_heads)
    redundancy_per_head = np.zeros(num_heads)

    for h in range(num_heads):
        # All pairs involving head h (row h and column h, excluding diagonal)
        mask = np.ones(num_heads, dtype=bool)
        mask[h] = False
        synergy_per_head[h] = np.mean(sts_matrix[h, mask])
        redundancy_per_head[h] = np.mean(rtr_matrix[h, mask])

    return synergy_per_head, redundancy_per_head


def compute_syn_red_rank(synergy_per_head, redundancy_per_head):
    """
    Paper's ranking method:
    1. Rank all heads by synergy (ordinal rank, 1 = lowest synergy)
    2. Rank all heads by redundancy (ordinal rank, 1 = lowest redundancy)
    3. syn_red_score = synergy_rank - redundancy_rank
    4. Min-max normalize: (score - min) / (max - min) -> range [0, 1]

    High score = more synergistic, Low score = more redundant.

    Returns:
        syn_red_rank: np.ndarray of shape (num_heads,) in [0, 1]
    """
    # Ordinal ranking (1 = lowest value)
    syn_rank = rankdata(synergy_per_head, method='ordinal')
    red_rank = rankdata(redundancy_per_head, method='ordinal')

    # Syn-red score
    raw_score = syn_rank - red_rank

    # Min-max normalize to [0, 1]
    score_min = raw_score.min()
    score_max = raw_score.max()
    if score_max > score_min:
        syn_red_rank = (raw_score - score_min) / (score_max - score_min)
    else:
        syn_red_rank = np.full_like(raw_score, 0.5, dtype=float)

    return syn_red_rank, syn_rank, red_rank


def build_ranking_dataframe(synergy_per_head, redundancy_per_head,
                            syn_red_rank, syn_rank, red_rank,
                            num_layers, num_heads_per_layer):
    """
    Build a DataFrame with head ranking information.

    Returns:
        DataFrame with columns:
            head_idx, layer, head_in_layer, avg_synergy, avg_redundancy,
            syn_rank, red_rank, syn_red_score
    """
    mapping = get_head_layer_mapping(num_layers, num_heads_per_layer)
    num_heads = len(mapping)

    data = []
    for h in range(num_heads):
        layer, head_in_layer = mapping[h]
        data.append({
            'head_idx': h,
            'layer': layer,
            'head_in_layer': head_in_layer,
            'avg_synergy': synergy_per_head[h],
            'avg_redundancy': redundancy_per_head[h],
            'syn_rank': int(syn_rank[h]),
            'red_rank': int(red_rank[h]),
            'syn_red_score': syn_red_rank[h],
        })

    df = pd.DataFrame(data)

    # Log per-layer stats
    layer_means = df.groupby('layer')['syn_red_score'].mean()
    logger.info("Per-layer average syn_red_score:")
    for layer, score in layer_means.items():
        logger.info(f"  Layer {layer:2d}: {score:.4f}")

    return df
