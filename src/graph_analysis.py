"""
Graph-theoretic analysis of synergistic and redundant cores (Fig 3b, 3c).

Treats pairwise PhiID matrices as weighted adjacency matrices of undirected graphs.
Computes graph metrics (global efficiency, modularity) and provides data
structures for visualization.
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None
    logger.warning("networkx not installed. Install with: pip install networkx")

try:
    import community as community_louvain
except ImportError:
    community_louvain = None
    logger.warning("python-louvain not installed. Install with: pip install python-louvain")


def build_thresholded_graph(matrix: np.ndarray, top_pct: float = 0.1) -> "nx.Graph":
    """
    Build an undirected graph from a symmetric pairwise matrix,
    keeping only the top `top_pct` fraction of edges by weight.

    Args:
        matrix: (N, N) symmetric matrix (e.g., sts_matrix or rtr_matrix)
        top_pct: fraction of strongest edges to keep (0.1 = top 10%)

    Returns:
        G: networkx Graph with 'weight' edge attribute
    """
    if nx is None:
        raise ImportError("networkx is required. pip install networkx")

    N = matrix.shape[0]

    # Extract upper triangle values (undirected, no self-loops)
    triu_idx = np.triu_indices(N, k=1)
    weights = matrix[triu_idx]

    # Determine threshold for top_pct
    num_edges_total = len(weights)
    num_keep = max(1, int(num_edges_total * top_pct))
    threshold = np.sort(weights)[-num_keep]

    G = nx.Graph()
    G.add_nodes_from(range(N))

    for idx in range(num_edges_total):
        w = weights[idx]
        if w >= threshold:
            i, j = triu_idx[0][idx], triu_idx[1][idx]
            G.add_edge(i, j, weight=float(w))

    logger.info(
        f"Built graph: {N} nodes, {G.number_of_edges()} edges "
        f"(top {top_pct*100:.0f}%, threshold={threshold:.6f})"
    )
    return G


def compute_graph_metrics(
    sts_matrix: np.ndarray,
    rtr_matrix: np.ndarray,
    top_pct: float = 0.1,
) -> Dict[str, float]:
    """
    Compute graph-theoretic properties of synergy and redundancy networks (Fig 3c).

    Metrics:
        - Global efficiency: 1/N(N-1) * sum_{i!=j} 1/d(i,j)
          High efficiency = effective information integration.
          For weighted graphs, edge distances = 1/weight.
        - Modularity: Louvain community detection on the thresholded graph.
          High modularity = compartmentalized processing.

    Args:
        sts_matrix: (N, N) pairwise synergy matrix
        rtr_matrix: (N, N) pairwise redundancy matrix
        top_pct: fraction of strongest edges to keep

    Returns:
        dict with keys:
            syn_global_efficiency, red_global_efficiency,
            syn_modularity, red_modularity,
            syn_num_edges, red_num_edges,
            syn_num_components, red_num_components
    """
    if nx is None:
        raise ImportError("networkx is required. pip install networkx")

    results = {}

    for name, matrix in [("syn", sts_matrix), ("red", rtr_matrix)]:
        G = build_thresholded_graph(matrix, top_pct=top_pct)

        results[f"{name}_num_edges"] = G.number_of_edges()
        results[f"{name}_num_components"] = nx.number_connected_components(G)

        # Global efficiency using inverse weights as distances
        # For disconnected nodes, efficiency contribution is 0 (handled by nx)
        if G.number_of_edges() > 0:
            # Convert weights to distances: higher weight = shorter distance
            G_dist = G.copy()
            for u, v, d in G_dist.edges(data=True):
                d["distance"] = 1.0 / max(d["weight"], 1e-10)
            eff = nx.global_efficiency(G_dist, weight="distance")
        else:
            eff = 0.0
        results[f"{name}_global_efficiency"] = eff

        # Modularity via Louvain
        if community_louvain is not None and G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G, weight="weight", random_state=42)
            mod = community_louvain.modularity(partition, G, weight="weight")
            results[f"{name}_modularity"] = mod
            results[f"{name}_num_communities"] = len(set(partition.values()))
        else:
            results[f"{name}_modularity"] = float("nan")
            results[f"{name}_num_communities"] = 0

        logger.info(
            f"{name}: efficiency={results[f'{name}_global_efficiency']:.4f}, "
            f"modularity={results[f'{name}_modularity']:.4f}, "
            f"components={results[f'{name}_num_components']}, "
            f"edges={results[f'{name}_num_edges']}"
        )

    return results


def get_node_layer_colors(
    num_nodes: int, num_layers: int, num_heads_per_layer: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign layer index and color to each node.

    Returns:
        layer_ids: (N,) array of layer indices
        colors: (N,) array of normalized layer positions [0, 1] for colormap
    """
    layer_ids = np.array([i // num_heads_per_layer for i in range(num_nodes)])
    colors = layer_ids / max(num_layers - 1, 1)
    return layer_ids, colors


def get_graph_layout(
    G: "nx.Graph",
    num_layers: int,
    num_heads_per_layer: int,
    layout: str = "circular_by_layer",
) -> dict:
    """
    Compute node positions for graph visualization.

    Args:
        G: networkx Graph
        num_layers: number of layers
        num_heads_per_layer: heads per layer
        layout: "circular_by_layer" (default) or "spring"

    Returns:
        dict mapping node -> (x, y) position
    """
    if layout == "spring":
        return nx.spring_layout(G, k=1.5 / np.sqrt(G.number_of_nodes()),
                                iterations=100, seed=42)

    # Circular layout: nodes arranged in concentric rings by layer,
    # or in a single circle ordered by layer
    N = G.number_of_nodes()
    pos = {}

    for node in range(N):
        layer = node // num_heads_per_layer
        head = node % num_heads_per_layer

        # Place nodes in a circle, grouped by layer
        # Angle based on global position
        angle = 2 * np.pi * node / N
        radius = 1.0
        pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

    return pos
