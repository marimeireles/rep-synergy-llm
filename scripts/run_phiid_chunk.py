#!/usr/bin/env python
"""
Chunked PhiID computation for multi-node parallelism.

Computes a slice of all head pairs on a single node, to be merged later
by merge_phiid_chunks.py. Designed for SLURM job arrays.

Usage:
  python scripts/run_phiid_chunk.py --model qwen3-8b --chunk-id 0 --num-chunks 8 --max-workers 64
  python scripts/run_phiid_chunk.py --model qwen3-8b --chunk-id 3 --num-chunks 8

Environment variable alternative (for SLURM arrays):
  SLURM_ARRAY_TASK_ID is used as chunk-id if --chunk-id is not provided.
"""

import argparse
import logging
import multiprocessing
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from src.utils import set_seed, setup_logging, ensure_dirs, get_result_path
from src.phiid_computation import compute_all_pairs_phiid

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Chunked PhiID computation")
    parser.add_argument("--model", required=True,
                        choices=["pythia-1b", "gemma3-4b", "qwen3-8b"],
                        help="Model to analyze")
    parser.add_argument("--revision", default=None,
                        help="Model revision (e.g., step1000 for Pythia checkpoints)")
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Chunk index (0-based). Falls back to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--num-chunks", type=int, required=True,
                        help="Total number of chunks to split into")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: all available CPUs)")
    args = parser.parse_args()

    setup_logging()

    # Resolve chunk ID
    chunk_id = args.chunk_id
    if chunk_id is None:
        chunk_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if chunk_id is not None:
            chunk_id = int(chunk_id)
        else:
            logger.error("No --chunk-id provided and SLURM_ARRAY_TASK_ID not set.")
            sys.exit(1)

    num_chunks = args.num_chunks
    if chunk_id < 0 or chunk_id >= num_chunks:
        logger.error(f"chunk-id {chunk_id} out of range [0, {num_chunks})")
        sys.exit(1)

    config = get_config(args.model)
    set_seed(config["seed"])
    ensure_dirs(config["results_dir"])

    model_id = config["model_id"]
    results_dir = config["results_dir"]

    # Load activations
    act_path = get_result_path(results_dir, "activations", model_id,
                               "activations.npz", args.revision)
    if not os.path.exists(act_path):
        logger.error(f"Activations not found at {act_path}. Run phase 1 first.")
        sys.exit(1)

    activations = np.load(act_path)['activations']
    num_heads = activations.shape[1]
    logger.info(f"Loaded activations: shape {activations.shape}, {num_heads} heads")

    # Compute pair range for this chunk
    total_pairs = num_heads * (num_heads - 1) // 2
    chunk_size = (total_pairs + num_chunks - 1) // num_chunks  # ceiling division
    pair_start = chunk_id * chunk_size
    pair_end = min(pair_start + chunk_size, total_pairs)

    logger.info(f"Chunk {chunk_id}/{num_chunks}: pairs [{pair_start}:{pair_end}] "
                f"({pair_end - pair_start} of {total_pairs} total)")

    if pair_start >= total_pairs:
        logger.info("This chunk has no pairs to compute (total already covered). Exiting.")
        return

    # Determine workers
    if args.max_workers is None:
        max_workers = multiprocessing.cpu_count()
    else:
        max_workers = args.max_workers
    logger.info(f"Using {max_workers} workers on {multiprocessing.cpu_count()} available CPUs")

    # Output path for this chunk
    rev_suffix = f"_{args.revision}" if args.revision else ""
    chunk_path = os.path.join(
        results_dir, "phiid_scores",
        f"{model_id}{rev_suffix}_pairwise_phiid_chunk{chunk_id:04d}.npz"
    )

    if os.path.exists(chunk_path):
        logger.info(f"Chunk output already exists at {chunk_path}. Skipping.")
        return

    sts_matrix, rtr_matrix = compute_all_pairs_phiid(
        activations=activations,
        num_heads=num_heads,
        tau=config["phiid_tau"],
        kind=config["phiid_kind"],
        redundancy=config["phiid_redundancy"],
        max_workers=max_workers,
        save_path=chunk_path,
        checkpoint_interval=5000,
        pair_range=(pair_start, pair_end),
    )

    logger.info(f"Chunk {chunk_id} complete. Saved to {chunk_path}")
    logger.info(f"  sts non-zero: {np.count_nonzero(sts_matrix)}, "
                f"rtr non-zero: {np.count_nonzero(rtr_matrix)}")


if __name__ == "__main__":
    main()
