#!/usr/bin/env python
"""
Merge chunked PhiID results into a single pairwise matrix.

After all SLURM array tasks from run_phiid_chunk.py complete, run this to
combine the partial matrices into the final pairwise_phiid.npz.

Usage:
  python scripts/merge_phiid_chunks.py --model qwen3-8b --num-chunks 8
  python scripts/merge_phiid_chunks.py --model qwen3-8b --num-chunks 8 --cleanup
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from src.utils import setup_logging, ensure_dirs, get_result_path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge chunked PhiID results")
    parser.add_argument("--model", required=True,
                        choices=["pythia-1b", "gemma3-4b", "qwen3-8b"],
                        help="Model to analyze")
    parser.add_argument("--revision", default=None,
                        help="Model revision")
    parser.add_argument("--num-chunks", type=int, required=True,
                        help="Total number of chunks")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete chunk files after successful merge")
    args = parser.parse_args()

    setup_logging()

    config = get_config(args.model)
    model_id = config["model_id"]
    results_dir = config["results_dir"]
    ensure_dirs(results_dir)

    rev_suffix = f"_{args.revision}" if args.revision else ""

    # Check that all chunks exist
    chunk_paths = []
    missing = []
    for i in range(args.num_chunks):
        path = os.path.join(
            results_dir, "phiid_scores",
            f"{model_id}{rev_suffix}_pairwise_phiid_chunk{i:04d}.npz"
        )
        chunk_paths.append(path)
        if not os.path.exists(path):
            missing.append(i)

    if missing:
        logger.error(f"Missing {len(missing)} chunk(s): {missing}")
        logger.error("Run the missing chunks before merging.")
        sys.exit(1)

    logger.info(f"All {args.num_chunks} chunks found. Merging...")

    # Load first chunk to get matrix dimensions
    first = np.load(chunk_paths[0])
    num_heads = first['sts_matrix'].shape[0]
    sts_merged = np.zeros((num_heads, num_heads))
    rtr_merged = np.zeros((num_heads, num_heads))

    # Sum all chunks (each chunk only has values for its slice of pairs)
    for i, path in enumerate(chunk_paths):
        data = np.load(path)
        sts_merged += data['sts_matrix']
        rtr_merged += data['rtr_matrix']
        nonzero_sts = np.count_nonzero(data['sts_matrix'])
        logger.info(f"  Chunk {i}: {nonzero_sts // 2} pairs (sts non-zero entries: {nonzero_sts})")

    # Verify: total non-zero entries should be 2 * C(num_heads, 2) for a symmetric matrix
    expected_entries = num_heads * (num_heads - 1)  # 2 * C(n,2)
    actual_sts = np.count_nonzero(sts_merged)
    logger.info(f"Merged matrix: {actual_sts} non-zero sts entries "
                f"(expected {expected_entries} for {num_heads} heads)")

    if actual_sts < expected_entries:
        logger.warning(f"Some pairs may have zero values (expected {expected_entries}, "
                       f"got {actual_sts}). This could be from failed computations or "
                       f"genuinely zero PhiID values.")

    # Save final merged result
    final_path = get_result_path(results_dir, "phiid_scores", model_id,
                                 "pairwise_phiid.npz", args.revision)
    np.savez(final_path, sts_matrix=sts_merged, rtr_matrix=rtr_merged)
    logger.info(f"Saved merged PhiID to {final_path}")
    logger.info(f"  sts range: [{sts_merged.min():.6f}, {sts_merged.max():.6f}]")
    logger.info(f"  rtr range: [{rtr_merged.min():.6f}, {rtr_merged.max():.6f}]")

    # Cleanup chunk files
    if args.cleanup:
        for path in chunk_paths:
            os.remove(path)
            # Also remove any checkpoint files
            ckpt = path.replace('.npz', '_checkpoint.npz')
            if os.path.exists(ckpt):
                os.remove(ckpt)
        logger.info(f"Cleaned up {len(chunk_paths)} chunk files.")
    else:
        logger.info("Chunk files preserved. Use --cleanup to delete them.")


if __name__ == "__main__":
    main()
