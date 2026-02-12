"""
PhiID (Integrated Information Decomposition) computation wrapper.

Uses the phyid library to compute PhiID atoms between pairs of attention heads.
Key atoms: sts (Syn->Syn, synergy) and rtr (Red->Red, redundancy).
"""

import logging
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _compute_single_pair(args):
    """
    Worker function for parallel PhiID computation.
    Must be at module level for pickling.
    """
    from phyid.calculate import calc_PhiID

    activations_i, activations_j, tau, kind, redundancy = args

    num_prompts = activations_i.shape[0]
    sts_values = []
    rtr_values = []

    for p in range(num_prompts):
        src = activations_i[p]  # shape (num_steps,)
        trg = activations_j[p]  # shape (num_steps,)

        try:
            atoms, _ = calc_PhiID(src, trg, tau=tau, kind=kind, redundancy=redundancy)
            # atoms['sts'] and atoms['rtr'] are LOCAL arrays of shape (N-tau,)
            sts_val = float(np.mean(atoms['sts']))
            rtr_val = float(np.mean(atoms['rtr']))
        except Exception as e:
            # If PhiID fails for a prompt (e.g., constant signal), use 0
            sts_val = 0.0
            rtr_val = 0.0

        sts_values.append(sts_val)
        rtr_values.append(rtr_val)

    # Average across prompts
    return float(np.mean(sts_values)), float(np.mean(rtr_values))


def compute_pairwise_phiid(activations, head_i, head_j, tau=1,
                           kind="gaussian", redundancy="MMI"):
    """
    Compute PhiID between two attention heads' activation time series.

    Args:
        activations: np.ndarray of shape (num_prompts, num_heads, num_steps)
        head_i, head_j: indices of the two heads
        tau: time lag (default 1)
        kind: PhiID method (default "gaussian")
        redundancy: redundancy measure (default "MMI")

    Returns:
        dict with keys 'sts' (float) and 'rtr' (float)
    """
    act_i = activations[:, head_i, :]  # (num_prompts, num_steps)
    act_j = activations[:, head_j, :]  # (num_prompts, num_steps)

    sts_mean, rtr_mean = _compute_single_pair(
        (act_i, act_j, tau, kind, redundancy)
    )

    return {'sts': sts_mean, 'rtr': rtr_mean}


def compute_all_pairs_phiid(activations, num_heads, tau=1,
                            kind="gaussian", redundancy="MMI",
                            max_workers=None, save_path=None,
                            checkpoint_interval=5000):
    """
    Compute PhiID for ALL pairs of attention heads.

    Pythia-1B: 256 heads -> C(256,2) = 32,640 pairs.
    Each pair averaged over all prompts.

    Args:
        activations: np.ndarray of shape (num_prompts, num_heads, num_steps)
        num_heads: total number of attention heads
        tau: time lag
        kind: PhiID method
        redundancy: redundancy measure
        max_workers: number of parallel workers (None = auto)
        save_path: if provided, save intermediate results
        checkpoint_interval: save checkpoint every N pairs

    Returns:
        sts_matrix: np.ndarray of shape (num_heads, num_heads) — synergy values
        rtr_matrix: np.ndarray of shape (num_heads, num_heads) — redundancy values
    """
    pairs = list(combinations(range(num_heads), 2))
    total_pairs = len(pairs)
    logger.info(f"Computing PhiID for {total_pairs} head pairs with {max_workers} workers")

    sts_matrix = np.zeros((num_heads, num_heads))
    rtr_matrix = np.zeros((num_heads, num_heads))

    # Check for existing checkpoint
    completed = 0
    if save_path is not None:
        import os
        ckpt_path = save_path.replace('.npz', '_checkpoint.npz')
        if os.path.exists(ckpt_path):
            ckpt = np.load(ckpt_path)
            sts_matrix = ckpt['sts_matrix']
            rtr_matrix = ckpt['rtr_matrix']
            completed = int(ckpt['completed'])
            logger.info(f"Resuming from checkpoint: {completed}/{total_pairs} pairs done")

    # Prepare work items (skip already completed)
    remaining_pairs = pairs[completed:]

    # Build argument list
    work_items = []
    for (i, j) in remaining_pairs:
        act_i = activations[:, i, :]
        act_j = activations[:, j, :]
        work_items.append((act_i, act_j, tau, kind, redundancy))

    # Process with parallelism
    if max_workers == 1:
        # Sequential for debugging
        for idx, (i, j) in enumerate(tqdm(remaining_pairs, desc="PhiID pairs",
                                           initial=completed, total=total_pairs)):
            sts_val, rtr_val = _compute_single_pair(work_items[idx])
            sts_matrix[i, j] = sts_val
            sts_matrix[j, i] = sts_val
            rtr_matrix[i, j] = rtr_val
            rtr_matrix[j, i] = rtr_val

            if save_path and (completed + idx + 1) % checkpoint_interval == 0:
                _save_checkpoint(save_path, sts_matrix, rtr_matrix, completed + idx + 1)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {}
            for idx, item in enumerate(work_items):
                future = executor.submit(_compute_single_pair, item)
                future_to_pair[future] = (remaining_pairs[idx], completed + idx)

            # Collect results with progress bar
            pbar = tqdm(total=total_pairs, initial=completed, desc="PhiID pairs")
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

                if save_path and done_count % checkpoint_interval == 0:
                    _save_checkpoint(save_path, sts_matrix, rtr_matrix, done_count)

            pbar.close()

    # Final save
    if save_path:
        np.savez(save_path, sts_matrix=sts_matrix, rtr_matrix=rtr_matrix)
        logger.info(f"Saved pairwise PhiID to {save_path}")
        # Clean up checkpoint
        import os
        ckpt_path = save_path.replace('.npz', '_checkpoint.npz')
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    return sts_matrix, rtr_matrix


def _save_checkpoint(save_path, sts_matrix, rtr_matrix, completed):
    """Save intermediate checkpoint."""
    ckpt_path = save_path.replace('.npz', '_checkpoint.npz')
    np.savez(ckpt_path, sts_matrix=sts_matrix, rtr_matrix=rtr_matrix,
             completed=np.array(completed))
    logger.info(f"Checkpoint saved: {completed} pairs completed")
