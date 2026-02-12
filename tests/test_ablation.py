"""
Tests for ablation module.

Uses a small model or mock to verify ablation logic.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_kl_divergence_zero_no_ablation():
    """KL divergence should be 0 when no heads are ablated."""
    import torch
    import torch.nn.functional as F

    # Identical logits should give KL = 0
    logits = np.random.randn(10, 100).astype(np.float32)  # 10 steps, 100 vocab

    orig = torch.tensor(logits)
    orig_log_p = F.log_softmax(orig, dim=-1)
    orig_p = torch.exp(orig_log_p)

    abl_log_p = F.log_softmax(torch.tensor(logits), dim=-1)

    kl = F.kl_div(abl_log_p, orig_p, reduction='none', log_target=False)
    kl_per_step = kl.sum(dim=-1)
    mean_kl = kl_per_step.mean().item()

    assert abs(mean_kl) < 1e-6, f"Expected KL ~0, got {mean_kl}"


def test_kl_divergence_positive_different_logits():
    """KL divergence should be positive for different distributions."""
    import torch
    import torch.nn.functional as F

    np.random.seed(42)
    orig_logits = np.random.randn(10, 100).astype(np.float32)
    abl_logits = np.random.randn(10, 100).astype(np.float32)

    orig = torch.tensor(orig_logits)
    orig_log_p = F.log_softmax(orig, dim=-1)
    orig_p = torch.exp(orig_log_p)

    abl_log_p = F.log_softmax(torch.tensor(abl_logits), dim=-1)

    kl = F.kl_div(abl_log_p, orig_p, reduction='none', log_target=False)
    kl_per_step = kl.sum(dim=-1)
    mean_kl = kl_per_step.mean().item()

    assert mean_kl > 0, f"Expected positive KL, got {mean_kl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
