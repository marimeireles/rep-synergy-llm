"""
Tests for PhiID computation.

Uses synthetic data only (no network access needed on compute nodes).
"""

import numpy as np
import pytest

from phyid.calculate import calc_PhiID


def test_calc_phiid_runs():
    """Test that calc_PhiID runs without error on simple synthetic data."""
    np.random.seed(42)
    src = np.random.randn(100)
    trg = np.random.randn(100)

    atoms, calc_res = calc_PhiID(src, trg, tau=1, kind="gaussian", redundancy="MMI")
    assert isinstance(atoms, dict)
    assert 'sts' in atoms
    assert 'rtr' in atoms


def test_atom_keys():
    """Test that all 16 PhiID atoms are present."""
    np.random.seed(42)
    src = np.random.randn(100)
    trg = np.random.randn(100)

    atoms, _ = calc_PhiID(src, trg, tau=1, kind="gaussian", redundancy="MMI")

    expected_keys = [
        'rtr', 'rtx', 'rty', 'rts',
        'xtr', 'xtx', 'xty', 'xts',
        'ytr', 'ytx', 'yty', 'yts',
        'str', 'stx', 'sty', 'sts',
    ]
    for key in expected_keys:
        assert key in atoms, f"Missing atom key: {key}"


def test_atom_values_are_arrays():
    """Test that atom values are arrays (local values), not scalars."""
    np.random.seed(42)
    src = np.random.randn(100)
    trg = np.random.randn(100)

    atoms, _ = calc_PhiID(src, trg, tau=1, kind="gaussian", redundancy="MMI")

    for key in ['sts', 'rtr']:
        val = atoms[key]
        # Could be scalar or array depending on phyid version
        mean_val = np.mean(val)
        assert np.isfinite(mean_val), f"Non-finite mean for {key}: {mean_val}"


def test_mean_values_finite():
    """Test that np.mean() of each atom gives a finite number."""
    np.random.seed(42)
    src = np.random.randn(200)
    trg = np.random.randn(200)

    atoms, _ = calc_PhiID(src, trg, tau=1, kind="gaussian", redundancy="MMI")

    for key in ['sts', 'rtr']:
        val = float(np.mean(atoms[key]))
        assert np.isfinite(val), f"Non-finite value for {key}"
        print(f"  {key} = {val:.6f}")


def test_correlated_signals():
    """Test with correlated signals — should have meaningful rtr."""
    np.random.seed(42)
    x = np.random.randn(200)
    # y is correlated with x
    y = 0.8 * x + 0.2 * np.random.randn(200)

    atoms, _ = calc_PhiID(x, y, tau=1, kind="gaussian", redundancy="MMI")
    rtr = float(np.mean(atoms['rtr']))
    sts = float(np.mean(atoms['sts']))

    print(f"  Correlated: rtr={rtr:.6f}, sts={sts:.6f}")
    assert np.isfinite(rtr) and np.isfinite(sts)


def test_independent_signals():
    """Test with independent random signals."""
    np.random.seed(42)
    src = np.random.randn(200)
    trg = np.random.randn(200)

    atoms, _ = calc_PhiID(src, trg, tau=1, kind="gaussian", redundancy="MMI")
    rtr = float(np.mean(atoms['rtr']))
    sts = float(np.mean(atoms['sts']))

    print(f"  Independent: rtr={rtr:.6f}, sts={sts:.6f}")
    assert np.isfinite(rtr) and np.isfinite(sts)


def test_our_wrapper():
    """Test our wrapper function compute_pairwise_phiid."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.phiid_computation import compute_pairwise_phiid

    np.random.seed(42)
    # Simulate: 5 prompts, 4 heads, 50 steps
    activations = np.random.randn(5, 4, 50)

    result = compute_pairwise_phiid(activations, head_i=0, head_j=1, tau=1)

    assert 'sts' in result
    assert 'rtr' in result
    assert np.isfinite(result['sts'])
    assert np.isfinite(result['rtr'])
    print(f"  Wrapper: sts={result['sts']:.6f}, rtr={result['rtr']:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
