#!/usr/bin/env python3
"""
stats.py - Paired bootstrap confidence intervals and permutation tests.

Library usage
    from scripts.stats import paired_bootstrap_diff, permutation_test_paired, bootstrap_mean_ci
    res = paired_bootstrap_diff(acc_a, acc_b)   # per-question 0/1 arrays aligned by question
    res["diff"], res["ci_lo"], res["ci_hi"], res["p_perm"]

All resampling is over QUESTIONS (paired), which is the unit of analysis in the
paper: the same MINT questions are evaluated under every configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_B = 10_000
DEFAULT_SEED = 12345


def _as_float_array(x: Iterable) -> np.ndarray:
    return np.asarray([float(v) for v in x], dtype=float)


def bootstrap_mean_ci(values: Iterable, n_boot: int = DEFAULT_B, alpha: float = 0.05, seed: int = DEFAULT_SEED) -> Dict:
    """Percentile bootstrap CI for a mean (e.g., accuracy) over questions."""
    v = _as_float_array(values)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci_lo": float(np.quantile(means, alpha / 2)),
        "ci_hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(n),
    }


def paired_bootstrap_diff(
    a: Iterable,
    b: Iterable,
    n_boot: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
    n_perm: Optional[int] = None,
) -> Dict:
    """Paired bootstrap CI for mean(a) - mean(b) and a paired permutation p-value.

    a, b: per-question outcomes aligned on the same questions (0/1 or real).
    The permutation test randomly swaps a_i and b_i (sign flips of the paired
    differences), which is exact under the null of exchangeability.
    """
    a = _as_float_array(a)
    b = _as_float_array(b)
    if len(a) != len(b):
        raise ValueError(f"paired arrays must have equal length ({len(a)} vs {len(b)})")
    n = len(a)
    if n == 0:
        return {"diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_perm": float("nan"), "n": 0,
                "mean_a": float("nan"), "mean_b": float("nan")}
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))

    n_perm = n_perm or n_boot
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * d).mean(axis=1)
    obs = float(d.mean())
    p_perm = float((np.sum(np.abs(perm_means) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": obs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_perm": p_perm,
        "n": int(n),
        "significant_05": bool(ci_lo > 0 or ci_hi < 0),
    }


def cohens_kappa_binary(agree_obs: float, p_a: Sequence[float], p_b: Sequence[float]) -> float:
    """Chance-corrected agreement given observed agreement and marginal distributions."""
    p_e = float(sum(pa * pb for pa, pb in zip(p_a, p_b)))
    if p_e >= 1.0:
        return float("nan")
    return (agree_obs - p_e) / (1.0 - p_e)


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj
