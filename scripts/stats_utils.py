#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    median: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> BootstrapResult:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return BootstrapResult(np.nan, np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots[i] = float(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    return BootstrapResult(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        ci_low=float(np.quantile(boots, alpha)),
        ci_high=float(np.quantile(boots, 1.0 - alpha)),
        n=int(arr.size),
    )


def fisher_exact(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    try:
        from scipy.stats import fisher_exact as scipy_fisher_exact

        odds_ratio, pvalue = scipy_fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return float(odds_ratio), float(pvalue)
    except Exception:
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        return float(odds_ratio), 1.0


def multiple_testing_correction(pvalues: Iterable[float], method: str = "bh") -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    if method.lower() in {"bh", "fdr_bh", "benjamini-hochberg"}:
        q = ranked * n / (np.arange(n) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
    elif method.lower() == "holm":
        q = (n - np.arange(n)) * ranked
        q = np.maximum.accumulate(q)
    else:
        raise ValueError(f"Unknown correction method: {method}")
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def ks_wasserstein(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, float]:
    xa = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    ya = np.asarray([v for v in y if v is not None and not np.isnan(v)], dtype=float)
    if xa.size == 0 or ya.size == 0:
        return np.nan, np.nan, np.nan
    try:
        from scipy.stats import ks_2samp, wasserstein_distance

        ks_stat, ks_p = ks_2samp(xa, ya)
        w = wasserstein_distance(xa, ya)
        return float(ks_stat), float(ks_p), float(w)
    except Exception:
        # fallback approximations
        x_sorted = np.sort(xa)
        y_sorted = np.sort(ya)
        grid = np.sort(np.unique(np.concatenate([x_sorted, y_sorted])))
        cdf_x = np.searchsorted(x_sorted, grid, side="right") / x_sorted.size
        cdf_y = np.searchsorted(y_sorted, grid, side="right") / y_sorted.size
        ks_stat = float(np.max(np.abs(cdf_x - cdf_y)))
        # crude earth mover estimate on quantiles
        q = np.linspace(0, 1, 200)
        w = float(np.mean(np.abs(np.quantile(x_sorted, q) - np.quantile(y_sorted, q))))
        return ks_stat, 1.0, w


def enrichment_2x2(pos_has: int, pos_not: int, neg_has: int, neg_not: int) -> dict:
    odds_ratio, p_value = fisher_exact(pos_has, pos_not, neg_has, neg_not)
    return {
        "pos_has": int(pos_has),
        "pos_not": int(pos_not),
        "neg_has": int(neg_has),
        "neg_not": int(neg_not),
        "odds_ratio": float(odds_ratio),
        "log_odds_ratio": float(np.log(max(odds_ratio, 1e-12))),
        "p_value": float(p_value),
    }
