#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import pandas as pd


def split_conformal_q(val_df: pd.DataFrame, coverage: float = 0.90) -> tuple[float, pd.DataFrame]:
    work = val_df.dropna(subset=["y", "yhat_mean"]).copy()
    scores = (work["y"] - work["yhat_mean"]).abs().to_numpy()
    n = len(scores)
    if n == 0:
        return np.nan, pd.DataFrame(columns=["n_val", "coverage_target", "q"])
    q_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    q = float(np.quantile(scores, q_level, method="higher" if hasattr(np, "quantile") else "linear"))
    cal = pd.DataFrame([{"n_val": n, "coverage_target": coverage, "q": q, "q_level": q_level}])
    return q, cal


def apply_conformal(test_df: pd.DataFrame, q: float) -> pd.DataFrame:
    out = test_df.copy()
    out["interval_lower"] = out["yhat_mean"] - q
    out["interval_upper"] = out["yhat_mean"] + q
    out["interval_width"] = out["interval_upper"] - out["interval_lower"]
    if "y" in out.columns:
        out["covered"] = (out["y"] >= out["interval_lower"]) & (out["y"] <= out["interval_upper"])
    return out


def summarize_coverage(interval_df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["split_name", "ablation"]
    avail = [c for c in group_cols if c in interval_df.columns]
    if not avail:
        avail = ["_all"]
        work = interval_df.copy()
        work["_all"] = "all"
    else:
        work = interval_df
    rows = []
    for key, g in work.groupby(avail, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: v for c, v in zip(avail, key)}
        row.update(
            {
                "n": int(len(g)),
                "empirical_coverage": float(g["covered"].mean()) if "covered" in g.columns else np.nan,
                "mean_interval_width": float(g["interval_width"].mean()) if "interval_width" in g.columns else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
