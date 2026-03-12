#!/usr/bin/env python
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def regression_metrics(y: pd.Series, yhat: pd.Series) -> dict[str, float]:
    err = yhat - y
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(np.square(err)))
    ss_tot = float(np.sum(np.square(y - y.mean())))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}


def seed_metrics(test_ensemble_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in run_df.iterrows():
        p = r.get("test_pred_path")
        if not isinstance(p, str):
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "y_true" not in df.columns:
            continue
        pred_col = "y_pred" if "y_pred" in df.columns else ("yhat" if "yhat" in df.columns else None)
        if pred_col is None:
            continue
        m = regression_metrics(df["y_true"], df[pred_col])
        rows.append({"split_name": r.get("split_name"), "ablation": r.get("ablation"), "seed": r.get("seed"), "run_id": r.get("run_id"), **m})
    return pd.DataFrame(rows)


def ablation_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame(columns=["split_name", "ablation", "n_seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"])
    agg = seed_df.groupby(["split_name", "ablation"], dropna=False).agg(
        n_seeds=("seed", "count"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    )
    return agg.reset_index()


def paired_bootstrap_tests(seed_df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame(columns=["split_name", "ablation_a", "ablation_b", "metric", "delta_mean", "p_value"])
    rng = np.random.default_rng(42)
    rows = []
    for split, gsplit in seed_df.groupby("split_name", dropna=False):
        abls = sorted([a for a in gsplit["ablation"].dropna().unique()])
        for a, b in itertools.combinations(abls, 2):
            ga = gsplit[gsplit["ablation"] == a].set_index("seed")
            gb = gsplit[gsplit["ablation"] == b].set_index("seed")
            common = ga.index.intersection(gb.index)
            if len(common) < 2:
                continue
            for metric in ["rmse", "mae", "r2"]:
                diffs = (ga.loc[common, metric] - gb.loc[common, metric]).to_numpy(dtype=float)
                boot = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(diffs), len(diffs))
                    boot.append(float(np.mean(diffs[idx])))
                boot = np.asarray(boot)
                p = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
                rows.append({"split_name": split, "ablation_a": a, "ablation_b": b, "metric": metric, "delta_mean": float(np.mean(diffs)), "p_value": p})
    return pd.DataFrame(rows)
