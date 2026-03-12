#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def detect_id_col(df: pd.DataFrame) -> str:
    for c in ["molecule_id", "InChIKey", "inchikey"]:
        if c in df.columns:
            return c
    raise ValueError("No identifier column found (molecule_id/InChIKey/inchikey)")


def detect_pred_col(df: pd.DataFrame) -> str:
    for c in ["y_pred", "yhat", "prediction", "pred", "pIC50_hat"]:
        if c in df.columns:
            return c
    raise ValueError("No prediction column found")


def detect_y_col(df: pd.DataFrame) -> str | None:
    for c in ["y_true", "y", "pIC50", "label"]:
        if c in df.columns:
            return c
    return None


def _load_pred(path: str | Path, run_member: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    id_col = detect_id_col(df)
    pred_col = detect_pred_col(df)
    y_col = detect_y_col(df)
    out = pd.DataFrame({id_col: df[id_col], f"yhat_{run_member}": pd.to_numeric(df[pred_col], errors="coerce")})
    if y_col is not None:
        out["y"] = pd.to_numeric(df[y_col], errors="coerce")
    passthrough = [c for c in ["split", "split_name", "env_id_manual", "env_id", "cns_mpo", "is_cns_like"] if c in df.columns]
    for c in passthrough:
        out[c] = df[c]
    return out


def ensemble_for_group(runs_df: pd.DataFrame, split_kind: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    members: list[str] = []
    for i, row in runs_df.reset_index(drop=True).iterrows():
        p = row[f"{split_kind}_pred_path"]
        if not isinstance(p, str) or not Path(p).exists():
            continue
        member = str(row.get("run_id", f"run{i}"))
        members.append(member)
        cur = _load_pred(p, member)
        id_col = detect_id_col(cur)
        if merged is None:
            merged = cur
        else:
            keep_cols = [id_col, f"yhat_{member}"]
            if "y" in cur.columns and "y" not in merged.columns:
                keep_cols.append("y")
            merged = merged.merge(cur[keep_cols], on=id_col, how="inner")
    if merged is None or not members:
        return pd.DataFrame()
    pred_cols = [c for c in merged.columns if c.startswith("yhat_")]
    merged["yhat_mean"] = merged[pred_cols].mean(axis=1)
    merged["yhat_std"] = merged[pred_cols].std(axis=1, ddof=0)
    if "y" in merged.columns:
        merged["abs_error"] = (merged["yhat_mean"] - merged["y"]).abs()
        merged["residual"] = merged["yhat_mean"] - merged["y"]
    return merged


def selective_prediction_curve(df: pd.DataFrame, coverages: Iterable[float] | None = None) -> pd.DataFrame:
    if coverages is None:
        coverages = np.round(np.arange(1.0, 0.49, -0.05), 2)
    work = df.dropna(subset=["y", "yhat_mean", "yhat_std"]).sort_values("yhat_std", ascending=True).copy()
    rows = []
    n = len(work)
    for cov in coverages:
        k = max(1, int(np.floor(n * float(cov))))
        s = work.iloc[:k]
        err = s["yhat_mean"] - s["y"]
        rows.append(
            {
                "coverage": float(cov),
                "n": int(k),
                "rmse": float(np.sqrt(np.mean(np.square(err)))) if k else np.nan,
                "mae": float(np.mean(np.abs(err))) if k else np.nan,
            }
        )
    return pd.DataFrame(rows)
