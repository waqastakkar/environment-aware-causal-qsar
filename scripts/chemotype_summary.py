#!/usr/bin/env python
from __future__ import annotations

import pandas as pd


def summarize_chemotypes(df: pd.DataFrame, id_col: str = "compound_id") -> pd.DataFrame:
    if "scaffold" not in df.columns:
        raise ValueError("Expected scaffold column for chemotype summaries.")
    return (
        df.groupby("scaffold", dropna=False)
        .agg(
            count=(id_col, "count"),
            best_score=("score_mean", "max"),
            median_score=("score_mean", "median"),
            best_cns_mpo=("cns_mpo", "max") if "cns_mpo" in df.columns else ("score_mean", "max"),
        )
        .reset_index()
        .sort_values(["count", "best_score"], ascending=[False, False])
    )


def pick_chemotype_leads(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["scaffold", "score_mean", "uncertainty_std"], ascending=[True, False, True])
        .groupby("scaffold", as_index=False)
        .head(1)
        .sort_values("score_mean", ascending=False)
    )
