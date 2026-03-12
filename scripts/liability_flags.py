#!/usr/bin/env python
from __future__ import annotations

import pandas as pd


def add_property_liability_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["flag_high_mw"] = out.get("MW", pd.Series(index=out.index, dtype=float)) > 500
    out["flag_high_tpsa"] = out.get("TPSA", pd.Series(index=out.index, dtype=float)) > 120
    out["flag_high_hbd"] = out.get("HBD", pd.Series(index=out.index, dtype=float)) > 3
    out["flag_high_logp"] = out.get("cLogP", pd.Series(index=out.index, dtype=float)) > 5
    out["flag_any_liability"] = out[["flag_high_mw", "flag_high_tpsa", "flag_high_hbd", "flag_high_logp"]].fillna(False).any(axis=1)
    return out
