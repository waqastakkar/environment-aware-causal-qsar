from __future__ import annotations

import numpy as np
import pandas as pd


def _des(x: pd.Series, low: float, high: float, mode: str = "inside") -> pd.Series:
    if mode == "inside":
        return ((x >= low) & (x <= high)).astype(float)
    if mode == "low":
        return (x <= high).astype(float)
    return (x >= low).astype(float)


def add_bbb_metrics(df: pd.DataFrame, cns_mpo_threshold: float = 4.0) -> pd.DataFrame:
    out = df.copy()
    req = ["MW", "LogP", "TPSA", "HBD", "RotB", "HBA"]
    if not all(c in out.columns for c in req):
        out["cns_mpo"] = np.nan
        out["cns_like"] = 0
        return out
    comps = pd.DataFrame(index=out.index)
    comps["d_MW"] = _des(out["MW"], 200, 450)
    comps["d_LogP"] = _des(out["LogP"], 1, 4)
    comps["d_TPSA"] = _des(out["TPSA"], 20, 90)
    comps["d_HBD"] = _des(out["HBD"], -np.inf, 3, "low")
    comps["d_RotB"] = _des(out["RotB"], -np.inf, 8, "low")
    comps["d_HBA"] = _des(out["HBA"], -np.inf, 10, "low")
    out["cns_mpo"] = comps.sum(axis=1)
    out["cns_like"] = (out["cns_mpo"] >= cns_mpo_threshold).astype(int)
    return out
