#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition

from stats_utils import bootstrap_ci


@dataclass
class RGroupOutputs:
    series_scaffolds: pd.DataFrame
    rgroup_table: pd.DataFrame
    effects: pd.DataFrame
    env_interactions: pd.DataFrame
    quality: pd.DataFrame


def _pick_series_core(smiles_list: list[str]) -> str | None:
    scaffolds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            continue
        scaffolds.append(Chem.MolToSmiles(scaf))
    if not scaffolds:
        return None
    return Counter(scaffolds).most_common(1)[0][0]


def run_rgroup_analysis(df: pd.DataFrame, series_min_n: int = 8, env_col: str | None = None) -> RGroupOutputs:
    env_col = env_col or ("env_id_manual" if "env_id_manual" in df.columns else ("assay_type" if "assay_type" in df.columns else None))
    rows = []
    scaffold_rows = []
    quality_rows = []

    grouped = df.groupby("series_id", dropna=True)
    for series_id, sdf in grouped:
        if len(sdf) < series_min_n:
            quality_rows.append({"series_id": series_id, "status": "skipped_small", "n": len(sdf)})
            continue
        core_smiles = _pick_series_core(sdf["smiles"].dropna().tolist())
        if not core_smiles:
            quality_rows.append({"series_id": series_id, "status": "failed_core", "n": len(sdf)})
            continue
        core = Chem.MolFromSmiles(core_smiles)
        rgd = rdRGroupDecomposition.RGroupDecomposition([core])
        ok_idx = []
        fails = 0
        for i, (_, r) in enumerate(sdf.iterrows()):
            mol = Chem.MolFromSmiles(str(r["smiles"]))
            if mol is None:
                fails += 1
                continue
            res = rgd.Add(mol)
            if res >= 0:
                ok_idx.append(i)
            else:
                fails += 1
        if not ok_idx:
            quality_rows.append({"series_id": series_id, "status": "failed_decompose", "n": len(sdf), "failed": fails})
            continue
        rgd.Process()
        rg_rows = rgd.GetRGroupsAsRows(asSmiles=True)
        sdf_ok = sdf.iloc[ok_idx].reset_index(drop=True)
        for i, rgr in enumerate(rg_rows):
            rec = sdf_ok.iloc[i].to_dict()
            rec["series_id"] = series_id
            rec["core_smiles"] = core_smiles
            for k, v in rgr.items():
                if k == "Core":
                    continue
                rec[str(k)] = v
            rows.append(rec)
        scaffold_rows.append({"series_id": series_id, "core_smiles": core_smiles, "n": len(sdf_ok)})
        quality_rows.append({"series_id": series_id, "status": "processed", "n": len(sdf_ok), "failed": fails})

    rtable = pd.DataFrame(rows)
    scaffolds = pd.DataFrame(scaffold_rows)
    quality = pd.DataFrame(quality_rows)
    if rtable.empty:
        return RGroupOutputs(scaffolds, rtable, pd.DataFrame(), pd.DataFrame(), quality)

    rcols = [c for c in rtable.columns if c.startswith("R")]
    effect_rows = []
    for (series_id, rcol, rsmi), g in (
        rtable.melt(id_vars=[c for c in rtable.columns if c not in rcols], value_vars=rcols, var_name="R_label", value_name="R_smiles")
        .dropna(subset=["R_smiles"]).groupby(["series_id", "R_label", "R_smiles"])
    ):
        p = bootstrap_ci(g["pIC50"].values) if "pIC50" in g.columns else bootstrap_ci([])
        y = bootstrap_ci(g["yhat"].values) if "yhat" in g.columns else bootstrap_ci([])
        effect_rows.append({
            "series_id": series_id,
            "R_label": rcol,
            "R_smiles": rsmi,
            "support_n": len(g),
            "pIC50_mean": p.mean,
            "pIC50_median": p.median,
            "pIC50_ci_low": p.ci_low,
            "pIC50_ci_high": p.ci_high,
            "yhat_mean": y.mean,
            "yhat_median": y.median,
            "yhat_ci_low": y.ci_low,
            "yhat_ci_high": y.ci_high,
        })
    effects = pd.DataFrame(effect_rows)

    env_rows = []
    if env_col and env_col in rtable.columns:
        long_df = rtable.melt(id_vars=[c for c in rtable.columns if c not in rcols], value_vars=rcols, var_name="R_label", value_name="R_smiles").dropna(subset=["R_smiles", env_col])
        for keys, g in long_df.groupby(["series_id", "R_label", "R_smiles", env_col]):
            series_id, rlabel, rsmi, env = keys
            env_rows.append({
                "series_id": series_id,
                "R_label": rlabel,
                "R_smiles": rsmi,
                "env": env,
                "n": len(g),
                "pIC50_mean": g["pIC50"].mean() if "pIC50" in g.columns else np.nan,
                "yhat_mean": g["yhat"].mean() if "yhat" in g.columns else np.nan,
            })
    env_df = pd.DataFrame(env_rows)
    return RGroupOutputs(scaffolds, rtable, effects, env_df, quality)
