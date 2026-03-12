#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition


def decompose_with_core(df: pd.DataFrame, core_smiles: str, smiles_col: str = "smiles") -> pd.DataFrame:
    core = Chem.MolFromSmiles(core_smiles) if isinstance(core_smiles, str) else None
    if core is None:
        return pd.DataFrame()
    rgd = rdRGroupDecomposition.RGroupDecomposition([core])
    kept = []
    for i, (_, r) in enumerate(df.iterrows()):
        mol = Chem.MolFromSmiles(str(r[smiles_col]))
        if mol is None:
            continue
        if rgd.Add(mol) >= 0:
            kept.append(i)
    if not kept:
        return pd.DataFrame()
    rgd.Process()
    rows = rgd.GetRGroupsAsRows(asSmiles=True)
    out = []
    src = df.iloc[kept].reset_index(drop=True)
    for i, rg in enumerate(rows):
        rec = src.iloc[i].to_dict()
        for k, v in rg.items():
            if str(k) != "Core":
                rec[str(k)] = v
        out.append(rec)
    return pd.DataFrame(out)


def transfer_rgroups(mapped_hits: pd.DataFrame, scaffold_col: str = "matched_scaffold", smiles_col: str = "smiles") -> pd.DataFrame:
    if mapped_hits.empty:
        return pd.DataFrame()
    rows = []
    for scaf, g in mapped_hits.dropna(subset=[scaffold_col]).groupby(scaffold_col):
        rtab = decompose_with_core(g, scaf, smiles_col=smiles_col)
        if rtab.empty:
            continue
        rtab["core_scaffold"] = scaf
        rows.append(rtab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
