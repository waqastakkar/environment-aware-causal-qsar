#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from scaffold_map import murcko_scaffold_smiles


def cluster_by_scaffold(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    out = df.copy()
    out["chemotype_core"] = out[smiles_col].apply(murcko_scaffold_smiles)
    out["chemotype_cluster"] = out["chemotype_core"].fillna("unassigned")
    return out


def cluster_by_butina(df: pd.DataFrame, threshold: float = 0.6, smiles_col: str = "smiles") -> pd.DataFrame:
    fps = []
    valid_idx = []
    for i, smi in enumerate(df[smiles_col].tolist()):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        valid_idx.append(i)
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(fps), threshold, isDistData=True) if fps else []
    labels = {vid: f"cluster_{ci}" for ci, cl in enumerate(clusters) for vid in [valid_idx[j] for j in cl]}
    out = df.copy()
    out["chemotype_cluster"] = [labels.get(i, "unassigned") for i in range(len(out))]
    out["chemotype_core"] = out[smiles_col].apply(murcko_scaffold_smiles)
    return out


def cluster_chemotypes(df: pd.DataFrame, method: str = "scaffold", threshold: float = 0.6, smiles_col: str = "smiles") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if method == "butina":
        return cluster_by_butina(df, threshold=threshold, smiles_col=smiles_col)
    return cluster_by_scaffold(df, smiles_col=smiles_col)
