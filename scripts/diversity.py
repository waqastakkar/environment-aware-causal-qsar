#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None
    Butina = None


@dataclass
class DiversityResult:
    clustering_summary: pd.DataFrame
    diversity_selection: pd.DataFrame


def _fp(smiles: str, radius: int = 2, nbits: int = 2048):
    if Chem is None:
        raise RuntimeError("RDKit is required for diversity analysis.")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def butina_cluster(df: pd.DataFrame, smiles_col: str, threshold: float) -> pd.DataFrame:
    if Butina is None:
        raise RuntimeError("RDKit Butina clustering unavailable.")
    work = df.copy().reset_index(drop=True)
    fps = [_fp(s) for s in work[smiles_col].fillna("")]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
    clusters = Butina.ClusterData(dists, nPts=n, distThresh=1.0 - threshold, isDistData=True)
    cluster_id = {}
    for cid, members in enumerate(clusters, start=1):
        for m in members:
            cluster_id[m] = cid
    work["cluster_id"] = [cluster_id.get(i, -1) for i in range(n)]
    work["cluster_size"] = work.groupby("cluster_id")["cluster_id"].transform("size")
    return work


def select_cluster_leaders(clustered_df: pd.DataFrame) -> pd.DataFrame:
    leaders = (
        clustered_df.sort_values(["cluster_id", "score_mean", "uncertainty_std"], ascending=[True, False, True])
        .groupby("cluster_id", as_index=False)
        .head(1)
        .copy()
    )
    leaders["selection_reason"] = "cluster_leader"
    return leaders.sort_values(["score_mean", "uncertainty_std"], ascending=[False, True])


def run_diversity_selection(df: pd.DataFrame, smiles_col: str, threshold: float = 0.65) -> DiversityResult:
    clustered = butina_cluster(df, smiles_col=smiles_col, threshold=threshold)
    leaders = select_cluster_leaders(clustered)
    summary = (
        clustered.groupby("cluster_id", as_index=False)
        .agg(cluster_size=("cluster_size", "first"), best_score=("score_mean", "max"), median_score=("score_mean", "median"))
        .sort_values("cluster_size", ascending=False)
    )
    return DiversityResult(clustering_summary=summary, diversity_selection=leaders)
