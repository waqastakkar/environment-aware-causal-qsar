#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None
    MurckoScaffold = None


@dataclass
class NoveltyResult:
    novelty_report: pd.DataFrame
    scaffold_novelty: pd.DataFrame
    topk_annotated: pd.DataFrame


def _require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError("RDKit is required for novelty analysis. Install rdkit first.")


def canonical_inchikey(smiles: str) -> str | None:
    _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def bemis_murcko(smiles: str) -> str | None:
    _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def max_tanimoto_to_train(query_fps: Iterable, train_fps: list) -> list[float]:
    _require_rdkit()
    if not train_fps:
        return [0.0 for _ in query_fps]
    out = []
    for q in query_fps:
        if q is None:
            out.append(float("nan"))
        else:
            out.append(float(max(DataStructs.BulkTanimotoSimilarity(q, train_fps))))
    return out


def compute_novelty(topk_df: pd.DataFrame, train_df: pd.DataFrame, smiles_col: str, id_col: str) -> NoveltyResult:
    top = topk_df.copy()
    train = train_df.copy()

    if "inchikey" not in top.columns:
        top["inchikey"] = top[smiles_col].map(canonical_inchikey)
    if "inchikey" not in train.columns:
        train["inchikey"] = train[smiles_col].map(canonical_inchikey)

    top["scaffold"] = top[smiles_col].map(bemis_murcko)
    train["scaffold"] = train[smiles_col].map(bemis_murcko)

    train_inchi = set(train["inchikey"].dropna())
    train_scaf = set(train["scaffold"].dropna())

    top["exact_seen_in_train"] = top["inchikey"].isin(train_inchi)
    top["scaffold_seen_in_train"] = top["scaffold"].isin(train_scaf)

    train_fps = [morgan_fp(s) for s in train[smiles_col].fillna("")]
    train_fps = [fp for fp in train_fps if fp is not None]
    qfps = [morgan_fp(s) for s in top[smiles_col].fillna("")]
    top["max_tanimoto_to_train"] = max_tanimoto_to_train(qfps, train_fps)

    novelty_report = pd.DataFrame(
        [
            {"metric": "topk_size", "value": int(len(top))},
            {"metric": "exact_overlap_count", "value": int(top["exact_seen_in_train"].sum())},
            {"metric": "exact_overlap_rate", "value": float(top["exact_seen_in_train"].mean()) if len(top) else 0.0},
            {
                "metric": "scaffold_novel_count",
                "value": int((~top["scaffold_seen_in_train"]).fillna(False).sum()),
            },
            {
                "metric": "scaffold_novel_rate",
                "value": float((~top["scaffold_seen_in_train"]).fillna(False).mean()) if len(top) else 0.0,
            },
            {
                "metric": "mean_max_tanimoto_to_train",
                "value": float(top["max_tanimoto_to_train"].dropna().mean()) if len(top) else 0.0,
            },
        ]
    )

    scaffold_novelty = (
        top.groupby("scaffold", dropna=False)
        .agg(
            count=(id_col, "count"),
            seen_in_train=("scaffold_seen_in_train", "max"),
            best_score=("score_mean", "max"),
            median_score=("score_mean", "median"),
        )
        .reset_index()
        .sort_values(["seen_in_train", "best_score", "count"], ascending=[True, False, False])
    )

    return NoveltyResult(novelty_report=novelty_report, scaffold_novelty=scaffold_novelty, topk_annotated=top)
