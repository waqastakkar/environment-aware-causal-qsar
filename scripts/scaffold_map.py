#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaf)


def map_hits_to_training_scaffolds(
    hits_df: pd.DataFrame,
    training_scaffolds: pd.DataFrame,
    mode: str = "exact",
    sim_threshold: float = 0.7,
    id_col: str = "molecule_id",
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    h = hits_df.copy()
    h["hit_scaffold"] = h[smiles_col].apply(murcko_scaffold_smiles)
    ts = training_scaffolds.copy()
    scaffold_col = "core_smiles" if "core_smiles" in ts.columns else ("scaffold" if "scaffold" in ts.columns else None)
    if scaffold_col is None:
        return h.assign(mapped=False, matched_scaffold=None, series_id=None, scaffold_similarity=0.0)

    train_rows = ts[[c for c in ["series_id", scaffold_col] if c in ts.columns]].drop_duplicates().rename(columns={scaffold_col: "train_scaffold"})
    if train_rows.empty:
        return h.assign(mapped=False, matched_scaffold=None, series_id=None, scaffold_similarity=0.0)

    if mode == "exact":
        merged = h.merge(train_rows, left_on="hit_scaffold", right_on="train_scaffold", how="left")
        merged["scaffold_similarity"] = (merged["hit_scaffold"] == merged["train_scaffold"]).astype(float)
        merged["mapped"] = merged["train_scaffold"].notna()
        merged["matched_scaffold"] = merged["train_scaffold"]
        return merged

    # similarity mode
    train_fp = {}
    for sc in train_rows["train_scaffold"].dropna().unique():
        m = Chem.MolFromSmiles(sc)
        if m is not None:
            train_fp[sc] = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
    out = []
    for _, r in h.iterrows():
        hs = r.get("hit_scaffold")
        best_sc, best_sid, best_sim = None, None, 0.0
        hm = Chem.MolFromSmiles(hs) if isinstance(hs, str) else None
        hfp = AllChem.GetMorganFingerprintAsBitVect(hm, radius=2, nBits=2048) if hm is not None else None
        if hfp is not None:
            for _, tr in train_rows.iterrows():
                sc = tr["train_scaffold"]
                tfp = train_fp.get(sc)
                if tfp is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(hfp, tfp)
                if sim > best_sim:
                    best_sc, best_sid, best_sim = sc, tr.get("series_id"), sim
        rec = r.to_dict()
        rec.update({
            "mapped": bool(best_sim >= sim_threshold),
            "matched_scaffold": best_sc if best_sim >= sim_threshold else None,
            "series_id": best_sid if best_sim >= sim_threshold else None,
            "scaffold_similarity": best_sim,
        })
        out.append(rec)
    return pd.DataFrame(out)
