#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from data_graph import read_split_ids


def _fp_from_smiles(smiles: str, radius: int = 2, nbits: int = 2048):
    m = Chem.MolFromSmiles(str(smiles))
    return AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits) if m is not None else None


def build_or_load_train_fingerprint_index(run_dir: str | Path, radius: int = 2, nbits: int = 2048) -> pd.DataFrame:
    run_dir = Path(run_dir)
    idx_path = run_dir / "artifacts" / "train_fingerprint_index.parquet"
    if idx_path.exists():
        return pd.read_parquet(idx_path)

    import yaml

    cfg = yaml.safe_load((run_dir / "configs" / "resolved_config.yaml").read_text(encoding="utf-8"))
    ds = pd.read_parquet(cfg["dataset_parquet"])
    split_ids = read_split_ids(cfg["splits_dir"], cfg["split_name"])
    train = ds[ds["molecule_id"].astype(str).isin({str(x) for x in split_ids["train"]})].copy()
    rows = []
    for r in train.itertuples(index=False):
        fp = _fp_from_smiles(getattr(r, "smiles"), radius=radius, nbits=nbits)
        if fp is None:
            continue
        rows.append({"molecule_id": str(getattr(r, "molecule_id")), "smiles": getattr(r, "smiles"), "fp_bits": fp.ToBitString()})
    out = pd.DataFrame(rows)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(idx_path, index=False)
    return out


def fingerprint_ad(train_df: pd.DataFrame, test_df: pd.DataFrame, radius: int = 2, nbits: int = 2048) -> pd.DataFrame:
    out = test_df[["molecule_id"]].copy() if "molecule_id" in test_df.columns else test_df[["compound_id"]].rename(columns={"compound_id": "molecule_id"})
    train_fp = []
    if "fp_bits" in train_df.columns:
        for b in train_df["fp_bits"].fillna(""):
            if b:
                train_fp.append(DataStructs.CreateFromBitString(str(b)))
    else:
        for s in train_df["smiles"].fillna(""):
            fp = _fp_from_smiles(s, radius=radius, nbits=nbits)
            if fp is not None:
                train_fp.append(fp)

    dists = np.full(len(test_df), np.nan, dtype=float)
    smiles_series = test_df["smiles"].fillna("")
    for i, s in enumerate(smiles_series):
        fp = _fp_from_smiles(s, radius=radius, nbits=nbits)
        if fp is None or not train_fp:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fp)
        dists[i] = 1.0 - float(max(sims)) if sims else np.nan
    out["ad_distance_fingerprint"] = dists
    return out


def embedding_ad(train_z: pd.DataFrame, test_z: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    out = test_z[["molecule_id"]].copy()
    zcols = [c for c in train_z.columns if c.startswith("z_inv_") and c in test_z.columns]
    if not zcols:
        out["ad_distance_embedding"] = np.nan
        return out
    tr = train_z[zcols].to_numpy(dtype=float)
    te = test_z[zcols].to_numpy(dtype=float)
    d = np.sqrt(((te[:, None, :] - tr[None, :, :]) ** 2).sum(axis=2))
    out["ad_distance_embedding"] = np.sort(d, axis=1)[:, : max(1, k)].mean(axis=1)
    return out


def binned_relationship(df: pd.DataFrame, xcol: str, ycol: str, bins: int = 10, out_x: str = "ad_bin") -> pd.DataFrame:
    work = df[[xcol, ycol]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=[out_x, "y_mean", "n"])
    work[out_x] = pd.qcut(work[xcol], q=min(bins, max(2, work[xcol].nunique())), duplicates="drop")
    return work.groupby(out_x, observed=False).agg(y_mean=(ycol, "mean"), n=(ycol, "size"), x_mean=(xcol, "mean")).reset_index()
