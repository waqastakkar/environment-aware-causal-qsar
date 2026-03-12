#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


@dataclass
class ShapeOutputs:
    descriptors: pd.DataFrame
    failures: pd.DataFrame
    vs_potency: pd.DataFrame
    vs_residuals: pd.DataFrame


def _shape_bin(npr1: float, npr2: float) -> str:
    if pd.isna(npr1) or pd.isna(npr2):
        return "unknown"
    if npr1 < 0.35 and npr2 > 0.85:
        return "rod"
    if npr1 > 0.7 and npr2 > 0.85:
        return "sphere"
    return "disk"


def _embed_and_select(mol: Chem.Mol, n_confs: int, seed: int, select: str) -> tuple[int | None, str | None]:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params)
    if not conf_ids:
        return None, "embed_failed"
    if select == "lowest_uff_energy":
        try:
            energies = []
            for cid in conf_ids:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                ff.Minimize(maxIts=200)
                energies.append((ff.CalcEnergy(), cid))
            energies.sort(key=lambda x: x[0])
            return int(energies[0][1]), None
        except Exception:
            return int(conf_ids[0]), "uff_failed_fallback_first"
    return int(conf_ids[0]), None


def run_shape_analysis(
    df: pd.DataFrame,
    n_confs: int = 10,
    seed: int = 42,
    select: str = "lowest_uff_energy",
    subset_ids: list[str] | None = None,
    id_col: str = "molecule_id",
    smiles_col: str = "smiles",
) -> ShapeOutputs:
    use_df = df
    if subset_ids is not None:
        use_df = df[df[id_col].astype(str).isin({str(x) for x in subset_ids})].copy()

    desc_rows: list[dict] = []
    fail_rows: list[dict] = []
    for _, r in use_df.iterrows():
        mid, smi = r.get(id_col), r.get(smiles_col)
        mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
        if mol is None:
            fail_rows.append({id_col: mid, "reason": "invalid_smiles"})
            continue
        mol = Chem.AddHs(mol)
        cid, warn = _embed_and_select(mol, n_confs, seed, select)
        if cid is None:
            fail_rows.append({id_col: mid, "reason": warn or "embed_failed"})
            continue
        try:
            npr1 = rdMolDescriptors.CalcNPR1(mol, confId=cid)
            npr2 = rdMolDescriptors.CalcNPR2(mol, confId=cid)
            rec = r.to_dict()
            rec.update(
                {
                    "NPR1": npr1,
                    "NPR2": npr2,
                    "radius_gyration": rdMolDescriptors.CalcRadiusOfGyration(mol, confId=cid),
                    "asphericity": rdMolDescriptors.CalcAsphericity(mol, confId=cid),
                    "eccentricity": rdMolDescriptors.CalcEccentricity(mol, confId=cid),
                    "fractionCSP3": Descriptors.FractionCSP3(Chem.RemoveHs(mol)),
                    "shape_bin": _shape_bin(npr1, npr2),
                    "shape_warning": warn,
                }
            )
            desc_rows.append(rec)
        except Exception as exc:
            fail_rows.append({id_col: mid, "reason": f"descriptor_failed:{exc}"})
    desc = pd.DataFrame(desc_rows)
    fails = pd.DataFrame(fail_rows)
    vs_p = desc[[c for c in [id_col, "split", "pIC50", "NPR1", "NPR2", "radius_gyration", "asphericity", "eccentricity", "fractionCSP3", "shape_bin", "cns_like"] if c in desc.columns]].copy()
    if "yhat" in desc.columns and "pIC50" in desc.columns:
        desc["residual_abs"] = (desc["pIC50"] - desc["yhat"]).abs()
    vs_r = desc[[c for c in [id_col, "split", "residual_abs", "NPR1", "NPR2", "radius_gyration", "asphericity", "eccentricity", "fractionCSP3", "shape_bin", "cns_like"] if c in desc.columns]].copy()
    return ShapeOutputs(desc, fails, vs_p, vs_r)
