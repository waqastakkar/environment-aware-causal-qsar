#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except Exception:
    class _NP:
        nan = float("nan")
        @staticmethod
        def isnan(x):
            try:
                return x != x
            except Exception:
                return True
    np = _NP()


@dataclass
class MoleculeRecord:
    smiles: str
    valid: bool
    reason: str | None
    props: dict[str, float]


def _nan_props() -> dict[str, float]:
    return {
        "mw": np.nan,
        "logp": np.nan,
        "hbd": np.nan,
        "hba": np.nan,
        "tpsa": np.nan,
        "rotatable_bonds": np.nan,
        "aromatic_rings": np.nan,
    }


def compute_properties(smiles: str) -> dict[str, float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return _nan_props()

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return _nan_props()

    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "aromatic_rings": float(Lipinski.RingCount(mol)),
    }


def cns_mpo_score(props: dict[str, float]) -> float:
    # Lightweight proxy score in [0, 6], designed as fallback when precomputed BBB labels are missing.
    mw = props.get("mw", np.nan)
    logp = props.get("logp", np.nan)
    hbd = props.get("hbd", np.nan)
    tpsa = props.get("tpsa", np.nan)
    if np.isnan(mw) or np.isnan(logp) or np.isnan(hbd) or np.isnan(tpsa):
        return np.nan
    score = 6.0
    score -= max(0.0, (mw - 360.0) / 120.0)
    score -= max(0.0, abs(logp - 2.5) - 1.5)
    score -= max(0.0, (hbd - 1.0) * 0.7)
    score -= max(0.0, (tpsa - 70.0) / 40.0)
    return float(max(0.0, min(6.0, score)))


def sanitize_smiles(smiles: str) -> MoleculeRecord:
    if not isinstance(smiles, str) or not smiles.strip():
        return MoleculeRecord(smiles=smiles, valid=False, reason="empty_smiles", props=_nan_props())
    try:
        from rdkit import Chem
    except Exception:
        # Fallback validation: basic character filter, not chemical sanitization.
        bad = any(ch.isspace() for ch in smiles)
        if bad:
            return MoleculeRecord(smiles=smiles, valid=False, reason="invalid_chars", props=_nan_props())
        return MoleculeRecord(smiles=smiles, valid=True, reason=None, props=compute_properties(smiles))

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return MoleculeRecord(smiles=smiles, valid=False, reason="rdkit_parse_failed", props=_nan_props())
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return MoleculeRecord(smiles=smiles, valid=False, reason=f"sanitize_failed:{type(exc).__name__}", props=_nan_props())
    if len(Chem.GetMolFrags(mol)) > 1:
        return MoleculeRecord(smiles=smiles, valid=False, reason="disconnected_fragments", props=_nan_props())
    clean = Chem.MolToSmiles(mol)
    return MoleculeRecord(smiles=clean, valid=True, reason=None, props=compute_properties(clean))


def synthetic_feasibility_ok(props: dict[str, float], mw_max: float = 650.0, logp_abs_max: float = 7.0, tpsa_max: float = 220.0) -> bool:
    mw = props.get("mw", np.nan)
    logp = props.get("logp", np.nan)
    tpsa = props.get("tpsa", np.nan)
    if np.isnan(mw) or np.isnan(logp) or np.isnan(tpsa):
        return False
    return bool(mw <= mw_max and abs(logp) <= logp_abs_max and tpsa <= tpsa_max)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:
        return float(smiles_a == smiles_b)
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return np.nan
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, radius=2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, radius=2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def murcko_scaffold(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def has_motif(smiles: str, motif_smarts: str) -> bool:
    if not motif_smarts:
        return True
    try:
        from rdkit import Chem
    except Exception:
        return motif_smarts in smiles
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts(motif_smarts)
    return bool(mol and patt and mol.HasSubstructMatch(patt))
