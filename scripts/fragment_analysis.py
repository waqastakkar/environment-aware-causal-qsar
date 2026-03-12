#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from stats_utils import enrichment_2x2, multiple_testing_correction

FUNCTIONAL_GROUP_SMARTS = {
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "sulfonamide": "[NX3][SX4](=[OX1])(=[OX1])[#6]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "carboxylic_acid": "C(=O)[OH]",
    "heteroaromatic": "[a;r5,r6]",
    "halogen": "[F,Cl,Br,I]",
    "ether": "[OD2]([#6])[#6]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
}


@dataclass
class FragmentOutputs:
    fragment_library: pd.DataFrame
    fragment_enrichment: pd.DataFrame
    functional_group_enrichment: pd.DataFrame
    quality_report: pd.DataFrame


def _extract_fragments(smiles: str, method: str = "brics") -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    if method == "brics":
        return set(BRICS.BRICSDecompose(mol))
    if method == "murcko":
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return {Chem.MolToSmiles(core)} if core and core.GetNumAtoms() else set()
    return {Chem.MolToSmiles(mol)}


def _functional_tags(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    tags = set()
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            tags.add(name)
    return tags


def build_feature_presence(df: pd.DataFrame, id_col: str = "molecule_id", smiles_col: str = "smiles", method: str = "brics") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frag_rows: list[dict] = []
    fg_rows: list[dict] = []
    quality: list[dict] = []
    for _, r in df.iterrows():
        mid = r.get(id_col)
        smi = r.get(smiles_col)
        if not isinstance(smi, str):
            quality.append({id_col: mid, "status": "invalid_smiles"})
            continue
        frags = _extract_fragments(smi, method=method)
        fgs = _functional_tags(smi)
        for f in frags:
            frag_rows.append({id_col: mid, "feature": f, "feature_type": "fragment", "present": 1})
        for fg in fgs:
            fg_rows.append({id_col: mid, "feature": fg, "feature_type": "functional_group", "present": 1})
    return pd.DataFrame(frag_rows), pd.DataFrame(fg_rows), pd.DataFrame(quality)


def enrichment_hits_vs_background(
    hit_features: pd.DataFrame,
    bg_features: pd.DataFrame,
    hit_ids: list[str],
    bg_ids: list[str],
    id_col: str = "molecule_id",
    correction: str = "bh",
) -> pd.DataFrame:
    feats = sorted(set(hit_features.get("feature", [])).union(set(bg_features.get("feature", []))))
    hit_set = {str(x) for x in hit_ids}
    bg_set = {str(x) for x in bg_ids}
    hit_has = hit_features.groupby("feature")[id_col].apply(lambda s: {str(x) for x in s}).to_dict() if not hit_features.empty else {}
    bg_has = bg_features.groupby("feature")[id_col].apply(lambda s: {str(x) for x in s}).to_dict() if not bg_features.empty else {}
    rows = []
    for f in feats:
        h = hit_has.get(f, set()) & hit_set
        b = bg_has.get(f, set()) & bg_set
        pos_has = len(h)
        pos_not = max(0, len(hit_set) - pos_has)
        neg_has = len(b)
        neg_not = max(0, len(bg_set) - neg_has)
        row = {"feature": f, **enrichment_2x2(pos_has, pos_not, neg_has, neg_not), "hit_prevalence": (pos_has / len(hit_set)) if hit_set else 0.0}
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = multiple_testing_correction(out["p_value"].values, method=correction)
    return out.sort_values(["q_value", "log_odds_ratio"], ascending=[True, False]).reset_index(drop=True)


def run_fragment_analysis(df: pd.DataFrame, method: str = "brics", id_col: str = "molecule_id", smiles_col: str = "smiles") -> FragmentOutputs:
    frag_df, fg_df, quality_df = build_feature_presence(df, id_col=id_col, smiles_col=smiles_col, method=method)
    frag_enrich = pd.DataFrame()
    fg_enrich = pd.DataFrame()
    if quality_df.empty:
        quality_df = pd.DataFrame([{"status": "ok", "n_molecules": len(df)}])
    return FragmentOutputs(frag_df, frag_enrich, fg_enrich, quality_df)
