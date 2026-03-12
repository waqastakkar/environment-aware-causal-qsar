from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


def compute_properties(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    out = df.copy()
    props = []
    for smi in out[smiles_col].fillna(""):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            props.append({k: None for k in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "fractionCSP3", "Rings"]})
            continue
        props.append(
            {
                "MW": float(Descriptors.MolWt(mol)),
                "LogP": float(Crippen.MolLogP(mol)),
                "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
                "HBD": int(Lipinski.NumHDonors(mol)),
                "HBA": int(Lipinski.NumHAcceptors(mol)),
                "RotB": int(Lipinski.NumRotatableBonds(mol)),
                "fractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                "Rings": int(rdMolDescriptors.CalcNumRings(mol)),
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(props)], axis=1)
