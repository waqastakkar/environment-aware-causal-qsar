from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data


ATOM_LIST = list(range(1, 119))
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices):
    v = [0.0] * len(choices)
    if value in choices:
        v[choices.index(value)] = 1.0
    return v


def atom_features(atom: Chem.rdchem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_LIST)
        + [atom.GetDegree(), atom.GetFormalCharge(), atom.GetTotalNumHs(), atom.GetIsAromatic()]
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION)
        + [atom.IsInRing()]
    )


def bond_features(bond: Chem.rdchem.Bond) -> list[float]:
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [bond.GetIsConjugated(), bond.IsInRing(), bond.GetStereo()]


@dataclass
class GraphBuildConfig:
    smiles_col: str = "smiles"
    id_col: str = "molecule_id"
    label_col: str = "pIC50"
    env_col: str = "env_id_manual"
    sample_weight_col: str | None = None


def smiles_to_data(smiles: str, y: float, env_id: int, molecule_id: str, sample_weight: float = 1.0) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    eattr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges += [[i, j], [j, i]]
        eattr += [bf, bf]
    if len(edges) == 0:
        edges = [[0, 0]]
        eattr = [[0.0] * (len(BOND_TYPES) + 3)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(eattr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.y = torch.tensor(y, dtype=torch.float)
    data.env = torch.tensor(int(env_id), dtype=torch.long)
    data.sample_weight = torch.tensor(float(sample_weight), dtype=torch.float)
    data.molecule_id = str(molecule_id)
    return data


def dataframe_to_graphs(df: pd.DataFrame, cfg: GraphBuildConfig) -> list[Data]:
    work = df.copy()
    if cfg.env_col in work.columns:
        env_numeric = pd.to_numeric(work[cfg.env_col], errors="coerce")
        if env_numeric.notna().all():
            work[cfg.env_col] = env_numeric.fillna(-1).astype(int)
        else:
            # Step 11 can provide string environment labels, so encode them to stable integer ids for graph tensors.
            work[cfg.env_col] = pd.Categorical(work[cfg.env_col].fillna("missing")).codes.astype(int)

    graphs = []
    for row in work.itertuples(index=False):
        d = smiles_to_data(
            getattr(row, cfg.smiles_col),
            float(getattr(row, cfg.label_col)),
            int(getattr(row, cfg.env_col)),
            getattr(row, cfg.id_col),
            float(getattr(row, cfg.sample_weight_col)) if cfg.sample_weight_col and hasattr(row, cfg.sample_weight_col) else 1.0,
        )
        if d is not None:
            graphs.append(d)
    return graphs


def remap_env_ids(df: pd.DataFrame, env_col: str) -> tuple[pd.DataFrame, dict]:
    env_values = sorted(df[env_col].dropna().unique().tolist())
    mapping = {e: i for i, e in enumerate(env_values)}
    out = df.copy()
    out[env_col] = out[env_col].map(mapping).astype(int)
    return out, mapping


def read_split_ids(splits_dir: str, split_name: str) -> dict[str, set[str]]:
    base = f"{splits_dir}/{split_name}"
    out = {}
    for part in ["train", "val", "test"]:
        part_df = pd.read_csv(f"{base}/{part}_ids.csv")
        col = "molecule_id" if "molecule_id" in part_df.columns else part_df.columns[0]
        out[part] = set(part_df[col].astype(str).tolist())
    return out


def split_dataframe_by_ids(df: pd.DataFrame, ids: dict[str, set[str]], id_col: str = "molecule_id") -> dict[str, pd.DataFrame]:
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    return {k: df[df[id_col].isin(v)].copy() for k, v in ids.items()}


def ensure_required_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        available = sorted(df.columns.tolist())
        raise ValueError(
            "Missing required columns: "
            f"{missing}. Available columns: {available}. "
            "Pass --env_col with an existing column name when launching training/benchmark scripts."
        )
