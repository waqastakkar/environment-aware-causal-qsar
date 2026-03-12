from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Batch

from data_graph import GraphBuildConfig, dataframe_to_graphs


def load_feature_schema(run_dir: str | Path) -> dict:
    p = Path(run_dir) / "artifacts" / "feature_schema.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing required feature schema: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def featurize_library(df: pd.DataFrame, run_dir: str | Path) -> tuple[list, pd.DataFrame, dict]:
    schema = load_feature_schema(run_dir)
    work = df.copy().rename(columns={"compound_id": "molecule_id", "canonical_smiles": "smiles"})
    work["dummy_y"] = 0.0
    work["dummy_env"] = 0
    cfg = GraphBuildConfig(smiles_col="smiles", id_col="molecule_id", label_col="dummy_y", env_col="dummy_env")
    graphs = dataframe_to_graphs(work, cfg)
    ok_ids = {g.molecule_id for g in graphs}
    feat_df = df[df["compound_id"].astype(str).isin(ok_ids)].copy()
    if not graphs:
        raise ValueError("No molecules featurized; cannot continue.")

    batch = Batch.from_data_list(graphs[:1])
    node_dim = int(batch.x.shape[1]); edge_dim = int(batch.edge_attr.shape[1])
    exp_node = schema.get("node_feature_dim")
    exp_edge = schema.get("edge_feature_dim")
    if exp_node is not None and node_dim != int(exp_node):
        raise ValueError(f"Feature schema mismatch: node_feature_dim expected={exp_node} observed={node_dim}")
    if exp_edge is not None and edge_dim != int(exp_edge):
        raise ValueError(f"Feature schema mismatch: edge_feature_dim expected={exp_edge} observed={edge_dim}")

    norm_path = Path(run_dir) / "artifacts" / "normalization_stats.json"
    norm_stats = json.loads(norm_path.read_text()) if norm_path.exists() else None
    report = {"featurized_ok": len(graphs), "featurized_fail": int(len(df) - len(graphs)), "normalization_applied": bool(norm_stats)}
    return graphs, feat_df, report
