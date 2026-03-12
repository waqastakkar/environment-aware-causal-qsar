from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

from model_gnn import CausalQSARModel


def _load_model(checkpoint: Path, run_dir: Path, node_dim: int, edge_dim: int, device: torch.device):
    cfg = yaml.safe_load((run_dir / "configs" / "resolved_config.yaml").read_text(encoding="utf-8"))
    state = torch.load(checkpoint, map_location=device)
    n_envs = int(state["adversary.2.weight"].shape[0]) if "adversary.2.weight" in state else 2
    model = CausalQSARModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        z_dim=int(cfg.get("z_dim", 128)),
        z_inv_dim=int(cfg.get("z_inv_dim", 64)),
        z_spu_dim=int(cfg.get("z_spu_dim", 64)),
        n_envs=n_envs,
        task=str(cfg.get("task", "regression")),
        encoder=str(cfg.get("encoder", "gine")),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _predict(model, graphs, batch_size: int, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    rows = []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            yhat = model(b)["yhat"].detach().cpu().numpy()
            for i in range(b.num_graphs):
                rows.append({"compound_id": b.molecule_id[i], "yhat": float(yhat[i])})
    return pd.DataFrame(rows)


def run_inference(graphs, run_dir: str | Path, ensemble_manifest: str | None = None, batch_size: int = 256) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = int(graphs[0].x.shape[1]); edge_dim = int(graphs[0].edge_attr.shape[1])

    single_ckpt = run_dir / "checkpoints" / "best.pt"
    single_model = _load_model(single_ckpt, run_dir, node_dim, edge_dim, device)
    single = _predict(single_model, graphs, batch_size, device).rename(columns={"yhat": "yhat_pIC50"})

    ckpts = [single_ckpt]
    if ensemble_manifest:
        manifest = json.loads(Path(ensemble_manifest).read_text(encoding="utf-8"))
        listed = manifest.get("checkpoint_paths") or manifest.get("checkpoints") or []
        ckpts = [Path(c) for c in listed if Path(c).exists()] or ckpts

    all_preds = []
    for c in ckpts:
        c_run = c.parents[1] if c.name.endswith(".pt") else run_dir
        m = _load_model(Path(c), c_run, node_dim, edge_dim, device)
        pr = _predict(m, graphs, batch_size, device).rename(columns={"yhat": f"pred_{len(all_preds)}"})
        all_preds.append(pr)
    ens = all_preds[0]
    for pr in all_preds[1:]:
        ens = ens.merge(pr, on="compound_id", how="inner")
    pred_cols = [c for c in ens.columns if c.startswith("pred_")]
    ens["score_mean"] = ens[pred_cols].mean(axis=1)
    ens["score_std"] = ens[pred_cols].std(axis=1).fillna(0.0)
    return single, ens
