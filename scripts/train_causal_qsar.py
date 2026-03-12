#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader

from data_graph import GraphBuildConfig, dataframe_to_graphs, ensure_required_columns, read_split_ids, remap_env_ids, split_dataframe_by_ids
from losses import adversary_env_loss, compute_env_class_weights, disentanglement_loss, irmv1_penalty, prediction_loss
from metrics import (
    classification_metrics,
    expected_calibration_error,
    linear_probe_env_predictability,
    per_environment_metrics,
    regression_calibration,
    regression_metrics,
)
from model_gnn import CausalQSARModel
from plot_style import PlotStyle, configure_matplotlib, parse_palette, style_axis


@dataclass
class TrainConfig:
    target: str
    dataset_parquet: str
    splits_dir: str
    split_name: str
    outdir: str
    task: str
    label_col: str
    env_col: str
    encoder: str
    z_dim: int
    z_inv_dim: int
    z_spu_dim: int
    lambda_adv: float
    lambda_irm: float
    lambda_dis: float
    loss_pred: str
    loss_cls: str
    loss_env: str
    irm_mode: str
    disentangle: str
    warmup_epochs: int
    ramp_epochs: int
    epochs: int
    early_stopping_patience: int
    batch_size: int
    max_rows: int
    lr: float
    seed: int
    bbb_parquet: str | None
    sample_weight_col: str | None
    svg: bool
    font: str
    bold_text: bool
    palette: str
    font_title: int
    font_label: int
    font_tick: int
    font_legend: int
    run_id: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def schedule_weight(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch - warmup_epochs) / float(ramp_epochs)))


def evaluate(model, loader, device, task, lambda_grl=1.0):
    model.eval()
    rows = []
    z_all = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, lambda_grl=lambda_grl)
            pred = out["yhat"].detach().cpu().numpy()
            if task == "classification":
                pred = 1 / (1 + np.exp(-pred))
            envhat = out["envhat"].argmax(dim=1).cpu().numpy()
            for i in range(batch.num_graphs):
                rows.append(
                    {
                        "molecule_id": batch.molecule_id[i],
                        "y_true": float(batch.y[i].cpu()),
                        "y_pred": float(pred[i]),
                        "env_id_manual": int(batch.env[i].cpu()),
                        "env_pred": int(envhat[i]),
                    }
                )
            z_all.append(out["z_inv"].cpu().numpy())
    df = pd.DataFrame(rows, columns=["molecule_id", "y_true", "y_pred", "env_id_manual", "env_pred"])
    z = np.concatenate(z_all, axis=0) if z_all else np.empty((0, 1))
    return df, z


def make_dirs(root: Path):
    for sub in ["checkpoints", "configs", "logs", "predictions", "reports", "figures", "provenance", "artifacts"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--dataset_parquet", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--split_name", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--label_col", required=True)
    p.add_argument("--env_col", required=True)
    p.add_argument("--encoder", default="gine", choices=["gine", "graph_transformer"])
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--z_inv_dim", type=int, default=64)
    p.add_argument("--z_spu_dim", type=int, default=64)
    p.add_argument("--lambda_adv", type=float, default=0.5)
    p.add_argument("--lambda_irm", type=float, default=0.0)
    p.add_argument("--lambda_dis", type=float, default=0.1)
    p.add_argument("--loss_pred", choices=["mse", "huber"], default="huber")
    p.add_argument("--loss_cls", choices=["bce", "focal"], default="bce")
    p.add_argument("--loss_env", choices=["ce", "weighted_ce"], default="ce")
    p.add_argument("--irm_mode", choices=["none", "irmv1"], default="none")
    p.add_argument("--disentangle", choices=["none", "orthogonality", "hsic"], default="orthogonality")
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--ramp_epochs", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--early_stopping_patience", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_rows", type=int, default=0, help="Optional cap on rows loaded from dataset_parquet (0 disables).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bbb_parquet", default=None)
    p.add_argument("--sample_weight_col", default=None)
    p.add_argument("--svg", action="store_true")
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--bold_text", action="store_true", default=True)
    p.add_argument("--palette", default="nature5")
    p.add_argument("--font_title", type=int, default=16)
    p.add_argument("--font_label", type=int, default=14)
    p.add_argument("--font_tick", type=int, default=12)
    p.add_argument("--font_legend", type=int, default=12)
    p.add_argument("--run_id", default=None)
    return p.parse_args()


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_training_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    id_col = _first_present(out, ["molecule_id", "molecule_chembl_id", "chembl_molecule_id", "compound_id", "mol_id", "id"])
    if id_col is None:
        logging.warning("No molecule identifier column found; creating molecule_id from dataframe index")
        out["molecule_id"] = out.index.astype(str)
    elif id_col != "molecule_id":
        logging.warning("Creating canonical molecule_id from source column '%s'", id_col)
        out["molecule_id"] = out[id_col]
    out["molecule_id"] = out["molecule_id"].astype(str)

    smiles_col = _first_present(out, ["smiles", "canonical_smiles", "smiles_canonical"])
    if smiles_col is None:
        raise ValueError("Missing required SMILES column: expected one of smiles/canonical_smiles/smiles_canonical")
    if smiles_col != "smiles":
        out["smiles"] = out[smiles_col]

    return out


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.run_id = run_id
    cfg = TrainConfig(**vars(args))
    seed_everything(cfg.seed)

    run_root = Path(cfg.outdir) / cfg.target / cfg.split_name / cfg.run_id
    make_dirs(run_root)

    logging.basicConfig(filename=run_root / "logs/train.log", level=logging.INFO)
    style = PlotStyle(
        font_family=cfg.font,
        font_title=cfg.font_title,
        font_label=cfg.font_label,
        font_tick=cfg.font_tick,
        font_legend=cfg.font_legend,
        bold_text=True,
        palette=parse_palette(cfg.palette),
    )
    configure_matplotlib(style, svg=True)

    with (run_root / "configs/train_config.yaml").open("w") as f:
        yaml.safe_dump(vars(args), f)
    with (run_root / "configs/resolved_config.yaml").open("w") as f:
        yaml.safe_dump(asdict(cfg), f)

    df = normalize_training_inputs(pd.read_parquet(cfg.dataset_parquet))
    if cfg.max_rows and cfg.max_rows > 0 and len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=cfg.seed).reset_index(drop=True)
        logging.info("Applied max_rows=%s; sampled dataset down to %s rows", cfg.max_rows, len(df))

    resolved_env_col = cfg.env_col
    if resolved_env_col not in df.columns:
        for candidate in ["env_id_manual", "env_id"]:
            if candidate in df.columns:
                logging.warning(
                    "Requested env_col '%s' not found; falling back to '%s'",
                    cfg.env_col,
                    candidate,
                )
                print(f"WARNING: requested env_col '{cfg.env_col}' not found; using '{candidate}'")
                resolved_env_col = candidate
                break

    cfg.env_col = resolved_env_col
    ensure_required_columns(df, ["molecule_id", "smiles", cfg.label_col, cfg.env_col])
    if cfg.sample_weight_col and cfg.sample_weight_col not in df.columns:
        raise ValueError(f"sample_weight_col='{cfg.sample_weight_col}' not in dataset")
    df, env_map = remap_env_ids(df, cfg.env_col)
    split_ids = read_split_ids(cfg.splits_dir, cfg.split_name)
    split_df = split_dataframe_by_ids(df, split_ids)

    gcfg = GraphBuildConfig(label_col=cfg.label_col, env_col=cfg.env_col, sample_weight_col=cfg.sample_weight_col)
    train_graphs = dataframe_to_graphs(split_df["train"], gcfg)
    val_graphs = dataframe_to_graphs(split_df["val"], gcfg)
    test_graphs = dataframe_to_graphs(split_df["test"], gcfg)

    if not train_graphs:
        raise ValueError("No training graphs were generated; cannot continue")
    feature_schema = {
        "node_feature_dim": int(train_graphs[0].x.shape[1]),
        "edge_feature_dim": int(train_graphs[0].edge_attr.shape[1]),
        "graph_builder": "data_graph.atom_features+bond_features",
    }
    (run_root / "artifacts" / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=cfg.batch_size, shuffle=False)

    train_env = torch.tensor([int(g.env.item()) for g in train_graphs], dtype=torch.long)
    env_class_weights = compute_env_class_weights(train_env, len(env_map))
    env_counts = torch.bincount(train_env, minlength=len(env_map)).tolist()
    env_balance = pd.DataFrame(
        {"env_id": list(range(len(env_map))), "count": env_counts, "weight": env_class_weights.tolist()}
    )
    env_balance.to_csv(run_root / "reports/env_balance.csv", index=False)

    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CausalQSARModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        z_dim=cfg.z_dim,
        z_inv_dim=cfg.z_inv_dim,
        z_spu_dim=cfg.z_spu_dim,
        n_envs=len(env_map),
        task=cfg.task,
        encoder=cfg.encoder,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    epochs_without_improvement = 0
    history = []
    schedule_rows = []
    irm_diag_rows = []
    dis_diag_rows = []
    jsonl = (run_root / "logs/metrics.jsonl").open("w")

    env_class_weights = env_class_weights.to(device)
    for epoch in range(1, cfg.epochs + 1):
        ramp = schedule_weight(epoch, cfg.warmup_epochs, cfg.ramp_epochs)
        lam_adv = cfg.lambda_adv * ramp
        lam_irm = cfg.lambda_irm * ramp
        lam_dis = cfg.lambda_dis * ramp
        schedule_rows.append({"epoch": epoch, "lambda_adv": lam_adv, "lambda_irm": lam_irm, "lambda_dis": lam_dis})

        model.train()
        losses = {"pred": 0.0, "adv": 0.0, "irm": 0.0, "dis": 0.0, "total": 0.0}
        metric_state = {"cosine": 0.0, "hsic": 0.0}
        n = 0

        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch, lambda_grl=lam_adv)
            y = batch.y.float()
            yhat = out["yhat"]
            sample_weight = batch.sample_weight.float() if hasattr(batch, "sample_weight") else None

            l_pred = prediction_loss(
                yhat,
                y,
                task=cfg.task,
                loss_pred=cfg.loss_pred,
                loss_cls=cfg.loss_cls,
                sample_weight=sample_weight,
            )
            l_adv = adversary_env_loss(out["envhat"], batch.env, loss_env=cfg.loss_env, env_class_weights=env_class_weights)

            if cfg.irm_mode == "irmv1" and lam_irm > 0:
                l_irm, irm_diags = irmv1_penalty(
                    yhat,
                    y,
                    batch.env,
                    task=cfg.task,
                    loss_pred=cfg.loss_pred,
                    loss_cls=cfg.loss_cls,
                    sample_weight=sample_weight,
                )
                for d in irm_diags:
                    irm_diag_rows.append({"epoch": epoch, "env_id": d.env_id, "risk_env": d.risk_env, "grad_norm_sq": d.grad_norm_sq})
            else:
                l_irm = torch.zeros((), device=device)

            l_dis, dis_metrics = disentanglement_loss(out["z_inv"], out["z_spu"], mode=cfg.disentangle)
            if cfg.disentangle != "none":
                dis_diag_rows.append({"epoch": epoch, "cosine_sim": dis_metrics["cosine_sim"], "hsic": dis_metrics["hsic"]})

            loss = l_pred + lam_adv * l_adv + lam_irm * l_irm + lam_dis * l_dis
            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = batch.num_graphs
            n += bs
            losses["pred"] += l_pred.item() * bs
            losses["adv"] += l_adv.item() * bs
            losses["irm"] += float(l_irm.item()) * bs
            losses["dis"] += l_dis.item() * bs
            losses["total"] += loss.item() * bs
            metric_state["cosine"] += dis_metrics["cosine_sim"] * bs
            metric_state["hsic"] += dis_metrics["hsic"] * bs

        train_pred_df, _ = evaluate(model, train_loader, device, cfg.task)
        val_pred_df, _ = evaluate(model, val_loader, device, cfg.task)
        if val_pred_df.empty:
            logging.warning("Validation split is empty for split_name=%s; using train metrics as fallback", cfg.split_name)
            if cfg.task == "regression":
                vmetric = regression_metrics(train_pred_df["y_true"], train_pred_df["y_pred"])
                score = vmetric["rmse"]
            else:
                vmetric = classification_metrics(train_pred_df["y_true"], train_pred_df["y_pred"])
                score = -float(vmetric.get("auc", 0.0) if not np.isnan(vmetric.get("auc", np.nan)) else 0.0)
        elif cfg.task == "regression":
            vmetric = regression_metrics(val_pred_df["y_true"], val_pred_df["y_pred"])
            score = vmetric["rmse"]
        else:
            vmetric = classification_metrics(val_pred_df["y_true"], val_pred_df["y_pred"])
            score = -float(vmetric.get("auc", 0.0) if not np.isnan(vmetric.get("auc", np.nan)) else 0.0)

        row = {
            "epoch": epoch,
            "L_pred": losses["pred"] / max(1, n),
            "L_env": losses["adv"] / max(1, n),
            "L_irm": losses["irm"] / max(1, n),
            "L_dis": losses["dis"] / max(1, n),
            "total": losses["total"] / max(1, n),
            "cosine_sim": metric_state["cosine"] / max(1, n),
            "hsic": metric_state["hsic"] / max(1, n),
            **{f"val_{k}": v for k, v in vmetric.items()},
        }
        history.append(row)
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()

        if score < best_val:
            best_val = score
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_root / "checkpoints/best.pt")
        else:
            epochs_without_improvement += 1
        torch.save(model.state_dict(), run_root / "checkpoints/last.pt")

        if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
            logging.info(
                "Early stopping triggered at epoch %s (patience=%s)",
                epoch,
                cfg.early_stopping_patience,
            )
            break

    jsonl.close()
    model.load_state_dict(torch.load(run_root / "checkpoints/best.pt", map_location=device, weights_only=True))

    hist = pd.DataFrame(history)
    hist[["epoch", "L_pred", "L_env", "L_irm", "L_dis", "total"]].to_csv(
        run_root / "reports/loss_breakdown.csv", index=False
    )
    pd.DataFrame(schedule_rows).to_csv(run_root / "reports/schedule.csv", index=False)
    if cfg.irm_mode == "irmv1":
        pd.DataFrame(irm_diag_rows, columns=["epoch", "env_id", "risk_env", "grad_norm_sq"]).to_csv(run_root / "reports/irm_diagnostics.csv", index=False)
    if cfg.disentangle != "none":
        pd.DataFrame(dis_diag_rows, columns=["epoch", "cosine_sim", "hsic"]).to_csv(run_root / "reports/disentanglement_diagnostics.csv", index=False)

    pred_dfs = {}
    z_splits = {}
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        pred_df, z = evaluate(model, loader, device, cfg.task)
        pred_dfs[split] = pred_df
        z_splits[split] = z
        pred_df.to_parquet(run_root / f"predictions/{split}_predictions.parquet", index=False)

    test_df = pred_dfs["test"]
    if test_df.empty:
        logging.warning("Test split is empty for split_name=%s; emitting NaN metrics", cfg.split_name)
        if cfg.task == "regression":
            ms = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan, "pearson": np.nan}
        else:
            ms = {"auc": np.nan, "pr_auc": np.nan, "f1": np.nan, "balanced_acc": np.nan}
    elif cfg.task == "regression":
        ms = regression_metrics(test_df["y_true"], test_df["y_pred"])
    else:
        ms = classification_metrics(test_df["y_true"], test_df["y_pred"])
    pd.DataFrame([{"split": "test", **ms}]).to_csv(run_root / "reports/metrics_summary.csv", index=False)

    per_env = per_environment_metrics(test_df, cfg.task, "env_id_manual", "y_true", "y_pred")
    per_env.to_csv(run_root / "reports/per_env_metrics.csv", index=False)

    env_adv_acc = accuracy_score(test_df["env_id_manual"], test_df["env_pred"]) if len(test_df) else np.nan
    z_probe = linear_probe_env_predictability(z_splits["test"], test_df["env_id_manual"].to_numpy()) if len(test_df) else np.nan
    if cfg.task == "regression" and "rmse" in per_env.columns:
        perf_var = float(per_env["rmse"].var())
    elif cfg.task == "classification" and "auc" in per_env.columns:
        perf_var = float(per_env["auc"].var())
    else:
        perf_var = float("nan")
    inv = pd.DataFrame(
        [
            {"metric": "adversary_accuracy", "value": env_adv_acc},
            {"metric": "linear_probe_env_predictability", "value": z_probe},
            {"metric": "performance_variance_across_env", "value": perf_var},
        ]
    )
    inv.to_csv(run_root / "reports/invariance_checks.csv", index=False)

    if test_df.empty:
        cal_df = pd.DataFrame()
    elif cfg.task == "classification":
        ece, cal_df = expected_calibration_error(test_df["y_true"], test_df["y_pred"])
        cal_df["ece"] = ece
    else:
        cal_df = regression_calibration(test_df["y_true"], test_df["y_pred"])
    cal_df.to_csv(run_root / "reports/calibration.csv", index=False)

    bbb_metrics = pd.DataFrame()
    merged_bbb = test_df.copy()
    if cfg.bbb_parquet:
        bbb = pd.read_parquet(cfg.bbb_parquet)
        bbb_id_col = _first_present(bbb, ["molecule_id", "molecule_chembl_id", "compound_id", "mol_id", "id"])
        if bbb_id_col is not None:
            bbb = bbb.copy()
            bbb["molecule_id"] = bbb[bbb_id_col].astype(str)
            bbb = bbb.drop_duplicates(subset=["molecule_id"])
            merged_bbb = merged_bbb.merge(bbb, on="molecule_id", how="left", suffixes=("", "_bbb"))

    strat_cols = [c for c in ["cns_like", "is_cns_like", "cns_mpo_bin"] if c in merged_bbb.columns]
    rows = []
    for c in strat_cols:
        for value, g in merged_bbb.groupby(c, dropna=True):
            if len(g) < 2:
                continue
            m = regression_metrics(g.y_true, g.y_pred) if cfg.task == "regression" else classification_metrics(g.y_true, g.y_pred)
            rows.append({"stratum": c, "value": value, **m, "n": len(g)})
    bbb_metrics = pd.DataFrame(rows)
    bbb_metrics.to_csv(run_root / "reports/bbb_metrics.csv", index=False)

    ablation = pd.DataFrame(
        [
            {
                "run_id": cfg.run_id,
                "split_name": cfg.split_name,
                "seed": cfg.seed,
                "encoder": cfg.encoder,
                "lambda_adv": cfg.lambda_adv,
                "lambda_irm": cfg.lambda_irm,
                "lambda_dis": cfg.lambda_dis,
                "loss_pred": cfg.loss_pred,
                "loss_cls": cfg.loss_cls,
                "loss_env": cfg.loss_env,
                "irm_mode": cfg.irm_mode,
                "disentangle": cfg.disentangle,
                **ms,
            }
        ]
    )
    ablation.to_csv(run_root / "reports/ablation_table.csv", index=False)

    # Step 6 figures
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in ["L_pred", "L_env", "L_irm", "L_dis", "total"]:
        ax.plot(hist["epoch"], hist[c], label=c)
    ax.legend()
    style_axis(ax, style, "Loss Components Over Time", "Epoch", "Loss")
    fig.tight_layout()
    fig.savefig(run_root / "figures/fig_loss_components_over_time.svg")

    fig, ax = plt.subplots(figsize=(8, 5))
    irm_curve = hist["L_irm"] if "L_irm" in hist.columns else np.zeros(len(hist))
    ax.plot(hist["epoch"], irm_curve, label="L_irm")
    ax.legend()
    style_axis(ax, style, "IRM Penalty Over Time", "Epoch", "IRM Penalty")
    fig.tight_layout()
    fig.savefig(run_root / "figures/fig_irm_penalty_over_time.svg")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["epoch"], hist["cosine_sim"], label="avg |cos(z_inv, z_spu)|")
    ax.plot(hist["epoch"], hist["hsic"], label="hsic")
    ax.legend()
    style_axis(ax, style, "Disentanglement Diagnostics", "Epoch", "Metric")
    fig.tight_layout()
    fig.savefig(run_root / "figures/fig_disentanglement_over_time.svg")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(env_balance["env_id"].astype(str), env_balance["weight"])
    style_axis(ax, style, "Environment Weights Distribution", "Environment", "Weight")
    fig.tight_layout()
    fig.savefig(run_root / "figures/fig_env_weights_distribution.svg")

    scripts = [
        "scripts/train_causal_qsar.py",
        "scripts/model_gnn.py",
        "scripts/data_graph.py",
        "scripts/metrics.py",
        "scripts/plot_style.py",
        "scripts/losses.py",
    ]
    prov = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_hash": git_commit(),
        "sha256_scripts": {s: sha256_file(Path(s)) for s in scripts if Path(s).exists()},
        "sha256_inputs": {
            "dataset_parquet": sha256_file(Path(cfg.dataset_parquet)),
            "train_ids": sha256_file(Path(cfg.splits_dir) / cfg.split_name / "train_ids.csv"),
            "val_ids": sha256_file(Path(cfg.splits_dir) / cfg.split_name / "val_ids.csv"),
            "test_ids": sha256_file(Path(cfg.splits_dir) / cfg.split_name / "test_ids.csv"),
        },
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "split_name": cfg.split_name,
        "model_hyperparams": {"encoder": cfg.encoder, "z_dim": cfg.z_dim, "z_inv_dim": cfg.z_inv_dim, "z_spu_dim": cfg.z_spu_dim},
        "loss_objective": {
            "L": "L_pred + lambda_adv * L_env + lambda_irm * L_irm + lambda_dis * L_dis",
            "task": cfg.task,
            "loss_pred": cfg.loss_pred,
            "loss_cls": cfg.loss_cls,
            "loss_env": cfg.loss_env,
            "irm_mode": cfg.irm_mode,
            "disentangle": cfg.disentangle,
            "warmup_epochs": cfg.warmup_epochs,
            "ramp_epochs": cfg.ramp_epochs,
            "lambda_targets": {"lambda_adv": cfg.lambda_adv, "lambda_irm": cfg.lambda_irm, "lambda_dis": cfg.lambda_dis},
            "env_class_weights": env_balance.set_index("env_id")["weight"].to_dict() if cfg.loss_env == "weighted_ce" else None,
        },
        "number_of_envs": len(env_map),
        "class_balance": float(df[cfg.label_col].mean()) if cfg.task == "classification" else None,
        "cns_stats": bbb_metrics.groupby("stratum").size().to_dict() if not bbb_metrics.empty else None,
    }
    (run_root / "provenance/provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (run_root / "provenance/run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    freeze = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True)
    (run_root / "provenance/environment.txt").write_text(freeze.stdout + freeze.stderr, encoding="utf-8")


if __name__ == "__main__":
    main()
