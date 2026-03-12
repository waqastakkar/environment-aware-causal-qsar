#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import DataLoader

from data_graph import GraphBuildConfig, dataframe_to_graphs
from metrics import expected_calibration_error
from model_gnn import CausalQSARModel
from plot_style import PlotStyle, configure_matplotlib, parse_palette, style_axis


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09 external active/inactive inhibition evaluation: pIC50 regression predictions -> binary calls"
    )
    p.add_argument("--target", default="CHEMBL335")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--external_parquet", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--primary_threshold", type=float, default=5.0)
    p.add_argument("--pic50_thresholds", type=float, nargs="+", default=[5.0, 5.5, 6.0])
    p.add_argument("--inhibition_active_threshold", type=float, default=50.0)
    p.add_argument("--enable_calibration", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--bbb_parquet", default=None)
    p.add_argument("--svg", action="store_true", default=True)
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--bold_text", action="store_true", default=True)
    p.add_argument("--palette", default="nature5")
    p.add_argument("--font_title", type=int, default=16)
    p.add_argument("--font_label", type=int, default=14)
    p.add_argument("--font_tick", type=int, default=12)
    p.add_argument("--font_legend", type=int, default=12)
    return p.parse_args()


def _load_training_cfg(run_dir: Path) -> dict:
    cfg = run_dir / "configs" / "resolved_config.yaml"
    if cfg.exists():
        return yaml.safe_load(cfg.read_text(encoding="utf-8"))
    return {}


def _infer_model_dims(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    return {
        "z_inv_dim": int(state_dict["f_inv.0.weight"].shape[0]),
        "z_dim": int(state_dict["f_inv.0.weight"].shape[1]),
        "z_spu_dim": int(state_dict["f_spu.0.weight"].shape[0]),
        "n_envs": int(state_dict["adversary.2.weight"].shape[0]),
    }


def _current_feature_dims(df: pd.DataFrame) -> tuple[int, int]:
    work = df.copy()
    if "molecule_id" not in work.columns:
        work["molecule_id"] = [f"ext_{i}" for i in range(len(work))]
    if "env_id_manual" not in work.columns:
        work["env_id_manual"] = 0
    if "y_dummy" not in work.columns:
        work["y_dummy"] = 0.0

    gcfg = GraphBuildConfig(
        smiles_col="smiles_canonical" if "smiles_canonical" in work.columns else "smiles",
        id_col="molecule_id",
        label_col="y_dummy",
        env_col="env_id_manual",
    )
    graphs = dataframe_to_graphs(work, gcfg)
    if not graphs:
        raise ValueError("No molecules could be featurized from external_parquet; cannot validate feature schema")
    node_dim = int(graphs[0].x.shape[1])
    edge_dim = int(graphs[0].edge_attr.shape[1])
    return node_dim, edge_dim


def _validate_checkpoint_schema(run_dir: Path, external: pd.DataFrame) -> tuple[int, int]:
    schema_path = run_dir / "artifacts" / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Missing required feature schema at {schema_path}. Re-run Step 06 with the current code."
        )
    saved_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if "node_feature_dim" not in saved_schema or "edge_feature_dim" not in saved_schema:
        raise ValueError(
            f"Invalid feature schema at {schema_path}: expected node_feature_dim and edge_feature_dim. "
            "Re-run Step 06 with the current code."
        )
    saved_node = int(saved_schema["node_feature_dim"])
    saved_edge = int(saved_schema["edge_feature_dim"])
    current_node, current_edge = _current_feature_dims(external)
    if saved_node != current_node or saved_edge != current_edge:
        raise ValueError(
            "Model schema mismatch.\n"
            f"Expected node_dim: {saved_node}\n"
            f"Current node_dim: {current_node}\n"
            f"Expected edge_dim: {saved_edge}\n"
            f"Current edge_dim: {current_edge}\n\n"
            "You must rerun Step 06 after featurization change."
        )
    return current_node, current_edge


def _ensure_activity_labels(df: pd.DataFrame, inhibition_active_threshold: float) -> pd.DataFrame:
    out = df.copy()

    inhibition_col = None
    for candidate in ["inhibition_percent", "inhibition", "standard_value"]:
        if candidate in out.columns:
            inhibition_col = candidate
            break
    if inhibition_col is None:
        raise ValueError(
            "external_parquet must include an inhibition measurement column to define true active/inactive labels. "
            "Expected one of: inhibition_percent, inhibition, standard_value."
        )

    inhibition = pd.to_numeric(out[inhibition_col], errors="coerce")
    if inhibition.isna().any():
        raise ValueError(
            f"external_parquet contains non-numeric values in {inhibition_col}; cannot derive true_active labels"
        )
    out["inhibition_percent"] = inhibition

    out["y_inhib_active"] = (inhibition >= inhibition_active_threshold).astype(int)
    return out


def _predict_pic50(model, df: pd.DataFrame, batch_size: int = 128) -> pd.DataFrame:
    work = df.copy()
    if "molecule_id" not in work.columns:
        work["molecule_id"] = [f"ext_{i}" for i in range(len(work))]
    if "env_id_manual" not in work.columns:
        work["env_id_manual"] = 0
    if "y_dummy" not in work.columns:
        work["y_dummy"] = 0.0

    gcfg = GraphBuildConfig(smiles_col="smiles_canonical" if "smiles_canonical" in work.columns else "smiles", id_col="molecule_id", label_col="y_dummy", env_col="env_id_manual")
    graphs = dataframe_to_graphs(work, gcfg)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            for i in range(batch.num_graphs):
                rows.append({"molecule_id": batch.molecule_id[i], "pIC50_hat": float(out["yhat"][i].cpu())})
    return pd.DataFrame(rows)


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _save_environment_txt(path: Path) -> None:
    path.write_text(
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}\npython={platform.python_version()}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    for sub in ["predictions", "reports", "figures", "provenance"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    style = PlotStyle(
        font_family=args.font,
        font_title=args.font_title,
        font_label=args.font_label,
        font_tick=args.font_tick,
        font_legend=args.font_legend,
        bold_text=True,
        palette=parse_palette(args.palette),
    )
    configure_matplotlib(style, svg=True)

    external = pd.read_parquet(args.external_parquet)
    external = _ensure_activity_labels(external, args.inhibition_active_threshold)

    node_dim, edge_dim = _validate_checkpoint_schema(run_dir, external)

    ckpt_path = run_dir / args.checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    dims = _infer_model_dims(state)

    cfg = _load_training_cfg(run_dir)
    model = CausalQSARModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        z_dim=dims["z_dim"],
        z_inv_dim=dims["z_inv_dim"],
        z_spu_dim=dims["z_spu_dim"],
        n_envs=dims["n_envs"],
        task="regression",
        encoder=cfg.get("encoder", "gine"),
    )
    model.load_state_dict(state)

    preds = _predict_pic50(model, external)
    preds.to_parquet(outdir / "predictions" / "external_predictions.parquet", index=False)

    join = external.copy()
    if "molecule_id" not in join.columns:
        join["molecule_id"] = [f"ext_{i}" for i in range(len(join))]
    scored = join.merge(preds, on="molecule_id", how="left")
    scored["y_true"] = scored["y_inhib_active"].astype(int)
    scored.to_parquet(outdir / "predictions" / "external_scored.parquet", index=False)

    y_true = scored["y_true"].to_numpy()
    y_score = scored["pIC50_hat"].to_numpy()
    ce_metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
    }
    ce_metrics.update(_binary_metrics(y_true, y_score, args.primary_threshold))
    main_cols = ["roc_auc", "pr_auc", "threshold", "accuracy", "balanced_accuracy", "f1", "tn", "fp", "fn", "tp"]
    pd.DataFrame([ce_metrics])[main_cols].to_csv(outdir / "reports" / "cross_endpoint_metrics.csv", index=False)

    sens = pd.DataFrame([_binary_metrics(y_true, y_score, t) for t in args.pic50_thresholds])
    sens.to_csv(outdir / "reports" / "threshold_sensitivity.csv", index=False)

    dist_df = pd.DataFrame(
        {
            "metric": ["pIC50_hat_min", "pIC50_hat_median", "pIC50_hat_max", "inhibition_min", "inhibition_median", "inhibition_max"],
            "value": [
                float(np.nanmin(y_score)),
                float(np.nanmedian(y_score)),
                float(np.nanmax(y_score)),
                float(np.nanmin(scored.get("inhibition_percent", pd.Series(np.nan, index=scored.index)))),
                float(np.nanmedian(scored.get("inhibition_percent", pd.Series(np.nan, index=scored.index)))),
                float(np.nanmax(scored.get("inhibition_percent", pd.Series(np.nan, index=scored.index)))),
            ],
        }
    )
    dist_df.to_csv(outdir / "reports" / "external_distribution.csv", index=False)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC={ce_metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.legend()
        style_axis(
            ax,
            style,
            "Step 09 external active/inactive inhibition evaluation (ROC)",
            "False positive rate",
            "True positive rate",
        )
        fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_roc_external.svg"); plt.close(fig)

        pre, rec, _ = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, pre, label=f"PR-AUC={ce_metrics['pr_auc']:.3f}")
        ax.legend()
        style_axis(
            ax,
            style,
            "Step 09 external active/inactive inhibition evaluation (PR)",
            "Recall",
            "Precision",
        )
        fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_pr_external.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_score, bins=30, alpha=0.65, label="Predicted pIC50")
    if "inhibition_percent" in scored.columns:
        ax.hist(scored["inhibition_percent"].to_numpy() / 10.0, bins=30, alpha=0.5, label="Inhibition % / 10")
    ax.legend()
    style_axis(ax, style, "External active/inactive inhibition score distribution", "Value", "Count")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_external_score_distribution.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sens["threshold"], sens["f1"], marker="o", label="F1")
    ax.plot(sens["threshold"], sens["balanced_accuracy"], marker="o", label="Balanced Acc")
    ax.legend()
    style_axis(ax, style, "External active/inactive inhibition threshold sensitivity", "pIC50 threshold", "Metric")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_threshold_curve.svg"); plt.close(fig)

    if args.enable_calibration:
        val_path = run_dir / "predictions" / "val_predictions.parquet"
        if val_path.exists():
            val = pd.read_parquet(val_path)
            yv = val["y_true"].to_numpy()
            sv = val["y_pred"].to_numpy()
            yv_bin = (yv >= args.primary_threshold).astype(int)

            platt = LogisticRegression(max_iter=1000)
            platt.fit(sv.reshape(-1, 1), yv_bin)
            prob_ext = platt.predict_proba(y_score.reshape(-1, 1))[:, 1]

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(sv, yv_bin)
            prob_ext_iso = iso.predict(y_score)

            ece_platt, _ = expected_calibration_error(y_true, prob_ext)
            ece_iso, _ = expected_calibration_error(y_true, prob_ext_iso)
            cal_df = pd.DataFrame(
                [
                    {"method": "platt", "ece": float(ece_platt), "brier": float(np.mean((prob_ext - y_true) ** 2))},
                    {"method": "isotonic", "ece": float(ece_iso), "brier": float(np.mean((prob_ext_iso - y_true) ** 2))},
                ]
            )
            cal_df.to_csv(outdir / "reports" / "calibration_external.csv", index=False)

            frac, mean_pred = calibration_curve(y_true, prob_ext, n_bins=10, strategy="uniform")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.plot(mean_pred, frac, marker="o", label="Platt")
            frac2, mean_pred2 = calibration_curve(y_true, prob_ext_iso, n_bins=10, strategy="uniform")
            ax.plot(mean_pred2, frac2, marker="s", label="Isotonic")
            ax.legend()
            style_axis(
                ax,
                style,
                "External active/inactive inhibition calibration",
                "Predicted probability",
                "Observed fraction",
            )
            fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_calibration_external.svg"); plt.close(fig)

            scored["prob_active_platt"] = prob_ext
            scored["prob_active_isotonic"] = prob_ext_iso
            scored.to_parquet(outdir / "predictions" / "external_scored.parquet", index=False)
        else:
            pd.DataFrame([{"note": "val_predictions.parquet not found; calibration skipped"}]).to_csv(
                outdir / "reports" / "calibration_external.csv", index=False
            )

    if args.bbb_parquet:
        bbb = pd.read_parquet(args.bbb_parquet)
        key = "molecule_id" if "molecule_id" in bbb.columns and "molecule_id" in scored.columns else "inchikey"
        if key in bbb.columns and key in scored.columns:
            col = "is_cns" if "is_cns" in bbb.columns else ("bbb_label" if "bbb_label" in bbb.columns else None)
            if col is not None:
                merged = scored.merge(bbb[[key, col]], on=key, how="left")
                rows = []
                for grp, g in merged.groupby(merged[col].fillna("unknown")):
                    yt = g["y_true"].to_numpy()
                    ys = g["pIC50_hat"].to_numpy()
                    r = _binary_metrics(yt, ys, args.primary_threshold)
                    r["group"] = grp
                    r["roc_auc"] = float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else np.nan
                    r["pr_auc"] = float(average_precision_score(yt, ys)) if len(np.unique(yt)) > 1 else np.nan
                    rows.append(r)
                pd.DataFrame(rows).to_csv(outdir / "reports" / "cns_stratified_metrics.csv", index=False)

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = pd.DataFrame(rows)
                ax.bar(bars["group"].astype(str), bars["balanced_accuracy"])
                style_axis(
                    ax,
                    style,
                    "External active/inactive inhibition CNS stratified balanced accuracy",
                    "Group",
                    "Balanced accuracy",
                )
                fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_cns_stratified_external.svg"); plt.close(fig)

    pd.DataFrame(columns=["note"]).to_csv(outdir / "reports" / "cf_consistency_external.csv", index=False)

    run_cfg = vars(args)
    (outdir / "provenance" / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": run_cfg,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "script_sha256": sha256_file(Path(__file__)),
        "input_hashes": {
            "external_parquet": sha256_file(Path(args.external_parquet)),
            "checkpoint": sha256_file(ckpt_path),
        },
        "counts": {"n_external": int(len(scored)), "n_positive": int(scored["y_true"].sum())},
    }
    (outdir / "provenance" / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    _save_environment_txt(outdir / "provenance" / "environment.txt")


if __name__ == "__main__":
    main()
