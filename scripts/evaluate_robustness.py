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
import yaml

from applicability_domain import binned_relationship, embedding_ad, fingerprint_ad
from conformal import apply_conformal, split_conformal_q, summarize_coverage
from embedding_extract import extract_zinv_embeddings
from ensemble_utils import ensemble_for_group, selective_prediction_curve
from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args
from stability_analysis import ablation_summary, paired_bootstrap_tests, seed_metrics


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def discover_runs(runs_root: Path) -> pd.DataFrame:
    rows = []
    for test_pred in sorted(runs_root.glob("**/predictions/test_predictions.parquet")):
        run_dir = test_pred.parents[1]
        ckpt = run_dir / "checkpoints" / "best.pt"
        if not ckpt.exists():
            continue
        cfg_path = run_dir / "configs" / "resolved_config.yaml"
        cfg = {}
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        folder = run_dir.as_posix().split("/")[-3:]
        rows.append(
            {
                "run_dir": str(run_dir),
                "split_name": cfg.get("split_name", folder[-2] if len(folder) >= 2 else "unknown"),
                "run_id": run_dir.name,
                "seed": cfg.get("seed", _parse_seed(run_dir.name)),
                "ablation": cfg.get("ablation", _parse_ablation(run_dir.name)),
                "task": cfg.get("task", "regression"),
                "label_col": cfg.get("label_col", "pIC50"),
                "env_col": cfg.get("env_col", "env_id_manual"),
                "checkpoint_path": str(ckpt),
                "train_pred_path": str(run_dir / "predictions" / "train_predictions.parquet"),
                "val_pred_path": str(run_dir / "predictions" / "val_predictions.parquet"),
                "test_pred_path": str(test_pred),
            }
        )
    return pd.DataFrame(rows)


def _parse_seed(name: str):
    import re

    m = re.search(r"seed[_-]?(\d+)", name)
    return int(m.group(1)) if m else np.nan


def _parse_ablation(name: str):
    for a in ["base", "adv", "adv_irm", "adv_dis", "full", "cf"]:
        if a in name:
            return a
    return "unknown"


def ensure_dirs(outdir: Path):
    for s in ["manifests", "ensemble", "conformal", "applicability_domain", "stability", "figures", "figure_data", "provenance"]:
        (outdir / s).mkdir(parents=True, exist_ok=True)


def save_empty_csv(path: Path, cols: list[str]) -> None:
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description="Step 11 Robustness & Uncertainty Suite")
    p.add_argument("--target", required=True)
    p.add_argument("--runs_root", required=True)
    p.add_argument("--dataset_parquet", required=True)
    p.add_argument("--bbb_parquet")
    p.add_argument("--external_scored_parquet")
    p.add_argument("--outdir", required=True)
    p.add_argument("--task", default="regression")
    p.add_argument("--label_col", default="pIC50")
    p.add_argument("--env_col", default="env_id_manual")
    p.add_argument("--group_by", nargs="+", default=["split_name", "ablation"])
    p.add_argument("--ensemble_size", type=int, default=5)
    p.add_argument("--conformal_coverage", type=float, default=0.90)
    p.add_argument("--ad_fingerprint", default="morgan")
    p.add_argument("--ad_radius", type=int, default=2)
    p.add_argument("--ad_nbits", type=int, default=2048)
    p.add_argument("--ad_embedding", default="z_inv")
    p.add_argument("--ad_k", type=int, default=1)
    add_plot_style_args(p)
    args = p.parse_args()

    outdir = Path(args.outdir)
    ensure_dirs(outdir)
    style = style_from_args(args)
    configure_matplotlib(style, svg=True)

    runs = discover_runs(Path(args.runs_root))
    runs.to_csv(outdir / "manifests" / "runs_index.csv", index=False)
    groups = runs.groupby(args.group_by, dropna=False).size().reset_index(name="n_runs") if not runs.empty else pd.DataFrame(columns=args.group_by + ["n_runs"])
    groups["eligible_for_ensemble"] = groups["n_runs"] >= args.ensemble_size
    groups.to_csv(outdir / "manifests" / "groups_index.csv", index=False)

    ens_train, ens_val, ens_test = [], [], []
    used_groups = []
    for gk, gdf in runs.groupby(args.group_by, dropna=False):
        if len(gdf) < args.ensemble_size:
            continue
        key = gk if isinstance(gk, tuple) else (gk,)
        tag = {k: v for k, v in zip(args.group_by, key)}
        tr = ensemble_for_group(gdf, "train")
        va = ensemble_for_group(gdf, "val")
        te = ensemble_for_group(gdf, "test")
        for d in [tr, va, te]:
            for k, v in tag.items():
                d[k] = v
        tr["split"] = "train"; va["split"] = "val"; te["split"] = "test"
        ens_train.append(tr); ens_val.append(va); ens_test.append(te)
        used_groups.append({**tag, "n_members": int(len(gdf))})

    ens_train_df = pd.concat(ens_train, ignore_index=True) if ens_train else pd.DataFrame()
    ens_val_df = pd.concat(ens_val, ignore_index=True) if ens_val else pd.DataFrame()
    ens_test_df = pd.concat(ens_test, ignore_index=True) if ens_test else pd.DataFrame()
    ens_train_df.to_parquet(outdir / "ensemble" / "ensemble_predictions_train.parquet", index=False)
    ens_val_df.to_parquet(outdir / "ensemble" / "ensemble_predictions_val.parquet", index=False)
    ens_test_df.to_parquet(outdir / "ensemble" / "ensemble_predictions_test.parquet", index=False)
    (outdir / "manifests" / "ensemble_manifest.json").write_text(json.dumps({"groups": used_groups}, indent=2), encoding="utf-8")

    unc = pd.concat([ens_train_df, ens_val_df, ens_test_df], ignore_index=True) if (not ens_train_df.empty or not ens_val_df.empty or not ens_test_df.empty) else pd.DataFrame(columns=["y", "yhat_mean", "yhat_std", "abs_error", "split"])
    unc[[c for c in ["y", "yhat_mean", "yhat_std", "abs_error", "split", args.env_col] if c in unc.columns]].to_csv(outdir / "ensemble" / "ensemble_uncertainty.csv", index=False)
    sel = selective_prediction_curve(ens_test_df) if not ens_test_df.empty else pd.DataFrame(columns=["coverage", "rmse", "mae", "n"])
    sel.to_csv(outdir / "ensemble" / "selective_prediction.csv", index=False)

    # conformal
    q, cal = split_conformal_q(ens_val_df, coverage=args.conformal_coverage) if not ens_val_df.empty else (np.nan, pd.DataFrame(columns=["n_val", "coverage_target", "q"]))
    cal.to_csv(outdir / "conformal" / "conformal_calibration.csv", index=False)
    intervals = apply_conformal(ens_test_df, q) if not ens_test_df.empty and np.isfinite(q) else pd.DataFrame(columns=list(ens_test_df.columns) + ["interval_lower", "interval_upper", "interval_width", "covered"])
    intervals.to_parquet(outdir / "conformal" / "conformal_intervals_test.parquet", index=False)
    cov = summarize_coverage(intervals)
    cov.to_csv(outdir / "conformal" / "conformal_coverage.csv", index=False)
    width = cov[[c for c in cov.columns if c in ["split_name", "ablation", "mean_interval_width", "n"]]] if not cov.empty else pd.DataFrame(columns=["split_name", "ablation", "mean_interval_width", "n"])
    width.to_csv(outdir / "conformal" / "interval_width_by_split.csv", index=False)

    # AD
    data = pd.read_parquet(args.dataset_parquet)
    smiles_col = "smiles_canonical" if "smiles_canonical" in data.columns else "smiles"
    if smiles_col != "smiles":
        data = data.rename(columns={smiles_col: "smiles"})
    train_ids = ens_train_df["molecule_id"].dropna().unique().tolist() if "molecule_id" in ens_train_df.columns else []
    test_ids = ens_test_df["molecule_id"].dropna().unique().tolist() if "molecule_id" in ens_test_df.columns else []
    train_m = data[data["molecule_id"].isin(train_ids)][["molecule_id", "smiles"]].drop_duplicates() if "molecule_id" in data.columns else pd.DataFrame(columns=["molecule_id", "smiles"])
    test_m = data[data["molecule_id"].isin(test_ids)][["molecule_id", "smiles"]].drop_duplicates() if "molecule_id" in data.columns else pd.DataFrame(columns=["molecule_id", "smiles"])
    ad_fp = fingerprint_ad(train_m, test_m, radius=args.ad_radius, nbits=args.ad_nbits)
    ad_fp.to_parquet(outdir / "applicability_domain" / "ad_fingerprint.parquet", index=False)

    run0 = Path(runs.iloc[0]["run_dir"]) if not runs.empty else None
    ztr = extract_zinv_embeddings(run0, data, train_ids, label_col=args.label_col, env_col=args.env_col) if run0 else pd.DataFrame(columns=["molecule_id"])
    zte = extract_zinv_embeddings(run0, data, test_ids, label_col=args.label_col, env_col=args.env_col) if run0 else pd.DataFrame(columns=["molecule_id"])
    ad_emb = embedding_ad(ztr, zte, k=args.ad_k)
    ad_emb.to_parquet(outdir / "applicability_domain" / "ad_embedding.parquet", index=False)

    test_eval = ens_test_df.merge(ad_fp, on="molecule_id", how="left").merge(ad_emb, on="molecule_id", how="left") if not ens_test_df.empty and "molecule_id" in ens_test_df.columns else pd.DataFrame()
    e_fp = binned_relationship(test_eval, "ad_distance_fingerprint", "abs_error", out_x="ad_bin") if not test_eval.empty else pd.DataFrame(columns=["ad_bin", "y_mean", "n", "x_mean"])
    e_fp.to_csv(outdir / "applicability_domain" / "error_vs_ad.csv", index=False)
    u_fp = binned_relationship(test_eval, "ad_distance_fingerprint", "yhat_std", out_x="ad_bin") if not test_eval.empty else pd.DataFrame(columns=["ad_bin", "y_mean", "n", "x_mean"])
    u_fp.to_csv(outdir / "applicability_domain" / "uncertainty_vs_ad.csv", index=False)

    # stability
    seed_df = seed_metrics(ens_test_df, runs)
    seed_df.to_csv(outdir / "stability" / "seed_stability_metrics.csv", index=False)
    abl = ablation_summary(seed_df)
    abl.to_csv(outdir / "stability" / "ablation_stability_table.csv", index=False)
    stats = paired_bootstrap_tests(seed_df)
    stats.to_csv(outdir / "stability" / "statistical_tests.csv", index=False)

    # figures + figure data
    def savefig_data(fig_name: str, data_name: str, df: pd.DataFrame):
        df.to_csv(outdir / "figure_data" / data_name, index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        return fig, ax

    d1 = unc[["yhat_std", "abs_error"]].dropna() if not unc.empty and {"yhat_std", "abs_error"}.issubset(unc.columns) else pd.DataFrame(columns=["yhat_std", "abs_error"])
    d1.to_csv(outdir / "figure_data" / "ensemble_uncertainty_vs_error.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not d1.empty: ax.scatter(d1["yhat_std"], d1["abs_error"], s=10, alpha=0.5)
    style_axis(ax, style, "Ensemble uncertainty vs error", "yhat_std", "|error|")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_ensemble_uncertainty_vs_error.svg"); plt.close(fig)

    sel.to_csv(outdir / "figure_data" / "selective_prediction_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not sel.empty: ax.plot(sel["coverage"], sel["rmse"], marker="o", label="RMSE"); ax.plot(sel["coverage"], sel["mae"], marker="o", label="MAE"); ax.legend()
    style_axis(ax, style, "Selective prediction", "Coverage", "Error")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_selective_prediction.svg"); plt.close(fig)

    cov.to_csv(outdir / "figure_data" / "conformal_coverage_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not cov.empty: ax.bar(np.arange(len(cov)), cov["empirical_coverage"])
    style_axis(ax, style, "Conformal coverage", "Group", "Coverage")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_conformal_coverage.svg"); plt.close(fig)

    width.to_csv(outdir / "figure_data" / "interval_width_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not width.empty: ax.bar(np.arange(len(width)), width["mean_interval_width"])
    style_axis(ax, style, "Interval width by split", "Group", "Width")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_interval_width_by_split.svg"); plt.close(fig)

    e_fp.to_csv(outdir / "figure_data" / "error_vs_ad_fingerprint_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not e_fp.empty: ax.plot(e_fp["x_mean"], e_fp["y_mean"], marker="o")
    style_axis(ax, style, "Error vs AD (fingerprint)", "AD distance", "Mean |error|")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_error_vs_ad_fingerprint.svg"); plt.close(fig)

    e_emb = binned_relationship(test_eval, "ad_distance_embedding", "abs_error", out_x="ad_bin") if not test_eval.empty else pd.DataFrame(columns=["x_mean", "y_mean"])
    e_emb.to_csv(outdir / "figure_data" / "error_vs_ad_embedding_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not e_emb.empty: ax.plot(e_emb["x_mean"], e_emb["y_mean"], marker="o")
    style_axis(ax, style, "Error vs AD (embedding)", "AD distance", "Mean |error|")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_error_vs_ad_embedding.svg"); plt.close(fig)

    u_fp.to_csv(outdir / "figure_data" / "uncertainty_vs_ad_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not u_fp.empty: ax.plot(u_fp["x_mean"], u_fp["y_mean"], marker="o")
    style_axis(ax, style, "Uncertainty vs AD", "AD distance", "Mean yhat_std")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_uncertainty_vs_ad.svg"); plt.close(fig)

    seed_df.to_csv(outdir / "figure_data" / "seed_stability_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not seed_df.empty:
        for k, g in seed_df.groupby("ablation"):
            ax.scatter([str(k)] * len(g), g["rmse"], label=str(k))
    style_axis(ax, style, "Seed stability", "Ablation", "RMSE")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_seed_stability.svg"); plt.close(fig)

    abl.to_csv(outdir / "figure_data" / "ablation_robustness_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not abl.empty: ax.bar(abl["ablation"].astype(str), abl["rmse_mean"])
    style_axis(ax, style, "Ablation robustness", "Ablation", "RMSE mean")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_ablation_robustness.svg"); plt.close(fig)

    # optional external coverage
    if args.external_scored_parquet and Path(args.external_scored_parquet).exists() and np.isfinite(q):
        ext = pd.read_parquet(args.external_scored_parquet)
        if {"y_true", "pIC50_hat"}.issubset(ext.columns):
            ext = ext.rename(columns={"pIC50_hat": "yhat_mean", "y_true": "y"})
            ext_int = apply_conformal(ext, q)
            ext_cov = summarize_coverage(ext_int, group_cols=[])
            ext_cov["dataset"] = "external"
            cov = pd.concat([cov, ext_cov], ignore_index=True)
            cov.to_csv(outdir / "conformal" / "conformal_coverage.csv", index=False)

    # provenance
    rcfg = vars(args).copy()
    (outdir / "provenance" / "run_config.json").write_text(json.dumps(rcfg, indent=2), encoding="utf-8")
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": rcfg,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "script_hashes": {s: sha256_file(Path(__file__).parent / s) for s in ["evaluate_robustness.py", "ensemble_utils.py", "conformal.py", "applicability_domain.py", "embedding_extract.py", "stability_analysis.py", "plot_style.py"]},
        "runs_index_hash": sha256_file(outdir / "manifests" / "runs_index.csv"),
        "dataset_hash": sha256_file(Path(args.dataset_parquet)),
        "ensemble_size": args.ensemble_size,
        "conformal_coverage": args.conformal_coverage,
        "ad_params": {"fingerprint": args.ad_fingerprint, "radius": args.ad_radius, "nbits": args.ad_nbits, "embedding": args.ad_embedding, "k": args.ad_k},
    }
    (outdir / "provenance" / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    env = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    (outdir / "provenance" / "environment.txt").write_text(env, encoding="utf-8")


if __name__ == "__main__":
    main()
