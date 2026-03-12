#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_style import NATURE5, PlotStyle, configure_matplotlib, style_axis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latent environment discovery by unsupervised clustering.")
    p.add_argument("--input_compound_parquet", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--features", nargs="+", default=["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings"])
    p.add_argument("--method", choices=["kmeans", "gmm"], default="kmeans")
    p.add_argument("--k_min", type=int, default=3)
    p.add_argument("--k_max", type=int, default=12)
    p.add_argument("--select_by", choices=["silhouette"], default="silhouette")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--svg", action="store_true")
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--bold_text", action="store_true")
    p.add_argument("--palette", default="nature5")
    p.add_argument("--font_title", type=int, default=16)
    p.add_argument("--font_label", type=int, default=14)
    p.add_argument("--font_tick", type=int, default=12)
    p.add_argument("--font_legend", type=int, default=12)
    return p.parse_args()


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


def pick(df: pd.DataFrame, options: list[str]) -> str | None:
    m = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in m:
            return m[o.lower()]
    return None


def write_env(path: Path) -> None:
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.outdir)
    data_dir = out / "data"
    reports_dir = out / "reports"
    figs_dir = out / "figures"
    prov_dir = out / "provenance"
    for d in [data_dir, reports_dir, figs_dir, prov_dir]:
        d.mkdir(parents=True, exist_ok=True)

    style = PlotStyle(args.font, args.font_title, args.font_label, args.font_tick, args.font_legend, tuple(NATURE5))
    configure_matplotlib(style)

    comp = pd.read_parquet(args.input_compound_parquet)
    molecule_col = pick(comp, ["molecule_chembl_id", "compound_id", "molecule_id", "mol_id"]) or "molecule_id"
    if molecule_col not in comp.columns:
        comp[molecule_col] = comp.index.astype(str)

    features = [f for f in args.features if f in comp.columns]
    if not features:
        raise ValueError("None of requested features found in input parquet.")

    X_raw = comp[features].copy()
    valid_idx = X_raw.dropna().index
    X_raw = X_raw.loc[valid_idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw.values)

    fm = pd.DataFrame(X, columns=[f"z_{c}" for c in features], index=valid_idx)
    fm.insert(0, molecule_col, comp.loc[valid_idx, molecule_col].values)
    fm.to_parquet(data_dir / "learned_env_feature_matrix.parquet", index=False)

    scaler_json = {
        "features": features,
        "mean": {f: float(m) for f, m in zip(features, scaler.mean_)},
        "scale": {f: float(s) for f, s in zip(features, scaler.scale_)},
    }
    (data_dir / "learned_env_scaler.json").write_text(json.dumps(scaler_json, indent=2), encoding="utf-8")

    stability_rows = []
    best = {"k": None, "silhouette": -np.inf, "labels": None, "model": None, "proba": None}
    for k in range(args.k_min, args.k_max + 1):
        labels_runs = []
        inertias = []
        sils = []
        for seed in [args.random_seed, args.random_seed + 1, args.random_seed + 2]:
            if args.method == "kmeans":
                model = KMeans(n_clusters=k, n_init=20, random_state=seed)
                labels = model.fit_predict(X)
                inertias.append(float(model.inertia_))
                proba = None
            else:
                model = GaussianMixture(n_components=k, random_state=seed)
                labels = model.fit_predict(X)
                inertias.append(np.nan)
                proba = model.predict_proba(X).max(axis=1)
            sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else np.nan
            sils.append(sil)
            labels_runs.append(labels)
            if seed == args.random_seed and sil > best["silhouette"]:
                best = {"k": k, "silhouette": sil, "labels": labels, "model": model, "proba": proba}

        ari_between_runs = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                ari_between_runs.append(adjusted_rand_score(labels_runs[i], labels_runs[j]))
        stability_rows.append({
            "k": k,
            "silhouette_mean": float(np.nanmean(sils)),
            "silhouette_std": float(np.nanstd(sils)),
            "inertia_mean": float(np.nanmean(inertias)),
            "stability_ari_mean": float(np.mean(ari_between_runs)) if ari_between_runs else np.nan,
        })

    stability = pd.DataFrame(stability_rows)
    best_k = int(stability.sort_values("silhouette_mean", ascending=False).iloc[0]["k"])
    if best["k"] != best_k:
        if args.method == "kmeans":
            model = KMeans(n_clusters=best_k, n_init=30, random_state=args.random_seed)
            labels = model.fit_predict(X)
            proba = None
        else:
            model = GaussianMixture(n_components=best_k, random_state=args.random_seed)
            labels = model.fit_predict(X)
            proba = model.predict_proba(X).max(axis=1)
        best = {"k": best_k, "silhouette": float(silhouette_score(X, labels)), "labels": labels, "model": model, "proba": proba}

    stability.to_csv(reports_dir / "clustering_stability.csv", index=False)

    assignments = pd.DataFrame({
        molecule_col: comp.loc[valid_idx, molecule_col].values,
        "learned_env_id": best["labels"],
    })
    if best["proba"] is not None:
        assignments["learned_env_prob"] = best["proba"]
    assignments.to_csv(data_dir / "learned_env_assignments.csv", index=False)

    comp["learned_env_id"] = np.nan
    comp.loc[valid_idx, "learned_env_id"] = best["labels"]
    comp.to_parquet(data_dir / "multienv_compound_level.parquet", index=False)

    manual_col = "env_id" if "env_id" in comp.columns else None
    align_rows = []
    if manual_col:
        sub = comp.loc[valid_idx].dropna(subset=[manual_col]).copy()
        if not sub.empty:
            y_true = sub[manual_col].astype(str)
            y_pred = sub["learned_env_id"].astype(int)
            align_rows.append({"component": "env_id", "ARI": adjusted_rand_score(y_true, y_pred), "NMI": normalized_mutual_info_score(y_true, y_pred)})
            for c in ["assay_type", "readout", "publication", "chemistry_regime"]:
                if c in sub.columns:
                    align_rows.append({"component": c, "ARI": adjusted_rand_score(sub[c].astype(str), y_pred), "NMI": normalized_mutual_info_score(sub[c].astype(str), y_pred)})
    alignment = pd.DataFrame(align_rows)
    alignment.to_csv(reports_dir / "alignment_metrics.csv", index=False)

    prof_cols = [c for c in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "pIC50"] if c in comp.columns]
    active_col = pick(comp, ["activity_label", "label", "active"])
    prof = comp.loc[valid_idx].groupby("learned_env_id")[prof_cols].agg(["mean", "std"])
    prof.columns = [f"{a}_{b}" for a, b in prof.columns]
    prof = prof.reset_index()
    prof["n"] = comp.loc[valid_idx].groupby("learned_env_id").size().values
    if active_col:
        prof["active_rate"] = comp.loc[valid_idx].groupby("learned_env_id")[active_col].mean().values
    prof.to_csv(reports_dir / "cluster_profiles.csv", index=False)

    purity_rows = []
    for comp_col in ["assay_type", "readout", "publication", "chemistry_regime"]:
        if comp_col not in comp.columns:
            continue
        for cid, g in comp.loc[valid_idx].groupby("learned_env_id"):
            top_frac = g[comp_col].astype(str).value_counts(normalize=True).iloc[0] if len(g) else np.nan
            purity_rows.append({"learned_env_id": cid, "component": comp_col, "purity": top_frac, "n": len(g)})
    purity = pd.DataFrame(purity_rows)
    purity.to_csv(reports_dir / "cluster_purity.csv", index=False)

    contingency = pd.DataFrame()
    if manual_col:
        contingency = pd.crosstab(comp.loc[valid_idx, manual_col].astype(str), comp.loc[valid_idx, "learned_env_id"].astype(int))
        contingency.reset_index().to_csv(reports_dir / "manual_vs_learned_contingency.csv", index=False)

    # figures
    sizes = assignments["learned_env_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sizes.index.astype(str), sizes.values, color=style.palette[0])
    style_axis(ax, style, title="Cluster sizes", xlabel="learned_env_id", ylabel="Count")
    fig.tight_layout(); fig.savefig(figs_dir / "fig_cluster_sizes.svg"); plt.close(fig)

    means = comp.loc[valid_idx].groupby("learned_env_id")[[c for c in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings"] if c in comp.columns]].mean()
    means_norm = (means - means.min()) / (means.max() - means.min() + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(means_norm.columns))
    w = 0.8 / max(len(means_norm), 1)
    for i, (cid, row) in enumerate(means_norm.iterrows()):
        ax.bar(x + i * w, row.values, width=w, label=f"cluster {cid}", color=style.palette[i % len(style.palette)])
    ax.set_xticks(x + w * max(len(means_norm) - 1, 0) / 2)
    ax.set_xticklabels(means_norm.columns)
    ax.legend()
    style_axis(ax, style, title="Cluster profile (normalized means)", xlabel="Descriptor", ylabel="Normalized mean")
    fig.tight_layout(); fig.savefig(figs_dir / "fig_cluster_profiles.svg"); plt.close(fig)

    if not alignment.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(alignment))
        ax.bar(x - 0.18, alignment["ARI"], width=0.36, label="ARI", color=style.palette[2])
        ax.bar(x + 0.18, alignment["NMI"], width=0.36, label="NMI", color=style.palette[3])
        ax.set_xticks(x)
        ax.set_xticklabels(alignment["component"], rotation=30, ha="right")
        ax.legend()
        style_axis(ax, style, title="Alignment: manual vs learned", ylabel="Score")
        fig.tight_layout(); fig.savefig(figs_dir / "fig_alignment_ari_nmi.svg"); plt.close(fig)

    if not contingency.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(contingency.values, cmap="Purples")
        ax.set_xticks(range(contingency.shape[1]))
        ax.set_xticklabels(contingency.columns)
        ax.set_yticks(range(contingency.shape[0]))
        ax.set_yticklabels(contingency.index)
        style_axis(ax, style, title="Manual vs learned contingency", xlabel="learned_env_id", ylabel="manual env_id")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(figs_dir / "fig_manual_vs_learned_contingency.svg"); plt.close(fig)

    run_config = {
        "step": "latent_env_discovery",
        "features": features,
        "method": args.method,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "select_by": args.select_by,
        "random_seed": args.random_seed,
        "plotting": {
            "font": args.font,
            "bold_text": bool(args.bold_text),
            "palette": "nature5",
            "font_title": args.font_title,
            "font_label": args.font_label,
            "font_tick": args.font_tick,
            "font_legend": args.font_legend,
            "savefig.format": "svg",
            "svg.fonttype": "none",
        },
    }
    (prov_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": sys.argv[1:],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_hash": git_commit(),
        "script_sha256": {
            "scripts/latent_env_discovery.py": sha256_file(Path("scripts/latent_env_discovery.py")),
            "scripts/plot_style.py": sha256_file(Path("scripts/plot_style.py")),
        },
        "input_files": [{"path": str(Path(args.input_compound_parquet)), "sha256": sha256_file(Path(args.input_compound_parquet))}],
        "selected_k": int(best["k"]),
        "selection_criterion": args.select_by,
        "num_clusters": int(pd.Series(best["labels"]).nunique()),
        "cluster_sizes": assignments["learned_env_id"].value_counts().to_dict(),
    }
    (prov_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    write_env(prov_dir / "environment.txt")

    print(f"Latent environment discovery complete. selected_k={best['k']}")


if __name__ == "__main__":
    main()
