#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.plot_style import PlotStyle, configure_matplotlib, style_axis
except ModuleNotFoundError:
    from plot_style import PlotStyle, configure_matplotlib, style_axis


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BBB/CNS stratification")
    p.add_argument("--target", required=True)
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--compute_cns_mpo", action="store_true")
    p.add_argument("--cns_mpo_threshold", type=float, default=4.0)
    p.add_argument("--cns_bins", nargs="+", type=float, default=[0.0, 2.0, 4.0, 6.0])
    p.add_argument("--pgp_model_path", default="")
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--font_title", type=int, default=16)
    p.add_argument("--font_label", type=int, default=14)
    p.add_argument("--font_tick", type=int, default=12)
    p.add_argument("--font_legend", type=int, default=12)
    return p.parse_args()


def desirability(x: pd.Series, low: float, high: float, mode: str = "inside") -> pd.Series:
    if mode == "inside":
        return ((x >= low) & (x <= high)).astype(float)
    if mode == "low":
        return (x <= high).astype(float)
    return (x >= low).astype(float)


def compute_mpo(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, bool]:
    required = ["MW", "LogP", "TPSA", "HBD", "RotB", "HBA"]
    if not all(c in df.columns for c in required):
        return pd.Series(np.nan, index=df.index), pd.DataFrame(index=df.index), False
    comps = pd.DataFrame(index=df.index)
    comps["d_MW"] = desirability(df["MW"], 200, 450, "inside")
    comps["d_LogP"] = desirability(df["LogP"], 1, 4, "inside")
    comps["d_TPSA"] = desirability(df["TPSA"], 20, 90, "inside")
    comps["d_HBD"] = desirability(df["HBD"], -np.inf, 3, "low")
    comps["d_RotB"] = desirability(df["RotB"], -np.inf, 8, "low")
    comps["d_HBA"] = desirability(df["HBA"], -np.inf, 10, "low")
    mpo = comps.sum(axis=1)
    return mpo, comps, True


def compute_rule_flag(df: pd.DataFrame) -> pd.Series:
    cond = pd.Series(True, index=df.index)
    if "TPSA" in df.columns:
        cond &= df["TPSA"] <= 90
    if "MW" in df.columns:
        cond &= df["MW"] <= 450
    if "HBD" in df.columns:
        cond &= df["HBD"] <= 3
    if "RotB" in df.columns:
        cond &= df["RotB"] <= 8
    if "LogP" in df.columns:
        cond &= df["LogP"].between(1, 4)
    return cond


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    data_dir, rep_dir, fig_dir, prov_dir = [outdir / x for x in ["data", "reports", "figures", "provenance"]]
    for d in [data_dir, rep_dir, fig_dir, prov_dir]:
        d.mkdir(parents=True, exist_ok=True)

    style = PlotStyle(font_family=args.font, font_title=args.font_title, font_label=args.font_label, font_tick=args.font_tick, font_legend=args.font_legend)
    configure_matplotlib(style)
    pal = list(style.palette)

    df = pd.read_parquet(args.input_parquet)
    mpo, comps, mpo_ok = compute_mpo(df) if args.compute_cns_mpo else (pd.Series(np.nan, index=df.index), pd.DataFrame(index=df.index), False)

    if mpo_ok:
        df["cns_mpo"] = mpo
        df["cns_like"] = (df["cns_mpo"] >= args.cns_mpo_threshold).astype(int)
        df["cns_flag_method"] = "mpo"
    else:
        flag = compute_rule_flag(df)
        df["cns_mpo"] = np.nan
        df["cns_like"] = flag.astype(int)
        df["cns_flag_method"] = "rules"

    labels = [f"[{args.cns_bins[i]}, {args.cns_bins[i+1]})" for i in range(len(args.cns_bins)-1)]
    df["cns_bin"] = pd.cut(df["cns_mpo"].fillna(-1), bins=args.cns_bins, labels=labels, include_lowest=True)

    if args.pgp_model_path:
        import joblib

        model = joblib.load(args.pgp_model_path)
        feat_cols = [c for c in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings"] if c in df.columns]
        pred = model.predict_proba(df[feat_cols].fillna(df[feat_cols].median()))[:, 1] if hasattr(model, "predict_proba") else model.predict(df[feat_cols].fillna(df[feat_cols].median()))
        df["pgp_risk"] = pred
        pd.DataFrame({"molecule_id": df["molecule_id"], "pgp_risk": pred}).to_csv(data_dir / "pgp_predictions.csv", index=False)

    df.to_parquet(data_dir / "bbb_annotations.parquet", index=False)
    comps.assign(molecule_id=df["molecule_id"]).to_csv(data_dir / "cns_mpo_components.csv", index=False)
    pd.DataFrame({"molecule_id": df["molecule_id"], "cns_bin": df["cns_bin"].astype(str)}).to_csv(data_dir / "cns_bins.csv", index=False)

    summary = pd.DataFrame([
        {"group": "all", "n": len(df), "cns_like_rate": df["cns_like"].mean(), "cns_mpo_mean": df["cns_mpo"].mean(), "cns_mpo_std": df["cns_mpo"].std(), "pIC50_mean": df["pIC50"].mean(), "pIC50_std": df["pIC50"].std()},
        {"group": "cns_like", "n": int(df["cns_like"].sum()), "cns_like_rate": 1.0, "cns_mpo_mean": df.loc[df["cns_like"] == 1, "cns_mpo"].mean(), "cns_mpo_std": df.loc[df["cns_like"] == 1, "cns_mpo"].std(), "pIC50_mean": df.loc[df["cns_like"] == 1, "pIC50"].mean(), "pIC50_std": df.loc[df["cns_like"] == 1, "pIC50"].std()},
        {"group": "non_cns", "n": int((df["cns_like"] == 0).sum()), "cns_like_rate": 0.0, "cns_mpo_mean": df.loc[df["cns_like"] == 0, "cns_mpo"].mean(), "cns_mpo_std": df.loc[df["cns_like"] == 0, "cns_mpo"].std(), "pIC50_mean": df.loc[df["cns_like"] == 0, "pIC50"].mean(), "pIC50_std": df.loc[df["cns_like"] == 0, "pIC50"].std()},
    ])
    summary.to_csv(rep_dir / "bbb_summary.csv", index=False)

    shift_rows = []
    splits_dir = Path(args.splits_dir)
    if splits_dir.exists():
        for sp in sorted([x for x in splits_dir.iterdir() if x.is_dir()]):
            for part in ["train", "val", "test"]:
                p = sp / f"{part}_ids.csv"
                if not p.exists():
                    continue
                ids = pd.read_csv(p)["molecule_id"]
                sub = df[df["molecule_id"].isin(ids)]
                shift_rows.append({"split": sp.name, "partition": part, "n": len(sub), "cns_like_rate": sub["cns_like"].mean() if len(sub) else np.nan, "cns_mpo_mean": sub["cns_mpo"].mean() if len(sub) else np.nan, "cns_mpo_std": sub["cns_mpo"].std() if len(sub) else np.nan})
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(rep_dir / "bbb_shift_by_split.csv", index=False)

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def scaf(sm):
        m = Chem.MolFromSmiles(str(sm)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else "INVALID"

    df["_scaffold"] = df["canonical_smiles"].map(scaf)
    env_col = "env_id_manual" if "env_id_manual" in df.columns else None
    cns = df[df["cns_like"] == 1]
    non = df[df["cns_like"] == 0]
    overlap = {
        "scaffold_overlap_count": len(set(cns["_scaffold"]) & set(non["_scaffold"])),
        "scaffold_overlap_fraction": len(set(cns["_scaffold"]) & set(non["_scaffold"])) / max(len(set(cns["_scaffold"])), 1),
    }
    if env_col:
        overlap["env_overlap_count"] = len(set(cns[env_col].astype(str)) & set(non[env_col].astype(str)))
        overlap["env_overlap_fraction"] = overlap["env_overlap_count"] / max(len(set(cns[env_col].astype(str))), 1)
    pd.DataFrame([overlap]).to_csv(rep_dir / "cns_vs_non_cns_overlap.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["cns_mpo"].dropna(), bins=20, color=pal[0], alpha=0.9)
    ax.axvline(args.cns_mpo_threshold, color=pal[3], linestyle="--", linewidth=2)
    style_axis(ax, style, "CNS MPO distribution", "CNS MPO", "Count")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_cns_mpo_distribution.svg"); plt.close(fig)

    if not shift_df.empty:
        pivot = shift_df[shift_df["partition"] == "test"].set_index("split")["cns_like_rate"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(pivot.index, pivot.values, color=pal[1])
        ax.set_xticklabels(pivot.index, rotation=45, ha="right")
        style_axis(ax, style, "CNS-like rate by split (test)", "Split", "Rate")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_cns_like_rate_by_split.svg"); plt.close(fig)
    else:
        (fig_dir / "fig_cns_like_rate_by_split.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">No split manifests found</text></svg>')

    fig, ax = plt.subplots(figsize=(7, 5))
    for val, color, label in [(0, pal[2], "non_cns"), (1, pal[4], "cns_like")]:
        sub = df[df["cns_like"] == val]
        ax.scatter(sub["cns_mpo"], sub["pIC50"], s=15, alpha=0.6, color=color, label=label)
    ax.legend(); style_axis(ax, style, "Potency vs CNS MPO", "CNS MPO", "pIC50")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_potency_vs_cns_mpo.svg"); plt.close(fig)

    tmp = df[["cns_mpo", "pIC50"]].dropna().sort_values("cns_mpo")
    if not tmp.empty:
        tmp["bin"] = pd.qcut(tmp["cns_mpo"], q=min(10, tmp["cns_mpo"].nunique()), duplicates="drop")
        frontier = tmp.groupby("bin", observed=True).agg(cns_mpo=("cns_mpo", "median"), best_pIC50=("pIC50", "max")).sort_values("cns_mpo")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(frontier["cns_mpo"], frontier["best_pIC50"], marker="o", color=pal[0])
        style_axis(ax, style, "Pareto frontier: potency vs CNS MPO", "CNS MPO", "Top pIC50")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_pareto_frontier.svg"); plt.close(fig)

    if "pgp_risk" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df["pgp_risk"].dropna(), bins=20, color=pal[3])
        style_axis(ax, style, "P-gp risk distribution", "P-gp risk", "Count")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_pgp_risk_distribution.svg"); plt.close(fig)

    run_cfg = vars(args)
    (prov_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    prov = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() if Path(".git").exists() else None,
        "script_sha256": {"scripts/bbb_stratify.py": sha256_file(Path(__file__)), "scripts/plot_style.py": sha256_file(Path("scripts/plot_style.py")) if Path("scripts/plot_style.py").exists() else None},
        "input_parquet": {"path": args.input_parquet, "sha256": sha256_file(Path(args.input_parquet))},
        "row_counts": {"input": int(len(df))},
        "mpo_threshold": args.cns_mpo_threshold,
        "bins": args.cns_bins,
        "rules_thresholds": {"TPSA_max": 90, "MW_max": 450, "HBD_max": 3, "RotB_max": 8, "LogP_min": 1, "LogP_max": 4},
        "pgp_model_used": bool(args.pgp_model_path),
    }
    (prov_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    (prov_dir / "environment.txt").write_text(subprocess.check_output(["pip", "freeze"], text=True))


if __name__ == "__main__":
    main()
