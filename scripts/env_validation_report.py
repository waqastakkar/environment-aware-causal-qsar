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
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_style import NATURE5, PlotStyle, configure_matplotlib, style_axis


DESC_COLS = ["MW", "TPSA", "LogP", "HBD", "HBA", "RotB", "Rings"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate assembled environments and produce reports/figures/provenance.")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--outdir", required=True)
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


def write_env(path: Path) -> None:
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")


def pick(df: pd.DataFrame, options: list[str]) -> str | None:
    m = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in m:
            return m[o.lower()]
    return None


def save_reports(row_df: pd.DataFrame, comp_df: pd.DataFrame, reports_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    env_col = "env_id"

    # label shift
    active_col = pick(comp_df, ["activity_label", "label", "active"])
    label = comp_df.groupby(env_col).agg(
        count=(env_col, "size"),
        mean_pIC50=("pIC50", "mean"),
        std_pIC50=("pIC50", "std"),
        active_rate=(active_col, "mean") if active_col else ("pIC50", lambda s: np.nan),
    ).reset_index()
    label.to_csv(reports_dir / "label_shift.csv", index=False)
    out["label_shift"] = label

    # scaffold overlap
    scaffold_col = pick(comp_df, ["scaffold_id", "scaffold", "murcko_scaffold"])
    env_scaf = {e: set(g[scaffold_col].dropna().astype(str)) for e, g in comp_df.groupby(env_col)} if scaffold_col else {}
    envs = sorted(comp_df[env_col].dropna().unique())
    mat = pd.DataFrame(index=envs, columns=envs, dtype=float)
    for e1 in envs:
        for e2 in envs:
            s1, s2 = env_scaf.get(e1, set()), env_scaf.get(e2, set())
            denom = len(s1 | s2)
            mat.loc[e1, e2] = (len(s1 & s2) / denom) if denom else 0.0
    mat.index.name = "env_id"
    mat.reset_index().to_csv(reports_dir / "scaffold_overlap.csv", index=False)
    out["scaffold_overlap"] = mat

    # shift metrics
    pairs = []
    descriptor_map = {d: pick(comp_df, [d, d.lower()]) for d in DESC_COLS}
    env_groups = {e: g for e, g in comp_df.groupby(env_col)}
    keys = sorted(env_groups)
    for i, e1 in enumerate(keys):
        for e2 in keys[i + 1 :]:
            row = {"env_a": e1, "env_b": e2}
            vals = []
            for d, col in descriptor_map.items():
                if col is None:
                    row[f"wasserstein_{d}"] = np.nan
                    continue
                a = env_groups[e1][col].dropna().values
                b = env_groups[e2][col].dropna().values
                wd = float(wasserstein_distance(a, b)) if len(a) and len(b) else np.nan
                row[f"wasserstein_{d}"] = wd
                if not np.isnan(wd):
                    vals.append(wd)
            row["aggregate_shift"] = float(np.mean(vals)) if vals else np.nan
            pairs.append(row)
    shift = pd.DataFrame(pairs)
    shift.to_csv(reports_dir / "shift_metrics.csv", index=False)
    out["shift_metrics"] = shift

    # env predictability
    num_cols = [c for c in descriptor_map.values() if c is not None]
    cat_cols = [c for c in [pick(comp_df, ["assay_type"]), pick(comp_df, ["species"]), pick(comp_df, ["readout"])] if c]
    model_df = comp_df[[env_col, *num_cols, *cat_cols]].dropna(subset=[env_col]).copy()
    model_df = model_df.dropna(how="all", subset=num_cols + cat_cols)
    if len(model_df) >= 20 and model_df[env_col].nunique() > 1:
        X = model_df[num_cols + cat_cols]
        y = model_df[env_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        ct = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])
        clf = Pipeline([
            ("prep", ct),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ])
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        auc = np.nan
        try:
            proba = clf.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba, multi_class="ovr")
        except Exception:
            pass
        pred_df = pd.DataFrame([{"model": "random_forest", "n_train": len(X_train), "n_test": len(X_test), "accuracy": acc, "roc_auc_ovr": auc}])
    else:
        pred_df = pd.DataFrame([{"model": "random_forest", "n_train": 0, "n_test": 0, "accuracy": np.nan, "roc_auc_ovr": np.nan}])
    pred_df.to_csv(reports_dir / "env_predictability.csv", index=False)
    out["env_predictability"] = pred_df

    # missingness
    key_cols = [env_col, "pIC50", active_col, scaffold_col, *num_cols, *cat_cols]
    key_cols = [c for c in key_cols if c]
    miss = comp_df.groupby(env_col)[key_cols].apply(lambda g: g.isna().mean()).reset_index()
    miss.to_csv(reports_dir / "missingness_by_env.csv", index=False)
    out["missingness"] = miss

    return out


def save_figures(comp_df: pd.DataFrame, rep: dict[str, pd.DataFrame], figures_dir: Path, style: PlotStyle) -> None:
    env_col = "env_id"

    c = comp_df[env_col].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(c.index.astype(str), c.values, color=style.palette[0])
    ax.tick_params(axis="x", rotation=90)
    style_axis(ax, style, title="Environment counts", xlabel="env_id", ylabel="Count")
    fig.tight_layout(); fig.savefig(figures_dir / "fig_env_counts.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = [g["pIC50"].dropna().values for _, g in comp_df.groupby(env_col)]
    labels = [str(e) for e, _ in comp_df.groupby(env_col)]
    ax.boxplot(groups, labels=labels, patch_artist=True)
    ax.tick_params(axis="x", rotation=90)
    style_axis(ax, style, title="pIC50 distribution by environment", xlabel="env_id", ylabel="pIC50")
    fig.tight_layout(); fig.savefig(figures_dir / "fig_label_distribution_by_env.svg"); plt.close(fig)

    label_df = rep["label_shift"].sort_values("count", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(label_df[env_col].astype(str), label_df["active_rate"], color=style.palette[1])
    ax.tick_params(axis="x", rotation=90)
    style_axis(ax, style, title="Active rate by environment", xlabel="env_id", ylabel="Active rate")
    fig.tight_layout(); fig.savefig(figures_dir / "fig_active_rate_by_env.svg"); plt.close(fig)

    ov = rep["scaffold_overlap"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(ov.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ov.columns))); ax.set_xticklabels(ov.columns, rotation=90)
    ax.set_yticks(range(len(ov.index))); ax.set_yticklabels(ov.index)
    style_axis(ax, style, title="Scaffold overlap (Jaccard)", xlabel="env", ylabel="env")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(figures_dir / "fig_scaffold_overlap_heatmap.svg"); plt.close(fig)

    shift = rep["shift_metrics"].sort_values("aggregate_shift", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = shift["env_a"].astype(str) + " vs " + shift["env_b"].astype(str)
    ax.bar(labels, shift["aggregate_shift"], color=style.palette[2])
    ax.tick_params(axis="x", rotation=90)
    style_axis(ax, style, title="Aggregate descriptor shift", xlabel="Env pair", ylabel="Mean Wasserstein")
    fig.tight_layout(); fig.savefig(figures_dir / "fig_shift_metrics.svg"); plt.close(fig)

    pred = rep["env_predictability"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Accuracy", "ROC-AUC"], [pred["accuracy"].iloc[0], pred["roc_auc_ovr"].iloc[0]], color=[style.palette[3], style.palette[4]])
    style_axis(ax, style, title="Environment predictability")
    fig.tight_layout(); fig.savefig(figures_dir / "fig_env_predictability.svg"); plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    reports_dir = outdir / "reports"
    figures_dir = outdir / "figures"
    prov_dir = outdir / "provenance"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)

    style = PlotStyle(
        font_family=args.font,
        font_title=args.font_title,
        font_label=args.font_label,
        font_tick=args.font_tick,
        font_legend=args.font_legend,
        palette=tuple(NATURE5),
    )
    configure_matplotlib(style)

    row_df = pd.read_parquet(input_dir / "multienv_row_level.parquet")
    comp_df = pd.read_parquet(input_dir / "multienv_compound_level.parquet")

    reports = save_reports(row_df, comp_df, reports_dir)
    save_figures(comp_df, reports, figures_dir, style)

    run_config = {
        "step": "env_validation_report",
        "input_dir": str(input_dir),
        "outdir": str(outdir),
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

    inputs = [input_dir / "multienv_row_level.parquet", input_dir / "multienv_compound_level.parquet", input_dir / "env_definitions.json", input_dir / "env_vector_schema.json"]
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": sys.argv[1:],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_hash": git_commit(),
        "script_sha256": {
            "scripts/env_validation_report.py": sha256_file(Path("scripts/env_validation_report.py")),
            "scripts/plot_style.py": sha256_file(Path("scripts/plot_style.py")),
        },
        "input_files": [{"path": str(p), "sha256": sha256_file(p)} for p in inputs if p.exists()],
        "row_counts": {"row_level": int(len(row_df)), "compound_level": int(len(comp_df)), "num_environments": int(comp_df["env_id"].nunique())},
    }
    (prov_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    write_env(prov_dir / "environment.txt")
    print(f"Wrote reports={reports_dir}, figures={figures_dir}, provenance={prov_dir}")


if __name__ == "__main__":
    main()
