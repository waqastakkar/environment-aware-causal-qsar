#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as exc:  # pragma: no cover
    raise RuntimeError("RDKit is required for splits_report.py") from exc

try:
    from scripts.plot_style import PlotStyle, configure_matplotlib, style_axis
except ModuleNotFoundError:
    from plot_style import PlotStyle, configure_matplotlib, style_axis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split reporting and SVG figures")
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--font_title", type=int, default=16)
    p.add_argument("--font_label", type=int, default=14)
    p.add_argument("--font_tick", type=int, default=12)
    p.add_argument("--font_legend", type=int, default=12)
    p.add_argument("--palette", default="nature5")
    p.add_argument("--id_col", default="molecule_id")
    return p.parse_args()


def mfp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def read_ids(path: Path, id_col: str) -> list:
    if not path.exists():
        return []
    ids = pd.read_csv(path)
    if id_col in ids.columns:
        return ids[id_col].astype(str).tolist()
    fallback_col = next((c for c in ["molecule_id", "molecule_chembl_id", "chembl_molecule_id", "compound_id", "mol_id", "id"] if c in ids.columns), None)
    if fallback_col is None:
        return []
    logging.warning("Split ID file %s missing column '%s'; using '%s'", path.name, id_col, fallback_col)
    return ids[fallback_col].astype(str).tolist()


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "NOSCAF"


def resolve_property_columns(columns: list[str]) -> dict[str, str]:
    alias_map = {
        "MW": ["mw", "molecularweight", "molecular_weight", "molweight", "exactmw", "exact_mw", "amw"],
        "logP": ["logp", "xlogp", "alogp", "clogp", "mol_logp", "mollogp"],
        "TPSA": ["tpsa", "topologicalpolar", "topological_polar_surface_area", "polar_surface_area", "psa"],
        "HBD": ["hbd", "h_donors", "hbonddonors", "numhdonors", "hbond_donors"],
        "HBA": ["hba", "h_acceptors", "hbondacceptors", "numhacceptors", "hbond_acceptors"],
        "RB": ["rb", "rotb", "rotatablebonds", "nrotb", "numrotatablebonds", "rotatable_bonds"],
    }

    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    normalized = {norm(c): c for c in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in normalized:
                resolved[canonical] = normalized[alias]
                break
    return resolved


def write_skip_panel(fig_path: Path, message: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    ax.text(0.02, 0.72, title, fontsize=14, fontweight="bold", ha="left", va="center", transform=ax.transAxes)
    ax.text(0.02, 0.35, message, fontsize=12, ha="left", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp, wasserstein_distance

    style = PlotStyle(font_family=args.font, font_title=args.font_title, font_label=args.font_label, font_tick=args.font_tick, font_legend=args.font_legend)
    configure_matplotlib(style)

    outdir = Path(args.outdir)
    reports = outdir / "reports"
    figures = outdir / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = pd.read_parquet(args.input_parquet)
    canonical_id = "molecule_id"
    if canonical_id not in df.columns:
        fallback_col = next((c for c in ["molecule_chembl_id", "chembl_molecule_id", "compound_id", "mol_id", "id"] if c in df.columns), None)
        if fallback_col is not None:
            logging.warning("Creating canonical molecule_id from source column '%s'", fallback_col)
            df[canonical_id] = df[fallback_col]

    if args.id_col not in df.columns:
        alternatives = [c for c in ["molecule_id", "molecule_chembl_id", "chembl_molecule_id", "compound_id", "mol_id", "id"] if c in df.columns]
        avail = ", ".join(map(str, df.columns))
        hint = f" Try --id_col one of: {', '.join(alternatives)}." if alternatives else ""
        raise ValueError(f"id column '{args.id_col}' not found. Available columns: {avail}.{hint}")

    if args.id_col != canonical_id:
        logging.warning("Using id_col '%s'; also standardizing canonical molecule_id from this column", args.id_col)
        df[canonical_id] = df[args.id_col]

    df[args.id_col] = df[args.id_col].astype(str)
    df[canonical_id] = df[canonical_id].astype(str)
    if "_scaffold" not in df.columns:
        df["_scaffold"] = df["canonical_smiles"].astype(str).map(scaffold)

    split_dirs = [p for p in Path(args.splits_dir).iterdir() if p.is_dir()]
    label_rows, cov_rows, scaf_rows, env_rows, sim_rows, time_rows, size_rows = [], [], [], [], [], [], []

    prop_cols = resolve_property_columns(df.columns.tolist())
    env_col = "env_id_manual" if "env_id_manual" in df.columns else ("assay_type" if "assay_type" in df.columns else None)

    run_cfg_path = outdir / "provenance" / "run_config.json"
    time_key = "publication_year"
    if run_cfg_path.exists():
        try:
            run_cfg = json.loads(run_cfg_path.read_text())
            time_key = run_cfg.get("time_key", time_key)
        except Exception as exc:
            logging.warning("Unable to parse %s (%s); defaulting time_key='%s'", run_cfg_path, exc, time_key)

    split_names = [p.name for p in split_dirs]
    time_split_enabled = any("time" in s.lower() for s in split_names)

    for sp in sorted(split_dirs):
        name = sp.name
        tr_ids = set(read_ids(sp / "train_ids.csv", args.id_col))
        va_ids = set(read_ids(sp / "val_ids.csv", args.id_col))
        te_ids = set(read_ids(sp / "test_ids.csv", args.id_col))
        tr = df[df[args.id_col].isin(tr_ids)]
        te = df[df[args.id_col].isin(te_ids)]
        va = df[df[args.id_col].isin(va_ids)]
        size_rows.append({"split": name, "n_train": len(tr), "n_val": len(va), "n_test": len(te)})
        if len(tr) == 0 or len(te) == 0:
            continue

        label_rows.append({
            "split": name,
            "train_mean_pIC50": tr["pIC50"].mean(),
            "test_mean_pIC50": te["pIC50"].mean(),
            "train_std_pIC50": tr["pIC50"].std(),
            "test_std_pIC50": te["pIC50"].std(),
            "train_active_rate": tr["activity_label"].mean(),
            "test_active_rate": te["activity_label"].mean(),
        })
        for prop_name, prop_col in prop_cols.items():
            if prop_col in tr.columns and prop_col in te.columns:
                tv = tr[prop_col].dropna().to_numpy()
                qv = te[prop_col].dropna().to_numpy()
                if len(tv) and len(qv):
                    cov_rows.append({"split": name, "property": prop_name, "source_column": prop_col, "ks_stat": ks_2samp(tv, qv).statistic, "wasserstein": wasserstein_distance(tv, qv)})

        tr_scaf = set(tr["_scaffold"].astype(str))
        te_scaf = set(te["_scaffold"].astype(str))
        scaf_rows.append({"split": name, "train_scaffolds": len(tr_scaf), "test_scaffolds": len(te_scaf), "overlap_count": len(tr_scaf & te_scaf), "overlap_fraction_test": len(tr_scaf & te_scaf) / max(len(te_scaf), 1)})

        if env_col:
            tr_env = set(tr[env_col].astype(str))
            te_env = set(te[env_col].astype(str))
            env_rows.append({"split": name, "env_col": env_col, "train_envs": len(tr_env), "test_envs": len(te_env), "overlap_count": len(tr_env & te_env), "overlap_fraction_test": len(tr_env & te_env) / max(len(te_env), 1)})

        tr_fps = [mfp(s) for s in tr["canonical_smiles"].astype(str)]
        tr_fps = [x for x in tr_fps if x is not None]
        max_sims = []
        for s in te["canonical_smiles"].astype(str):
            fp = mfp(s)
            if fp is None or not tr_fps:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, tr_fps)
            max_sims.append(max(sims) if sims else 0.0)
        arr = np.array(max_sims) if max_sims else np.array([0.0])
        sim_rows.append({"split": name, "n_test_scored": len(max_sims), "sim_min": np.min(arr), "sim_median": np.median(arr), "sim_max": np.max(arr), "count_ge_0.8": int((arr >= 0.8).sum()), "count_ge_0.85": int((arr >= 0.85).sum()), "count_ge_0.9": int((arr >= 0.9).sum())})

        if time_key in df.columns:
            time_rows.append({"split": name, "train_year_min": tr[time_key].min(), "train_year_max": tr[time_key].max(), "test_year_min": te[time_key].min(), "test_year_max": te[time_key].max()})

    label_df = pd.DataFrame(label_rows)
    cov_df = pd.DataFrame(cov_rows)
    scaf_df = pd.DataFrame(scaf_rows)
    env_df = pd.DataFrame(env_rows)
    sim_df = pd.DataFrame(sim_rows)
    time_df = pd.DataFrame(time_rows)
    size_df = pd.DataFrame(size_rows)

    size_df.to_csv(reports / "split_summary.csv", index=False)
    label_df.to_csv(reports / "label_shift.csv", index=False)
    cov_df.to_csv(reports / "covariate_shift.csv", index=False)
    scaf_df.to_csv(reports / "scaffold_overlap.csv", index=False)
    env_df.to_csv(reports / "env_overlap.csv", index=False)
    sim_df.to_csv(reports / "similarity_leakage.csv", index=False)
    time_df.to_csv(reports / "time_coverage.csv", index=False)
    if not (reports / "group_integrity_checks.csv").exists():
        pd.DataFrame().to_csv(reports / "group_integrity_checks.csv", index=False)
    if not (reports / "matching_quality.csv").exists():
        pd.DataFrame().to_csv(reports / "matching_quality.csv", index=False)

    pal = list(style.palette)
    plot_status = []

    if not size_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(size_df))
        ax.bar(x - 0.25, size_df["n_train"], width=0.25, label="train", color=pal[0])
        ax.bar(x, size_df["n_val"], width=0.25, label="val", color=pal[1])
        ax.bar(x + 0.25, size_df["n_test"], width=0.25, label="test", color=pal[2])
        ax.set_xticks(x)
        ax.set_xticklabels(size_df["split"], rotation=45, ha="right")
        ax.legend()
        style_axis(ax, style, "Split sizes", "Split", "Count")
        fig.tight_layout(); fig.savefig(figures / "fig_split_sizes.svg"); plt.close(fig)
        plot_status.append({"plot": "fig_split_sizes.svg", "status": "CREATED", "reason": "ok"})
    else:
        write_skip_panel(figures / "fig_split_sizes.svg", "SKIPPED: no split size rows", "Split sizes")
        plot_status.append({"plot": "fig_split_sizes.svg", "status": "SKIPPED", "reason": "SKIPPED: no split size rows"})

    if not label_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(label_df))
        ax.bar(x - 0.2, label_df["train_mean_pIC50"], width=0.4, label="train", color=pal[3])
        ax.bar(x + 0.2, label_df["test_mean_pIC50"], width=0.4, label="test", color=pal[4])
        ax.set_xticks(x); ax.set_xticklabels(label_df["split"], rotation=45, ha="right")
        ax.legend(); style_axis(ax, style, "Label shift by split", "Split", "Mean pIC50")
        fig.tight_layout(); fig.savefig(figures / "fig_label_shift_by_split.svg"); plt.close(fig)
        plot_status.append({"plot": "fig_label_shift_by_split.svg", "status": "CREATED", "reason": "ok"})
    else:
        write_skip_panel(figures / "fig_label_shift_by_split.svg", "SKIPPED: no label-shift rows", "Label shift by split")
        plot_status.append({"plot": "fig_label_shift_by_split.svg", "status": "SKIPPED", "reason": "SKIPPED: no label-shift rows"})

    cov_skip_reason = None
    if not prop_cols:
        cov_skip_reason = "SKIPPED: no property columns found"
        logging.warning("Covariate shift plot skipped: no property columns found")

    if not cov_df.empty and cov_skip_reason is None:
        pivot = cov_df.pivot_table(index="split", columns="property", values="ks_stat", aggfunc="mean").fillna(0)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index)
        style_axis(ax, style, "Covariate shift (KS)", "Property", "Split")
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(figures / "fig_covariate_shift_props.svg"); plt.close(fig)
        plot_status.append({"plot": "fig_covariate_shift_props.svg", "status": "CREATED", "reason": "ok"})
    else:
        reason = cov_skip_reason or "SKIPPED: no covariate-shift data"
        write_skip_panel(figures / "fig_covariate_shift_props.svg", reason, "Covariate shift")
        plot_status.append({"plot": "fig_covariate_shift_props.svg", "status": "SKIPPED", "reason": reason})

    for fname, dff, y, title in [
        ("fig_scaffold_overlap_by_split.svg", scaf_df, "overlap_fraction_test", "Scaffold overlap by split"),
        ("fig_env_overlap_by_split.svg", env_df, "overlap_fraction_test", "Environment overlap by split"),
        ("fig_similarity_leakage.svg", sim_df, "sim_median", "Similarity leakage (median max sim)"),
    ]:
        if not dff.empty and y in dff.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(dff["split"], dff[y], color=pal[0:len(dff)])
            ax.set_xticklabels(dff["split"], rotation=45, ha="right")
            style_axis(ax, style, title, "Split", y)
            fig.tight_layout(); fig.savefig(figures / fname); plt.close(fig)
            plot_status.append({"plot": fname, "status": "CREATED", "reason": "ok"})
        else:
            (figures / fname).write_text('<svg xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">No data</text></svg>')
            plot_status.append({"plot": fname, "status": "SKIPPED", "reason": "SKIPPED: no data"})

    time_skip_reason = None
    if not time_split_enabled or time_key not in df.columns:
        time_skip_reason = f"SKIPPED: time split not enabled or missing date column {time_key}"
        logging.warning("Time split timeline skipped: time split not enabled or missing date column '%s'", time_key)

    if not time_df.empty and time_skip_reason is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, row in time_df.reset_index(drop=True).iterrows():
            ax.hlines(i, row["train_year_min"], row["train_year_max"], color=pal[0], linewidth=5, label="train" if i == 0 else "")
            ax.hlines(i, row["test_year_min"], row["test_year_max"], color=pal[3], linewidth=5, label="test" if i == 0 else "")
        ax.set_yticks(range(len(time_df))); ax.set_yticklabels(time_df["split"])
        ax.legend(); style_axis(ax, style, "Time split timeline", time_key, "Split")
        fig.tight_layout(); fig.savefig(figures / "fig_time_split_timeline.svg"); plt.close(fig)
        plot_status.append({"plot": "fig_time_split_timeline.svg", "status": "CREATED", "reason": "ok"})
    else:
        reason = time_skip_reason or f"SKIPPED: time split not enabled or missing date column {time_key}"
        if time_skip_reason is None:
            logging.warning("Time split timeline skipped: no timeline rows available for date column '%s'", time_key)
        write_skip_panel(figures / "fig_time_split_timeline.svg", reason, "Time split timeline")
        plot_status.append({"plot": "fig_time_split_timeline.svg", "status": "SKIPPED", "reason": reason})

    if (reports / "matching_quality.csv").exists() and (mq := pd.read_csv(reports / "matching_quality.csv")).shape[0] > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(mq))
        ax.bar(x - 0.2, mq["ks_before"], width=0.4, label="KS before", color=pal[1])
        ax.bar(x + 0.2, mq["ks_after"], width=0.4, label="KS after", color=pal[2])
        ax.set_xticks(x); ax.set_xticklabels(mq["property"], rotation=45, ha="right")
        ax.legend(); style_axis(ax, style, "Matching quality", "Property", "KS")
        fig.tight_layout(); fig.savefig(figures / "fig_matching_quality.svg"); plt.close(fig)
        plot_status.append({"plot": "fig_matching_quality.svg", "status": "CREATED", "reason": "ok"})
    else:
        reason = "SKIPPED: no matched pairs"
        logging.warning("Matching quality plot skipped: no matched pairs")
        write_skip_panel(figures / "fig_matching_quality.svg", reason, "Matching quality")
        plot_status.append({"plot": "fig_matching_quality.svg", "status": "SKIPPED", "reason": reason})

    summary_df = pd.DataFrame(plot_status)
    if not summary_df.empty:
        logging.info("Step 4 plot generation summary:\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
