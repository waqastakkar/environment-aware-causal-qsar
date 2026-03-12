#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from attribution_gnn import run_attribution_analysis
from fragment_analysis import run_fragment_analysis
from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args
from rgroup_decompose import run_rgroup_analysis
from shape_analysis import run_shape_analysis


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    pred_dir = run_dir / "predictions"
    rows = []
    for split in ["train", "val", "test"]:
        fp = pred_dir / f"{split}_predictions.parquet"
        if fp.exists():
            cols = _available_columns(fp, ["molecule_id", "compound_id", "mol_id", "yhat", "prediction", "pred", "y_pred", "pIC50_hat", "prediction_mean", "y", "split"])
            d = pd.read_parquet(fp, columns=cols if cols else None)
            d["split"] = split
            rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _rss_mb() -> float:
    # Linux ru_maxrss is KiB, macOS is bytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _log_stage(msg: str) -> None:
    print(f"[step10] {msg} | rss_max_mb={_rss_mb():.1f}", flush=True)


def _available_columns(path: Path, wanted: list[str]) -> list[str]:
    if pq is None:
        return []
    try:
        names = set(pq.ParquetFile(path).schema.names)
    except Exception:
        return []
    return [c for c in wanted if c in names]


def _parquet_rows(path: Path) -> int | None:
    if pq is None:
        return None
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of columns found: {candidates}")
    return None


def _pick_prediction_col(df: pd.DataFrame) -> str:
    candidates = ["yhat", "prediction", "pred", "y_pred", "pIC50_hat", "prediction_mean"]
    selected = _pick_col(df, candidates)
    print(f"[step10] Selected prediction column: {selected}")
    return selected


def _save_env_txt(path: Path) -> None:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:
        out = f"pip freeze failed: {exc}\n"
    path.write_text(out)


def _plot_all(outdir: Path, style, rgroup_effects: pd.DataFrame, rgroup_table: pd.DataFrame, frag_enrich: pd.DataFrame, fg_enrich: pd.DataFrame, shape_desc: pd.DataFrame, shape_vs_p: pd.DataFrame, shape_vs_r: pd.DataFrame, atom_attr: pd.DataFrame, attr_stability: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig_dir = outdir / "figures"
    data_dir = outdir / "figure_data"

    # R-group effect sizes
    top = rgroup_effects.sort_values("yhat_mean", ascending=False).head(20) if not rgroup_effects.empty else pd.DataFrame(columns=["R_smiles", "yhat_mean", "yhat_ci_low", "yhat_ci_high"])
    top.to_csv(data_dir / "rgroup_effect_sizes.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not top.empty:
        err_low = (top["yhat_mean"] - top["yhat_ci_low"]).clip(lower=0)
        err_high = (top["yhat_ci_high"] - top["yhat_mean"]).clip(lower=0)
        ax.barh(top["R_smiles"].astype(str), top["yhat_mean"], xerr=[err_low, err_high])
    style_axis(ax, style, "R-group effect sizes", "Predicted effect (yhat)", "Substitution")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_rgroup_effect_sizes.svg"); plt.close(fig)

    # top series SAR counts
    sar = rgroup_table.groupby("series_id").size().sort_values(ascending=False).head(15).reset_index(name="count") if not rgroup_table.empty else pd.DataFrame(columns=["series_id", "count"])
    sar.to_csv(data_dir / "attribution_examples.csv", index=False)  # overwritten below with actual attribution examples
    fig, ax = plt.subplots(figsize=(8, 5))
    if not sar.empty:
        ax.bar(sar["series_id"].astype(str), sar["count"])
        ax.tick_params(axis="x", rotation=45)
    style_axis(ax, style, "Top series SAR coverage", "Series", "Molecule count")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_rgroup_sar_top_series.svg"); plt.close(fig)

    # Fragment enrichment
    fplot = frag_enrich.sort_values("q_value").head(20) if not frag_enrich.empty else pd.DataFrame(columns=["feature", "odds_ratio"])
    fplot.to_csv(data_dir / "fragment_enrichment_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not fplot.empty:
        ax.barh(fplot["feature"].astype(str), fplot["odds_ratio"])
    style_axis(ax, style, "Fragment enrichment", "Odds ratio", "Fragment")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_fragment_enrichment.svg"); plt.close(fig)

    fgplot = fg_enrich.sort_values("q_value").head(20) if not fg_enrich.empty else pd.DataFrame(columns=["feature", "odds_ratio"])
    fgplot.to_csv(data_dir / "functional_group_enrichment_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not fgplot.empty:
        ax.barh(fgplot["feature"].astype(str), fgplot["odds_ratio"])
    style_axis(ax, style, "Functional group enrichment", "Odds ratio", "Functional group")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_functional_group_enrichment.svg"); plt.close(fig)

    # Shape plots
    tri = shape_desc[[c for c in ["molecule_id", "NPR1", "NPR2", "split", "cns_like"] if c in shape_desc.columns]].copy() if not shape_desc.empty else pd.DataFrame()
    tri.to_csv(data_dir / "shape_triangle.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    if not tri.empty:
        color_col = "cns_like" if "cns_like" in tri.columns else ("split" if "split" in tri.columns else None)
        if color_col:
            for grp, g in tri.groupby(color_col):
                ax.scatter(g["NPR1"], g["NPR2"], s=18, alpha=0.8, label=str(grp))
            ax.legend()
        else:
            ax.scatter(tri["NPR1"], tri["NPR2"], s=18, alpha=0.8)
    style_axis(ax, style, "Shape triangle", "NPR1", "NPR2")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_shape_triangle.svg"); plt.close(fig)

    shape_vs_p.to_csv(data_dir / "shape_vs_potency_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    if not shape_vs_p.empty and "pIC50" in shape_vs_p.columns:
        ax.scatter(shape_vs_p["NPR1"], shape_vs_p["pIC50"], s=18, alpha=0.8)
    style_axis(ax, style, "Shape vs potency", "NPR1", "pIC50")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_shape_vs_potency.svg"); plt.close(fig)

    shape_vs_r.to_csv(data_dir / "shape_vs_residuals_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    if not shape_vs_r.empty and "residual_abs" in shape_vs_r.columns:
        ax.scatter(shape_vs_r["NPR1"], shape_vs_r["residual_abs"], s=18, alpha=0.8)
    style_axis(ax, style, "Shape vs residuals", "NPR1", "|y-yhat|")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_shape_vs_residuals.svg"); plt.close(fig)

    # Atom attribution examples
    ap = atom_attr.groupby(["molecule_id", "atom_symbol"], as_index=False)["attribution"].mean().sort_values("attribution", ascending=False).head(30) if not atom_attr.empty else pd.DataFrame(columns=["molecule_id", "atom_symbol", "attribution"])
    ap.to_csv(data_dir / "attribution_examples.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not ap.empty:
        ax.bar(range(len(ap)), ap["attribution"])
        ax.set_xticks(range(len(ap)))
        ax.set_xticklabels(ap["atom_symbol"].tolist(), rotation=90)
    style_axis(ax, style, "Atom attribution examples", "Atoms", "Attribution")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_atom_attribution_examples.svg"); plt.close(fig)

    attr_stability.to_csv(data_dir / "attribution_stability.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    if not attr_stability.empty:
        labels = attr_stability["env_a"].astype(str) + " vs " + attr_stability["env_b"].astype(str)
        ax.barh(labels, attr_stability["correlation"])
    style_axis(ax, style, "z_inv attribution stability", "Correlation", "Env pair")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_zinv_attribution_stability.svg"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10 interpretability orchestrator")
    parser.add_argument("--target", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset_parquet", required=True)
    parser.add_argument("--bbb_parquet")
    parser.add_argument("--counterfactuals_parquet")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rgroup_series_min_n", type=int, default=8)
    parser.add_argument("--fragment_method", default="brics", choices=["brics", "murcko"])
    parser.add_argument("--mtc_method", default="bh", choices=["bh", "holm"])
    parser.add_argument("--shape_etkdg_confs", type=int, default=10)
    parser.add_argument("--shape_seed", type=int, default=42)
    parser.add_argument("--shape_select", default="lowest_uff_energy")
    parser.add_argument("--attribution_method", default="integrated_gradients")
    parser.add_argument("--attribution_target", default="z_inv")
    parser.add_argument("--max_counterfactuals", type=int, default=None)
    parser.add_argument("--max_compounds", type=int, default=None)
    parser.add_argument("--max_rgroup_compounds", type=int, default=500)
    parser.add_argument("--max_rgroup_per_series", type=int, default=100)
    parser.add_argument("--skip_attribution", action="store_true")
    parser.add_argument("--skip_shape", action="store_true")
    parser.add_argument("--skip_fragments", action="store_true")
    add_plot_style_args(parser)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    (outdir / "rgroup").mkdir(parents=True, exist_ok=True)
    (outdir / "fragments").mkdir(parents=True, exist_ok=True)
    (outdir / "shape").mkdir(parents=True, exist_ok=True)
    (outdir / "attribution").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "figure_data").mkdir(parents=True, exist_ok=True)
    (outdir / "provenance").mkdir(parents=True, exist_ok=True)

    _log_stage(f"Loading dataset parquet: {args.dataset_parquet}")
    dataset_cols = _available_columns(
        Path(args.dataset_parquet),
        [
            "smiles",
            "canonical_smiles",
            "molecule_id",
            "molecule_chembl_id",
            "compound_id",
            "mol_id",
            "id",
            "series_id",
            "env_id_manual",
            "assay_type",
            "split",
            "pIC50",
            "yhat",
            "cns_like",
        ],
    )
    df = pd.read_parquet(args.dataset_parquet, columns=dataset_cols if dataset_cols else None)
    smiles_col = _pick_col(df, ["smiles", "canonical_smiles"])
    mol_col = _pick_col(
        df,
        [
            "molecule_id",
            "molecule_chembl_id",
            "compound_id",
            "mol_id",
            "id",
        ],
        required=False,
    )
    series_col = _pick_col(df, ["series_id"], required=False)
    if series_col is None:
        df["series_id"] = "series_unknown"
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})
    if mol_col is None:
        df["molecule_id"] = df.index.astype(str)
    elif mol_col != "molecule_id":
        df = df.rename(columns={mol_col: "molecule_id"})

    if args.max_compounds is not None and args.max_compounds > 0 and len(df) > args.max_compounds:
        _log_stage(f"Applying --max_compounds={args.max_compounds} to dataset rows={len(df)}")
        df = df.head(args.max_compounds).copy()

    _log_stage("Loading predictions")
    preds = _load_predictions(run_dir)
    if not preds.empty:
        pmol = _pick_col(preds, ["molecule_id", "compound_id", "mol_id"])
        pyhat = _pick_prediction_col(preds)
        preds = preds.rename(columns={pmol: "molecule_id", pyhat: "yhat"})
        cols = [c for c in ["molecule_id", "yhat", "split"] if c in preds.columns]
        if "y" in preds.columns and "pIC50" not in df.columns:
            preds = preds.rename(columns={"y": "pIC50"})
            cols.append("pIC50")
        df = df.merge(preds[cols].drop_duplicates(subset=["molecule_id", "split"]), on="molecule_id", how="left")
    del preds
    gc.collect()

    _log_stage("Loading BBB annotations")
    bbb = None
    if args.bbb_parquet and Path(args.bbb_parquet).exists():
        bbb_path = Path(args.bbb_parquet)
        bbb_cols = _available_columns(bbb_path, ["molecule_id", "compound_id", "mol_id", "cns_like", "is_cns", "bbb_class"])
        bbb = pd.read_parquet(args.bbb_parquet, columns=bbb_cols if bbb_cols else None)
    if bbb is not None:
        bmol = _pick_col(bbb, ["molecule_id", "compound_id", "mol_id"])
        if bmol != "molecule_id":
            bbb = bbb.rename(columns={bmol: "molecule_id"})
        cns_col = _pick_col(bbb, ["cns_like", "is_cns", "bbb_class"], required=False)
        if cns_col:
            if cns_col == "bbb_class":
                bbb["cns_like"] = bbb[cns_col].astype(str).str.contains("cns|pass", case=False, regex=True).astype(int)
            elif cns_col != "cns_like":
                bbb = bbb.rename(columns={cns_col: "cns_like"})
        df = df.merge(bbb[[c for c in ["molecule_id", "cns_like"] if c in bbb.columns]].drop_duplicates("molecule_id"), on="molecule_id", how="left")

    counterfactuals = None
    if args.counterfactuals_parquet and Path(args.counterfactuals_parquet).exists():
        _log_stage(f"Loading counterfactuals parquet: {args.counterfactuals_parquet}")
        counter_path = Path(args.counterfactuals_parquet)
        counter_rows = _parquet_rows(counter_path)
        safe_limit = None
        if args.max_counterfactuals is not None and args.max_counterfactuals > 0:
            safe_limit = args.max_counterfactuals
        elif counter_rows is not None:
            if counter_rows > 200_000:
                safe_limit = 500
            elif counter_rows > 50_000:
                safe_limit = 1000
        if safe_limit is not None:
            _log_stage(f"Counterfactual safe-mode active: limiting to top {safe_limit} rows (estimated rows={counter_rows})")
        c_cols = _available_columns(counter_path, ["molecule_id", "compound_id", "mol_id", "smiles", "canonical_smiles", "score", "rank", "yhat", "delta", "is_hit"])
        counterfactuals = pd.read_parquet(counter_path, columns=c_cols if c_cols else None)
        if safe_limit is not None and len(counterfactuals) > safe_limit:
            rank_col = _pick_col(counterfactuals, ["rank"], required=False)
            if rank_col is not None:
                counterfactuals = counterfactuals.sort_values(rank_col, ascending=True).head(safe_limit).copy()
            else:
                counterfactuals = counterfactuals.head(safe_limit).copy()

    style = style_from_args(args)
    configure_matplotlib(style, svg=True)

    # R-group decomposition can spike memory usage for large tables.
    # Use a bounded copy of the dataframe for this stage only.
    rgroup_df = df
    if args.max_rgroup_per_series and args.max_rgroup_per_series > 0 and "series_id" in rgroup_df.columns:
        per_series = int(args.max_rgroup_per_series)
        rgroup_df = (
            rgroup_df.sort_values(["series_id", "molecule_id"]).groupby("series_id", sort=False, dropna=False).head(per_series).copy()
        )
        _log_stage(f"R-group per-series cap active: max_rgroup_per_series={per_series}, rows={len(rgroup_df)}")
    if args.max_rgroup_compounds and args.max_rgroup_compounds > 0 and len(rgroup_df) > args.max_rgroup_compounds:
        cap = int(args.max_rgroup_compounds)
        rgroup_df = rgroup_df.head(cap).copy()
        _log_stage(f"R-group global cap active: max_rgroup_compounds={cap}, rows={len(rgroup_df)}")

    _log_stage("Running R-group analysis")
    rgo = run_rgroup_analysis(rgroup_df, series_min_n=args.rgroup_series_min_n)
    del rgroup_df
    gc.collect()
    rgo.series_scaffolds.to_csv(outdir / "rgroup" / "series_scaffolds.csv", index=False)
    rgo.rgroup_table.to_parquet(outdir / "rgroup" / "rgroup_table.parquet", index=False)
    rgo.effects.to_csv(outdir / "rgroup" / "rgroup_effects.csv", index=False)
    rgo.env_interactions.to_csv(outdir / "rgroup" / "rgroup_env_interactions.csv", index=False)
    rgo.quality.to_csv(outdir / "rgroup" / "rgroup_quality_report.csv", index=False)

    if args.skip_fragments:
        _log_stage("Skipping fragment analysis (--skip_fragments)")
        fro = run_fragment_analysis(pd.DataFrame(columns=["molecule_id", "smiles"]), method=args.fragment_method)
    else:
        _log_stage("Running fragment analysis")
        fro = run_fragment_analysis(df, method=args.fragment_method)
    fro.fragment_library.to_parquet(outdir / "fragments" / "fragment_library.parquet", index=False)
    fro.fragment_enrichment.to_csv(outdir / "fragments" / "fragment_enrichment.csv", index=False)
    fro.functional_group_enrichment.to_csv(outdir / "fragments" / "functional_group_enrichment.csv", index=False)
    fro.quality_report.to_csv(outdir / "fragments" / "enrichment_quality_report.csv", index=False)
    del counterfactuals
    del bbb
    gc.collect()

    if args.skip_shape:
        _log_stage("Skipping shape analysis (--skip_shape)")
        sho = run_shape_analysis(pd.DataFrame(columns=["molecule_id", "smiles"]), n_confs=args.shape_etkdg_confs, seed=args.shape_seed, select=args.shape_select)
    else:
        _log_stage("Running shape analysis")
        sho = run_shape_analysis(df, n_confs=args.shape_etkdg_confs, seed=args.shape_seed, select=args.shape_select)
    sho.descriptors.to_parquet(outdir / "shape" / "shape_descriptors.parquet", index=False)
    sho.failures.to_csv(outdir / "shape" / "shape_failures.csv", index=False)
    sho.vs_potency.to_csv(outdir / "shape" / "shape_vs_potency.csv", index=False)
    sho.vs_residuals.to_csv(outdir / "shape" / "shape_vs_residuals.csv", index=False)
    gc.collect()

    if args.skip_attribution:
        _log_stage("Skipping attribution analysis (--skip_attribution)")
        ato = run_attribution_analysis(pd.DataFrame(columns=["molecule_id", "smiles"]))
    else:
        _log_stage("Running attribution analysis")
        ato = run_attribution_analysis(df, method=args.attribution_method, target=args.attribution_target, fragment_library=fro.fragment_library, rgroup_table=rgo.rgroup_table)
    ato.atom_attributions.to_parquet(outdir / "attribution" / "atom_attributions.parquet", index=False)
    ato.fragment_attributions.to_csv(outdir / "attribution" / "fragment_attributions.csv", index=False)
    ato.rgroup_attributions.to_csv(outdir / "attribution" / "rgroup_attributions.csv", index=False)
    ato.stability.to_csv(outdir / "attribution" / "attribution_stability_across_env.csv", index=False)

    _log_stage("Generating plots")

    _plot_all(outdir, style, rgo.effects, rgo.rgroup_table, fro.fragment_enrichment, fro.functional_group_enrichment, sho.descriptors, sho.vs_potency, sho.vs_residuals, ato.atom_attributions, ato.stability)

    gc.collect()

    # provenance
    _log_stage("Writing provenance")
    prov_dir = outdir / "provenance"
    run_cfg = vars(args)
    (prov_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    _save_env_txt(prov_dir / "environment.txt")

    script_paths = [SCRIPT_DIR / p for p in ["interpret_model.py", "rgroup_decompose.py", "fragment_analysis.py", "shape_analysis.py", "attribution_gnn.py", "plot_style.py", "stats_utils.py"]]
    checkpoint = run_dir / "checkpoints" / "best.pt"
    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": run_cfg,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "script_hashes": {p.name: _sha256(p) for p in script_paths if p.exists()},
        "dataset_hash": _sha256(Path(args.dataset_parquet)) if Path(args.dataset_parquet).exists() else None,
        "checkpoint_hash": _sha256(checkpoint) if checkpoint.exists() else None,
        "n_molecules_processed": int(len(df)),
        "n_series_processed": int((rgo.quality["status"] == "processed").sum()) if not rgo.quality.empty and "status" in rgo.quality.columns else 0,
        "etkdg": {"seed": args.shape_seed, "n_confs": args.shape_etkdg_confs, "shape_select": args.shape_select},
        "plotting": {"font": style.font_family, "bold_text": style.bold_text, "palette": list(style.palette), "font_title": style.font_title, "font_label": style.font_label, "font_tick": style.font_tick, "font_legend": style.font_legend, "svg.fonttype": "none"},
    }
    (prov_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
