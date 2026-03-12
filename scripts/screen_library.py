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

from applicability_domain import build_or_load_train_fingerprint_index, fingerprint_ad
from bbb_rules import add_bbb_metrics
from featurize import featurize_library
from infer import run_inference
from library_clean import clean_library
from library_io import parse_library
from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args
from property_calc import compute_properties
from screening_reports import build_rankings


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).lower() in {"1", "true", "yes", "y"}


def _mkdirs(root: Path):
    for s in ["input", "processed", "predictions", "ranking", "figures", "figure_data", "provenance", "artifacts"]:
        (root / s).mkdir(parents=True, exist_ok=True)


def _write_screen_feature_schema(run_dirs: list[Path], out: Path) -> Path:
    schema_paths = [rd / "artifacts" / "feature_schema.json" for rd in run_dirs]
    missing = [str(p) for p in schema_paths if not p.exists()]
    if missing:
        raise SystemExit(
            "missing required feature schema in run_dir artifacts: "
            + ", ".join(missing)
        )

    schema_hashes = [_sha256(p) for p in schema_paths]
    if len(set(schema_hashes)) != 1:
        joined = ", ".join(f"{p}:{h}" for p, h in zip(schema_paths, schema_hashes))
        raise SystemExit(
            "cannot screen with mixed feature schemas across run_dirs; rerun Step 06 with a consistent schema. "
            f"Resolved schema hashes: {joined}"
        )

    dest = out / "artifacts" / "feature_schema.json"
    dest.write_text(schema_paths[0].read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def parse_args():
    p = argparse.ArgumentParser(description="Step 12b: screen prepared library")
    p.add_argument("--target", required=True)
    p.add_argument("--run_dir", default=None)
    p.add_argument("--run_dirs", nargs="+", default=None)
    p.add_argument("--runs_root", default=None)
    p.add_argument("--prepared_library_path", default=None)
    p.add_argument("--input_path", default=None)
    p.add_argument("--input_format", choices=["smi", "csv"], default="csv")
    p.add_argument("--outdir", required=True)
    p.add_argument("--screen_id", default=None)
    p.add_argument("--smi_layout", choices=["smiles_id", "smiles_name_id", "smiles_only"], default="smiles_id")
    p.add_argument("--header", default="auto")
    p.add_argument("--comment_prefix", default="#")
    p.add_argument("--allow_cpp_comments", default="true")
    p.add_argument("--name_is_rest", default="true")
    p.add_argument("--smi_quoted_name", default="false")
    p.add_argument("--sep", default=",")
    p.add_argument("--quotechar", default='"')
    p.add_argument("--smiles_col")
    p.add_argument("--id_col")
    p.add_argument("--name_col")
    p.add_argument("--use_ensemble_manifest")
    p.add_argument("--compute_bbb", default="true")
    p.add_argument("--cns_mpo_threshold", type=float, default=4.0)
    p.add_argument("--compute_ad", default="true")
    p.add_argument("--ad_mode", default="fingerprint", choices=["fingerprint", "embedding", "fingerprint+embedding"])
    p.add_argument("--ad_threshold", type=float, default=None)
    p.add_argument("--topk", type=int, default=500)
    add_plot_style_args(p)
    args = p.parse_args()
    if args.run_dir and (args.run_dirs or args.runs_root):
        raise SystemExit("--run_dir cannot be combined with --run_dirs/--runs_root")
    if args.run_dirs and args.runs_root:
        raise SystemExit("--run_dirs cannot be combined with --runs_root")
    if not args.run_dir and not args.run_dirs and not args.runs_root:
        raise SystemExit("one of --run_dir, --run_dirs, or --runs_root is required")
    return args


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        candidates = [Path(args.run_dir)]
    elif args.run_dirs:
        candidates = [Path(p) for p in args.run_dirs]
    else:
        root = Path(args.runs_root)
        if not root.exists():
            raise SystemExit(f"runs_root does not exist: {root}")
        candidates = [p for p in sorted(root.iterdir()) if p.is_dir()]

    run_dirs: list[Path] = []
    seen: set[str] = set()
    for rd in candidates:
        r = rd.resolve()
        key = str(r)
        if key in seen:
            continue
        if not r.is_dir():
            raise SystemExit(f"run_dir is not a directory: {r}")
        fs = r / "artifacts" / "feature_schema.json"
        if not fs.exists():
            raise SystemExit(f"missing required feature schema for screening: {fs}")
        ckpt = r / "checkpoints" / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing required checkpoint for screening: {ckpt}")
        seen.add(key)
        run_dirs.append(r)
    if not run_dirs:
        raise SystemExit("no valid run directories resolved for screening")
    return run_dirs




def _normalize_prepared_library_smiles(df: pd.DataFrame, source_path: str | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    if "canonical_smiles" not in out.columns:
        raise SystemExit("prepared library is missing required column: canonical_smiles")

    initial_rows = len(out)
    null_mask = out["canonical_smiles"].isna()
    out = out.loc[~null_mask].copy()
    out["canonical_smiles"] = out["canonical_smiles"].astype(str).str.strip()

    header_like_mask = out["canonical_smiles"].str.lower().eq("canonical_smiles") | out["canonical_smiles"].str.lower().eq("smiles")
    empty_mask = out["canonical_smiles"].eq("")

    removed_null = int(null_mask.sum())
    removed_header = int(header_like_mask.sum())
    removed_empty = int(empty_mask.sum())

    out = out.loc[~(header_like_mask | empty_mask)].copy()

    if source_path and str(source_path).lower().endswith(".csv") and removed_header > 0:
        raise SystemExit(
            "prepared library CSV appears malformed: header row values were retained in canonical_smiles. "
            "Regenerate the prepared library with correct CSV header handling before screening."
        )

    if out.empty:
        src = f" from {source_path}" if source_path else ""
        raise SystemExit(
            "prepared library validation removed all rows; check CSV header handling and smiles/canonical_smiles column formatting"
            f"{src}"
        )

    report = {
        "prepared_rows_input": int(initial_rows),
        "prepared_rows_removed_null_smiles": removed_null,
        "prepared_rows_removed_header_like": removed_header,
        "prepared_rows_removed_empty_smiles": removed_empty,
        "prepared_rows_after_smiles_validation": int(len(out)),
    }
    return out, report
def _load_deduplicated_library(args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if args.prepared_library_path:
        p = Path(args.prepared_library_path)
        if not p.exists():
            raise SystemExit(f"missing required prepared library file: {p}")
        dedup = pd.read_parquet(p)
        dedup, prep_report = _normalize_prepared_library_smiles(dedup, source_path=str(p))
        report = {"clean_dedup_rows": int(len(dedup)), "prepared_input_mode": 1, **prep_report}
        return dedup, report

    if not args.input_path:
        raise SystemExit("either --prepared_library_path or --input_path must be provided")

    parsed, manifest = parse_library(
        input_path=args.input_path,
        input_format=args.input_format,
        smi_layout=args.smi_layout,
        header=args.header,
        comment_prefix=args.comment_prefix,
        allow_cpp_comments=_bool(args.allow_cpp_comments),
        name_is_rest=_bool(args.name_is_rest),
        smi_quoted_name=_bool(args.smi_quoted_name),
        sep=args.sep,
        quotechar=args.quotechar,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        name_col=args.name_col,
    )
    parsed.to_parquet(out / "processed/library_raw_parsed.parquet", index=False)
    pd.DataFrame([manifest]).to_csv(out / "input/library_manifest.csv", index=False)
    (out / "input/input_fingerprint.json").write_text(json.dumps({"sha256": _sha256(Path(args.input_path))}, indent=2))

    clean, dedup, clean_report = clean_library(parsed)
    clean.to_parquet(out / "processed/library_clean.parquet", index=False)
    dedup.to_parquet(out / "processed/library_dedup.parquet", index=False)
    return dedup, clean_report


def main():
    args = parse_args()
    run_dirs = _resolve_run_dirs(args)
    run_dir = run_dirs[0]
    screen_id = args.screen_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.outdir) if Path(args.outdir).name == screen_id else Path(args.outdir) / args.target / screen_id
    _mkdirs(out)
    _write_screen_feature_schema(run_dirs, out)

    style = style_from_args(args)
    configure_matplotlib(style, svg=True)

    dedup, clean_report = _load_deduplicated_library(args, out)
    if "prepared_rows_removed_header_like" in clean_report:
        print(
            "Prepared-library smiles validation: "
            f"removed_null={clean_report.get('prepared_rows_removed_null_smiles', 0)}, "
            f"removed_header_like={clean_report.get('prepared_rows_removed_header_like', 0)}, "
            f"removed_empty={clean_report.get('prepared_rows_removed_empty_smiles', 0)}, "
            f"remaining={clean_report.get('prepared_rows_after_smiles_validation', len(dedup))}"
        )

    with_props = compute_properties(dedup, smiles_col="canonical_smiles")
    if _bool(args.compute_bbb):
        with_props = add_bbb_metrics(with_props, cns_mpo_threshold=args.cns_mpo_threshold)
    with_props.to_parquet(out / "processed/library_with_props.parquet", index=False)

    graphs, feat_df, feat_report = featurize_library(with_props, run_dir)

    seed_frames: list[pd.DataFrame] = []
    for idx, rd in enumerate(run_dirs, start=1):
        single, ens = run_inference(graphs, rd, ensemble_manifest=args.use_ensemble_manifest)
        pred_col = f"prediction_seed{idx}"
        seed_pred = ens[["compound_id", "score_mean"]].rename(columns={"score_mean": pred_col})
        seed_frames.append(seed_pred)
        per_seed = single.rename(columns={"yhat_pIC50": "prediction"})
        per_seed.insert(1, "model_label", f"seed{idx}")
        per_seed.insert(2, "run_dir", str(rd))
        per_seed.to_parquet(out / f"predictions/predictions_seed{idx}.parquet", index=False)
        per_seed.to_csv(out / f"predictions/predictions_seed{idx}.csv", index=False)

    ens_all = seed_frames[0]
    for sf in seed_frames[1:]:
        ens_all = ens_all.merge(sf, on="compound_id", how="inner")

    pred_cols = [c for c in ens_all.columns if c.startswith("prediction_seed")]
    ens_all["prediction_mean"] = ens_all[pred_cols].mean(axis=1)
    ens_all["prediction_std"] = ens_all[pred_cols].std(axis=1).fillna(0.0)
    ens_all["n_models"] = len(pred_cols)
    ens_all["score_mean"] = ens_all["prediction_mean"]
    ens_all["score_std"] = ens_all["prediction_std"]
    ens_all.to_parquet(out / "predictions/predictions_ensemble.parquet", index=False)
    ens_all.to_csv(out / "predictions/predictions_ensemble.csv", index=False)
    ens_all.to_parquet(out / "predictions/scored_ensemble.parquet", index=False)

    if len(run_dirs) == 1:
        single_out = pd.read_parquet(out / "predictions/predictions_seed1.parquet")[["compound_id", "prediction"]].rename(columns={"prediction": "yhat_pIC50"})
        single_out.to_parquet(out / "predictions/scored_single_model.parquet", index=False)

    scored = feat_df.merge(ens_all[["compound_id", "prediction_mean", "prediction_std", "n_models", "score_mean", "score_std", *pred_cols]], on="compound_id", how="inner")
    ad_invalid_smiles_rows = 0
    if _bool(args.compute_ad) and "fingerprint" in args.ad_mode:
        if "canonical_smiles" not in scored.columns:
            raise SystemExit("missing required column for fingerprint AD: canonical_smiles")
        ad_input = scored[["compound_id", "canonical_smiles"]].rename(columns={"compound_id": "molecule_id", "canonical_smiles": "smiles"}).copy()
        ad_input["smiles"] = ad_input["smiles"].astype(str).str.strip()
        invalid_ad_mask = ad_input["smiles"].eq("") | ad_input["smiles"].str.lower().eq("smiles")
        ad_invalid_smiles_rows = int(invalid_ad_mask.sum())
        idx = build_or_load_train_fingerprint_index(run_dir)
        ad = fingerprint_ad(idx, ad_input)
        scored = scored.merge(ad.rename(columns={"molecule_id": "compound_id", "ad_distance_fingerprint": "ad_distance"}), on="compound_id", how="left")
    if "ad_distance" not in scored.columns:
        scored["ad_distance"] = np.nan
    scored.to_parquet(out / "predictions/scored_with_uncertainty.parquet", index=False)

    ranks, sel = build_rankings(scored, args.cns_mpo_threshold, args.ad_threshold, args.topk)
    for k in ["ranked_all", "ranked_cns_like", "ranked_in_domain", "ranked_cns_like_in_domain"]:
        ranks[k].to_parquet(out / f"ranking/{k}.parquet", index=False)
    ranks["best"].head(100).to_csv(out / "ranking/top_100.csv", index=False)
    ranks["best"].head(500).to_csv(out / "ranking/top_500.csv", index=False)
    sel.to_csv(out / "ranking/selection_report.csv", index=False)

    report = {**clean_report, **feat_report, "ad_invalid_smiles_rows": int(ad_invalid_smiles_rows), "final_scored_count": int(len(scored))}
    pd.DataFrame([{"metric": k, "value": v} for k, v in report.items()]).to_csv(out / "processed/featurization_report.csv", index=False)

    s = scored[["score_mean"]].dropna(); s.to_csv(out / "figure_data/score_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(s["score_mean"], bins=30); style_axis(ax, style, "Score distribution", "score_mean", "Count"); fig.tight_layout(); fig.savefig(out / "figures/fig_score_distribution.svg"); plt.close(fig)

    u = scored[["score_std"]].dropna(); u.to_csv(out / "figure_data/uncertainty_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(u["score_std"], bins=30); style_axis(ax, style, "Uncertainty distribution", "score_std", "Count"); fig.tight_layout(); fig.savefig(out / "figures/fig_uncertainty_distribution.svg"); plt.close(fig)

    p = scored[["score_mean", "cns_mpo"]].dropna() if "cns_mpo" in scored.columns else pd.DataFrame(columns=["score_mean", "cns_mpo"])
    p.to_csv(out / "figure_data/pareto_score_vs_cns.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not p.empty: ax.scatter(p["score_mean"], p["cns_mpo"], s=10, alpha=0.6)
    style_axis(ax, style, "Score vs CNS MPO", "score_mean", "cns_mpo"); fig.tight_layout(); fig.savefig(out / "figures/fig_pareto_score_vs_cns.svg"); plt.close(fig)

    a = scored[["score_mean", "ad_distance"]].dropna()
    a.to_csv(out / "figure_data/score_vs_ad.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not a.empty: ax.scatter(a["ad_distance"], a["score_mean"], s=10, alpha=0.6)
    style_axis(ax, style, "Score vs AD", "ad_distance", "score_mean"); fig.tight_layout(); fig.savefig(out / "figures/fig_score_vs_ad.svg"); plt.close(fig)

    props = [c for c in ["MW", "LogP", "TPSA"] if c in scored.columns]
    top = ranks["best"].head(args.topk)
    box = pd.concat([scored[props].assign(group="all"), top[props].assign(group=f"top_{args.topk}")], ignore_index=True) if props else pd.DataFrame()
    box.to_csv(out / "figure_data/topk_property_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if props and not box.empty:
        ax.boxplot([box[box["group"] == "all"][props[0]].dropna(), box[box["group"] != "all"][props[0]].dropna()], labels=["all", f"top_{args.topk}"])
        ylabel = props[0]
    else:
        ylabel = "value"
    style_axis(ax, style, "TopK property summary", "Group", ylabel); fig.tight_layout(); fig.savefig(out / "figures/fig_topk_property_summary.svg"); plt.close(fig)

    cfg = vars(args)
    (out / "provenance/run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    input_hash = None
    if args.input_path:
        input_hash = _sha256(Path(args.input_path))
    elif args.prepared_library_path:
        input_hash = _sha256(Path(args.prepared_library_path))
    run_artifact_hashes = {}
    checkpoint_hashes = {}
    for rd in run_dirs:
        schema = rd / "artifacts/feature_schema.json"
        resolved_cfg = rd / "configs/resolved_config.yaml"
        ckpt = rd / "checkpoints/best.pt"
        if schema.exists():
            run_artifact_hashes[f"{rd}/artifacts/feature_schema.json"] = _sha256(schema)
        if resolved_cfg.exists():
            run_artifact_hashes[f"{rd}/configs/resolved_config.yaml"] = _sha256(resolved_cfg)
        if ckpt.exists():
            checkpoint_hashes[str(ckpt)] = _sha256(ckpt)

    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": ([f"fingerprint_ad_invalid_smiles_rows={ad_invalid_smiles_rows}"] if ad_invalid_smiles_rows else []),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_hash": input_hash,
        "run_dirs": [str(rd) for rd in run_dirs],
        "run_artifact_hashes": run_artifact_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "script_hashes": {str(Path(__file__).name): _sha256(Path(__file__))},
    }
    (out / "provenance/provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (out / "provenance/environment.txt").write_text(subprocess.check_output(["pip", "freeze"], text=True), encoding="utf-8")


if __name__ == "__main__":
    main()
