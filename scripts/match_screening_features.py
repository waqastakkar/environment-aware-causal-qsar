#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from chemotype_cluster import cluster_chemotypes
from fragment_analysis import build_feature_presence, enrichment_hits_vs_background
from plot_style import add_plot_style_args, set_manuscript_style, style_axis, style_from_args
from screening_compat import load_screening_tables, resolve_step12_screen_outputs
from rgroup_transfer import decompose_with_core, transfer_rgroups
from scaffold_map import map_hits_to_training_scaffolds
from shape_analysis import run_shape_analysis
from stats_utils import ks_wasserstein


def _bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y"}


def _pick(df: pd.DataFrame, *cands: str, default: str | None = None) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return default


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mkdirs(outdir: Path) -> dict[str, Path]:
    d = {}
    for s in ["matched", "reports", "selections", "figures", "figure_data", "provenance"]:
        p = outdir / s
        p.mkdir(parents=True, exist_ok=True)
        d[s] = p
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 14 screening feature matching")
    p.add_argument("--target", required=True)
    p.add_argument("--screen_dir", required=True)
    p.add_argument("--screen_analysis_dir", default="")
    p.add_argument("--train_parquet", required=True)
    p.add_argument("--interpret_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--hits_source", choices=["top100_diverse", "ranked_topk", "file"], default="top100_diverse")
    p.add_argument("--hits_file", default="")
    p.add_argument("--hits_topk", type=int, default=100)
    p.add_argument("--shape_etkdg_confs", type=int, default=10)
    p.add_argument("--shape_seed", type=int, default=42)
    p.add_argument("--shape_select", default="lowest_uff_energy")
    p.add_argument("--fragment_method", choices=["brics", "murcko"], default="brics")
    p.add_argument("--rgroup_transfer", default="true")
    p.add_argument("--scaffold_match", choices=["exact", "similarity"], default="exact")
    p.add_argument("--scaffold_sim_threshold", type=float, default=0.7)
    p.add_argument("--chemotype_cluster_method", choices=["scaffold", "butina"], default="scaffold")
    p.add_argument("--cluster_threshold", type=float, default=0.6)
    p.add_argument("--library_sample_max", type=int, default=5000)
    add_plot_style_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    style = style_from_args(args)
    set_manuscript_style(style, svg=True)

    outdir = Path(args.outdir)
    dirs = _mkdirs(outdir)
    screen_dir = Path(args.screen_dir)
    interpret_dir = Path(args.interpret_dir)

    resolved = resolve_step12_screen_outputs(screen_dir)
    screen_dir = resolved["screen_dir"]
    _, ranked_all, ranked_cns_in = load_screening_tables(screen_dir)
    ranked = ranked_cns_in if len(ranked_cns_in) else ranked_all
    ranked_path = screen_dir / "ranking" / "ranked_cns_like_in_domain.parquet"
    if not ranked_path.exists():
        ranked_path = screen_dir / "ranking" / "ranked_all.parquet"
    id_col = _pick(ranked, "molecule_id", "compound_id", "id", default="molecule_id")
    smiles_col = _pick(ranked, "smiles", "SMILES", default="smiles")
    if id_col not in ranked.columns:
        ranked[id_col] = ranked.index.astype(str)

    hits: pd.DataFrame
    if args.hits_source == "top100_diverse" and args.screen_analysis_dir:
        p = Path(args.screen_analysis_dir) / "selections" / "top100_diverse.csv"
        if p.exists():
            sel = pd.read_csv(p)
            sid = _pick(sel, id_col, "molecule_id", "compound_id", "id", default=id_col)
            hits = ranked[ranked[id_col].astype(str).isin(set(sel[sid].astype(str)))].copy()
        else:
            hits = ranked.head(args.hits_topk).copy()
    elif args.hits_source == "file" and args.hits_file:
        sel = pd.read_csv(args.hits_file)
        sid = _pick(sel, id_col, "molecule_id", "compound_id", "id", default=id_col)
        hits = ranked[ranked[id_col].astype(str).isin(set(sel[sid].astype(str)))].copy()
    else:
        hits = ranked.head(args.hits_topk).copy()
    hits.to_parquet(dirs["matched"] / "hits_topk.parquet", index=False)

    # shape
    hit_shape = run_shape_analysis(hits, n_confs=args.shape_etkdg_confs, seed=args.shape_seed, select=args.shape_select, id_col=id_col, smiles_col=smiles_col).descriptors
    hit_shape.to_parquet(dirs["matched"] / "hits_with_shape.parquet", index=False)
    train_shape = pd.read_parquet(interpret_dir / "shape" / "shape_descriptors.parquet")
    library = pd.read_parquet(screen_dir / "processed" / "library_with_props.parquet") if (screen_dir / "processed" / "library_with_props.parquet").exists() else ranked
    lib_sample = library.sample(min(len(library), args.library_sample_max), random_state=args.shape_seed) if len(library) else library

    metrics = []
    for c in ["NPR1", "NPR2", "radius_gyration", "asphericity"]:
        if c in train_shape.columns and c in hit_shape.columns:
            ks, kp, w = ks_wasserstein(hit_shape[c], train_shape[c])
            metrics.append({"descriptor": c, "ks_stat": ks, "ks_p": kp, "wasserstein": w, "hits_mean": hit_shape[c].mean(), "train_mean": train_shape[c].mean()})
    if "shape_bin" in hit_shape.columns:
        for b, n in hit_shape["shape_bin"].value_counts().items():
            metrics.append({"descriptor": f"shape_bin_{b}", "ks_stat": np.nan, "ks_p": np.nan, "wasserstein": np.nan, "hits_mean": n / max(len(hit_shape), 1), "train_mean": np.nan})
    pd.DataFrame(metrics).to_csv(dirs["reports"] / "shape_shift_report.csv", index=False)

    tri = []
    for name, df in [("train", train_shape), ("library", lib_sample), ("hits", hit_shape)]:
        if {"NPR1", "NPR2"}.issubset(df.columns):
            tri.append(df[["NPR1", "NPR2"]].assign(dataset=name))
    tri_df = pd.concat(tri, ignore_index=True) if tri else pd.DataFrame(columns=["NPR1", "NPR2", "dataset"])
    tri_df.to_csv(dirs["figure_data"] / "shape_triangle_overlay.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    for ds, g in tri_df.groupby("dataset"):
        ax.scatter(g["NPR1"], g["NPR2"], s=12 if ds != "hits" else 30, alpha=0.6, label=ds)
    style_axis(ax, style, "Shape triangle overlay", "NPR1", "NPR2")
    ax.legend()
    fig.tight_layout(); fig.savefig(dirs["figures"] / "fig_shape_triangle_train_vs_hits.svg"); plt.close(fig)

    shift_plot = pd.DataFrame(metrics)
    shift_plot.to_csv(dirs["figure_data"] / "shape_shift_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); tmp = shift_plot[shift_plot["descriptor"].isin(["NPR1", "NPR2", "radius_gyration", "asphericity"])]
    ax.bar(tmp["descriptor"], tmp["wasserstein"])
    style_axis(ax, style, "Shape shift hits vs train", "Descriptor", "Wasserstein")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "fig_shape_shift_topk.svg"); plt.close(fig)

    # fragments
    hit_frag, hit_fg, _ = build_feature_presence(hits, id_col=id_col, smiles_col=smiles_col, method=args.fragment_method)
    bg_frag, bg_fg, _ = build_feature_presence(lib_sample, id_col=id_col, smiles_col=smiles_col, method=args.fragment_method)
    hit_frag.to_parquet(dirs["matched"] / "hits_with_fragments.parquet", index=False)
    hit_ids = hits[id_col].astype(str).tolist(); bg_ids = lib_sample[id_col].astype(str).tolist()
    frag_enr = enrichment_hits_vs_background(hit_frag, bg_frag, hit_ids, bg_ids, id_col=id_col)
    fg_enr = enrichment_hits_vs_background(hit_fg, bg_fg, hit_ids, bg_ids, id_col=id_col)
    frag_enr.rename(columns={"feature": "fragment_id"}).to_csv(dirs["reports"] / "fragment_enrichment_hits_vs_library.csv", index=False)
    fg_enr.rename(columns={"feature": "functional_group"}).to_csv(dirs["reports"] / "functional_group_enrichment_hits_vs_library.csv", index=False)
    frag_enr.head(30).to_csv(dirs["figure_data"] / "fragment_enrichment_hits_plot.csv", index=False)
    fg_enr.head(30).to_csv(dirs["figure_data"] / "functional_group_enrichment_hits_plot.csv", index=False)
    for dfp, figname, title, xcol in [
        (frag_enr, "fig_fragment_enrichment_hits.svg", "Fragment enrichment", "feature"),
        (fg_enr, "fig_functional_group_enrichment_hits.svg", "Functional-group enrichment", "feature"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4)); top = dfp.head(15)
        ax.barh(top[xcol], top["log_odds_ratio"]); ax.invert_yaxis()
        style_axis(ax, style, title, "log(odds ratio)", "Feature")
        fig.tight_layout(); fig.savefig(dirs["figures"] / figname); plt.close(fig)

    # zinv alignment
    zinv = pd.read_csv(interpret_dir / "attribution" / "fragment_attributions.csv")
    zcol = "mean_attribution" if "mean_attribution" in zinv.columns else _pick(zinv, "attribution", "score", default="mean_attribution")
    zagg = zinv.groupby("fragment", dropna=True)[zcol].mean().reset_index().rename(columns={"fragment": "feature", zcol: "zinv_attr_score"}) if "fragment" in zinv.columns else pd.DataFrame(columns=["feature", "zinv_attr_score"])
    overlap = frag_enr.merge(zagg, on="feature", how="left").rename(columns={"feature": "fragment_id"})
    overlap = overlap[[c for c in ["fragment_id", "log_odds_ratio", "q_value", "zinv_attr_score", "hit_prevalence"] if c in overlap.columns]]
    overlap.to_csv(dirs["reports"] / "fragment_overlap_with_zinv.csv", index=False)
    overlap.to_csv(dirs["figure_data"] / "fragment_overlap_zinv_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    if not overlap.empty:
        s = 40 + 300 * overlap["hit_prevalence"].fillna(0)
        ax.scatter(overlap["log_odds_ratio"], overlap["zinv_attr_score"], s=s, alpha=0.7)
        for _, r in overlap.sort_values(["q_value", "hit_prevalence"]).head(10).iterrows():
            ax.text(r["log_odds_ratio"], r["zinv_attr_score"], str(r["fragment_id"])[:15])
    style_axis(ax, style, "Fragment enrichment vs z_inv attribution", "log OR", "z_inv attribution")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "fig_overlap_fragments_zinv.svg"); plt.close(fig)

    # scaffold mapping and rgroup tier1
    series_scaf_path = interpret_dir / "rgroup" / "series_scaffolds.csv"
    train_scaffolds = pd.read_csv(series_scaf_path) if series_scaf_path.exists() else pd.DataFrame(columns=["series_id", "core_smiles"])
    mapping = map_hits_to_training_scaffolds(hits, train_scaffolds, mode=args.scaffold_match, sim_threshold=args.scaffold_sim_threshold, id_col=id_col, smiles_col=smiles_col)
    mapping.to_parquet(dirs["matched"] / "hits_scaffold_mapping.parquet", index=False)
    map_report = pd.DataFrame([{"n_hits": len(hits), "n_mapped": int(mapping.get("mapped", pd.Series(dtype=bool)).sum()) if not mapping.empty else 0, "mapping_rate": float(mapping.get("mapped", pd.Series(dtype=float)).mean()) if not mapping.empty else 0.0}])
    map_report.to_csv(dirs["reports"] / "scaffold_mapping_report.csv", index=False)
    mapping[[c for c in [id_col, "mapped", "matched_scaffold", "scaffold_similarity", "series_id"] if c in mapping.columns]].to_csv(dirs["figure_data"] / "scaffold_mapping_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4)); vals = map_report.iloc[0]
    ax.bar(["mapped", "unmapped"], [vals["n_mapped"], vals["n_hits"] - vals["n_mapped"]])
    style_axis(ax, style, "Scaffold mapping", "Class", "Count")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "fig_scaffold_mapping.svg"); plt.close(fig)

    if _bool(args.rgroup_transfer):
        mapped = mapping[mapping["mapped"]].copy() if not mapping.empty and "mapped" in mapping.columns else mapping.iloc[0:0].copy()
        rtab = transfer_rgroups(mapped, scaffold_col="matched_scaffold", smiles_col=smiles_col)
    else:
        rtab = pd.DataFrame()
    rtab.to_parquet(dirs["matched"] / "hits_rgroup_transfer.parquet", index=False)
    rcols = [c for c in rtab.columns if c.startswith("R")]
    rsum = pd.DataFrame([{"n_mapped_hits": len(rtab), "n_rgroups_columns": len(rcols), "n_unique_cores": int(rtab.get("core_scaffold", pd.Series(dtype=object)).nunique()) if not rtab.empty else 0}])
    rsum.to_csv(dirs["reports"] / "rgroup_transferability_report.csv", index=False)
    rplot = pd.DataFrame([{ "R_label": c, "n_non_null": int(rtab[c].notna().sum())} for c in rcols]) if rcols else pd.DataFrame(columns=["R_label", "n_non_null"])
    rplot.to_csv(dirs["figure_data"] / "rgroup_transfer_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not rplot.empty: ax.bar(rplot["R_label"], rplot["n_non_null"])
    style_axis(ax, style, "R-group transfer summary", "R-group", "Count")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "fig_rgroup_transfer_summary.svg"); plt.close(fig)

    # tier2 chemotypes
    unmapped = mapping[~mapping["mapped"]].copy() if (not mapping.empty and "mapped" in mapping.columns) else mapping.copy()
    chemotypes = cluster_chemotypes(unmapped, method=args.chemotype_cluster_method, threshold=args.cluster_threshold, smiles_col=smiles_col)
    chemotypes.to_parquet(dirs["matched"] / "hits_chemotype_clusters.parquet", index=False)
    csum = chemotypes.groupby("chemotype_cluster", dropna=False).agg(cluster_size=(id_col, "count"), mean_score=("score_mean", "mean") if "score_mean" in chemotypes.columns else (id_col, "count")).reset_index() if not chemotypes.empty else pd.DataFrame(columns=["chemotype_cluster", "cluster_size", "mean_score"])
    csum.to_csv(dirs["reports"] / "chemotype_summary.csv", index=False)
    chemotypes.sort_values("score_mean", ascending=False).groupby("chemotype_cluster").head(1).to_csv(dirs["selections"] / "chemotype_leads.csv", index=False)
    csum.head(30).to_csv(dirs["selections"] / "chemotype_panels.csv", index=False)

    # feature cards + selections
    cards = hits.copy()
    cards = cards.merge(hit_shape[[c for c in [id_col, "NPR1", "NPR2", "shape_bin"] if c in hit_shape.columns]], on=id_col, how="left") if not hit_shape.empty else cards
    if not rtab.empty:
        card_r = rtab[[c for c in [id_col, "series_id", "core_scaffold"] + [x for x in rtab.columns if x.startswith("R")] if c in rtab.columns]]
        cards = cards.merge(card_r, on=id_col, how="left")
    frag_map = hit_frag.groupby(id_col)["feature"].apply(lambda x: ";".join(list(pd.Series(x).dropna().astype(str).head(5)))).reset_index(name="top_enriched_fragments_present") if not hit_frag.empty else pd.DataFrame(columns=[id_col, "top_enriched_fragments_present"])
    cards = cards.merge(frag_map, on=id_col, how="left")
    keep = [id_col, "name", smiles_col, "inchikey", "score_mean", "uncertainty_std", "ad_distance", "cns_mpo", "MW", "logP", "TPSA", "HBD", "HBA", "RotB", "NPR1", "NPR2", "shape_bin", "top_enriched_fragments_present", "series_id", "core_scaffold"]
    keep += [c for c in cards.columns if c.startswith("R")]
    cards = cards[[c for c in keep if c in cards.columns]]
    cards.to_csv(dirs["reports"] / "top_hits_feature_cards.csv", index=False)
    cards.head(50).to_csv(dirs["selections"] / "top50_hits_with_features.csv", index=False)
    cards.head(100).to_csv(dirs["selections"] / "top100_hits_with_features.csv", index=False)
    cards.head(20).to_csv(dirs["figure_data"] / "feature_cards_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4));
    if "score_mean" in cards.columns:
        top20 = cards.head(20).reset_index(drop=True)
        ax.bar(top20.index.astype(str), top20["score_mean"])
    style_axis(ax, style, "Top-20 feature cards", "Hit rank", "score_mean")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "fig_feature_cards_top20.svg"); plt.close(fig)

    # provenance
    run_cfg = vars(args)
    (dirs["provenance"] / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    input_files = [ranked_path, Path(args.train_parquet), interpret_dir / "shape" / "shape_descriptors.parquet", interpret_dir / "attribution" / "fragment_attributions.csv", interpret_dir / "rgroup" / "series_scaffolds.csv"]
    prov = {
        "inputs": [{"path": str(p), "sha256": _hash_file(p)} for p in input_files if p.exists()],
        "counts": {"ranked_n": len(ranked), "hits_n": len(hits), "mapped_n": int(mapping.get("mapped", pd.Series(dtype=bool)).sum()) if not mapping.empty and "mapped" in mapping.columns else 0},
        "failure_stats": {"shape_failures": int(max(0, len(hits) - len(hit_shape)))},
        "python": platform.python_version(),
    }
    (dirs["provenance"] / "provenance.json").write_text(json.dumps(prov, indent=2))
    env = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True)
    (dirs["provenance"] / "environment.txt").write_text(env.stdout)


if __name__ == "__main__":
    main()
