#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd

from chemotype_summary import pick_chemotype_leads, summarize_chemotypes
from diversity import run_diversity_selection
from liability_flags import add_property_liability_flags
from novelty import compute_novelty
from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args
from screening_compat import load_screening_tables, resolve_step12_screen_outputs


SCORE_TABLE_CANDIDATES = (
    "predictions/scored_with_uncertainty.parquet",
    "predictions/scored_ensemble.parquet",
    "predictions/predictions_ensemble.parquet",
    "predictions/scored_single_model.parquet",
    "predictions/predictions_seed1.parquet",
    "ranking/ranked_all.parquet",
)


def _bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 13 screening analysis")
    p.add_argument("--target", required=True)
    p.add_argument("--screen_dir", required=True)
    p.add_argument("--train_parquet", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--diverse_k", type=int, default=100)
    p.add_argument("--cluster_method", default="butina", choices=["butina"])
    p.add_argument("--cluster_threshold", type=float, default=0.65)
    p.add_argument("--risk_control", default="true")
    p.add_argument("--score_threshold", type=float, default=7.0)
    p.add_argument("--uncertainty_threshold", type=float, default=0.25)
    p.add_argument("--ad_threshold", type=float, default=0.35)
    p.add_argument("--cns_mpo_threshold", type=float, default=4.0)
    p.add_argument("--triage_color_by", default="ad_distance", choices=["ad_distance", "cns_mpo"])
    p.add_argument("--ad_prefer", default="fingerprint")
    add_plot_style_args(p)
    return p.parse_args()


def _mkdirs(out: Path) -> dict[str, Path]:
    dirs = {}
    for sub in ["reports", "selections", "figures", "figure_data", "provenance"]:
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        dirs[sub] = d
    return dirs


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_col(df: pd.DataFrame, *cands: str) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _save_placeholder_svg(path: Path, text: str, style) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    style_axis(ax, style, title="Not available")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    style = style_from_args(args)
    configure_matplotlib(style, svg=True)
    import matplotlib.pyplot as plt

    screen_dir = Path(args.screen_dir)
    outdir = Path(args.outdir)
    dirs = _mkdirs(outdir)

    resolved = resolve_step12_screen_outputs(screen_dir)
    screen_dir = resolved["screen_dir"]
    score_path = next((screen_dir / rel for rel in SCORE_TABLE_CANDIDATES if (screen_dir / rel).exists()), None)
    if score_path is None:
        raise SystemExit(
            f"missing Step12 prediction table in {screen_dir}; expected one of: "
            f"{', '.join(SCORE_TABLE_CANDIDATES)}"
        )
    scored, ranked_all, ranked_cns_in = load_screening_tables(screen_dir)
    ranked_all = ranked_all.sort_values("canonical_score", ascending=False)
    ranked_cns_in = ranked_cns_in.sort_values("canonical_score", ascending=False)
    train = pd.read_parquet(args.train_parquet)

    smiles_col = _pick_col(scored, "smiles", "SMILES") or "smiles"
    id_col = _pick_col(scored, "compound_id", "id", "CompoundID") or "compound_id"
    if id_col not in scored.columns:
        scored[id_col] = scored.index.astype(str)

    ad_col = _pick_col(scored, "ad_distance", "ad_fp_distance", "ad_embedding_distance")
    cns_col = _pick_col(scored, "cns_mpo", "CNS_MPO")
    unc_col = _pick_col(scored, "uncertainty_std", "pred_std")
    if unc_col is None:
        scored["uncertainty_std"] = 0.0
        unc_col = "uncertainty_std"

    topk = ranked_all.head(args.topk).copy()
    cns_subset = ranked_cns_in.head(args.topk).copy()
    in_domain_subset = ranked_all[ranked_all[ad_col] <= args.ad_threshold].copy() if ad_col else ranked_all.copy()

    summary = pd.DataFrame([
        {"subset": "all", "n": len(scored), "score_mean": float(scored["score_mean"].mean()), "uncertainty_mean": float(scored[unc_col].mean())},
        {"subset": f"top{args.topk}", "n": len(topk), "score_mean": float(topk["score_mean"].mean()), "uncertainty_mean": float(topk[unc_col].mean())},
        {"subset": "cns_subset", "n": len(cns_subset), "score_mean": float(cns_subset["score_mean"].mean()), "uncertainty_mean": float(cns_subset[unc_col].mean())},
        {"subset": "in_domain", "n": len(in_domain_subset), "score_mean": float(in_domain_subset["score_mean"].mean()), "uncertainty_mean": float(in_domain_subset[unc_col].mean())},
    ])
    summary.to_csv(dirs["reports"] / "screening_summary.csv", index=False)

    novelty = compute_novelty(topk_df=topk, train_df=train, smiles_col=smiles_col, id_col=id_col)
    novelty.novelty_report.to_csv(dirs["reports"] / "novelty_report.csv", index=False)
    novelty.scaffold_novelty.to_csv(dirs["reports"] / "scaffold_novelty.csv", index=False)
    topk = novelty.topk_annotated

    diversity = run_diversity_selection(topk, smiles_col=smiles_col, threshold=args.cluster_threshold)
    diversity.clustering_summary.to_csv(dirs["reports"] / "clustering_summary.csv", index=False)
    diversity.diversity_selection.to_csv(dirs["reports"] / "diversity_selection.csv", index=False)

    top100_diverse = diversity.diversity_selection.head(100)
    top100_diverse.to_csv(dirs["selections"] / "top100_diverse.csv", index=False)
    top200_cns_diverse = diversity.diversity_selection
    if cns_col:
        top200_cns_diverse = top200_cns_diverse[top200_cns_diverse[cns_col] >= args.cns_mpo_threshold]
    top200_cns_diverse = top200_cns_diverse.head(200)
    top200_cns_diverse.to_csv(dirs["selections"] / "top200_cns_diverse.csv", index=False)

    flagged = add_property_liability_flags(scored)
    flagged.to_csv(dirs["reports"] / "property_liability_flags.csv", index=False)

    risk = scored.copy()
    if _bool(args.risk_control):
        cond = (risk["score_mean"] >= args.score_threshold) & (risk[unc_col] <= args.uncertainty_threshold)
        if ad_col:
            cond = cond & (risk[ad_col] <= args.ad_threshold)
        if cns_col:
            cond = cond & (risk[cns_col] >= args.cns_mpo_threshold)
        risk = risk[cond]
    risk = risk.sort_values(["score_mean", unc_col], ascending=[False, True])
    risk.to_csv(dirs["reports"] / "risk_controlled_selection.csv", index=False)
    risk.head(50).to_csv(dirs["selections"] / "top50_risk_controlled.csv", index=False)

    series = summarize_chemotypes(topk, id_col=id_col)
    series.to_csv(dirs["reports"] / "series_discovery.csv", index=False)
    chemotype_leads = pick_chemotype_leads(topk)
    chemotype_leads.to_csv(dirs["selections"] / "chemotype_leads.csv", index=False)

    # Figures + figure_data
    figd = dirs["figure_data"]
    figs = dirs["figures"]

    hit_score = scored[[id_col, "score_mean"]].copy()
    hit_score.to_csv(figd / "hit_score_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(hit_score["score_mean"].dropna(), bins=40, alpha=0.8)
    style_axis(ax, style, title="Hit score distribution", xlabel="score_mean", ylabel="Count")
    fig.tight_layout(); fig.savefig(figs / "fig_hit_score_distribution.svg"); plt.close(fig)

    hit_unc = scored[[id_col, unc_col]].rename(columns={unc_col: "uncertainty_std"})
    hit_unc.to_csv(figd / "hit_uncertainty_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(hit_unc["uncertainty_std"].dropna(), bins=40, alpha=0.8)
    style_axis(ax, style, title="Hit uncertainty distribution", xlabel="uncertainty_std", ylabel="Count")
    fig.tight_layout(); fig.savefig(figs / "fig_hit_uncertainty_distribution.svg"); plt.close(fig)

    pareto_su = scored[[id_col, "score_mean", unc_col]].rename(columns={unc_col: "uncertainty_std"})
    pareto_su.to_csv(figd / "pareto_score_uncertainty.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(pareto_su["score_mean"], pareto_su["uncertainty_std"], s=12, alpha=0.6)
    style_axis(ax, style, title="Pareto: score vs uncertainty", xlabel="score_mean", ylabel="uncertainty_std")
    fig.tight_layout(); fig.savefig(figs / "fig_pareto_score_uncertainty.svg"); plt.close(fig)

    pareto_sc = scored[[id_col, "score_mean"] + ([cns_col] if cns_col else [])].copy()
    if cns_col:
        pareto_sc = pareto_sc.rename(columns={cns_col: "cns_mpo"})
        fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(pareto_sc["score_mean"], pareto_sc["cns_mpo"], s=12, alpha=0.6)
        style_axis(ax, style, title="Pareto: score vs CNS", xlabel="score_mean", ylabel="cns_mpo")
        fig.tight_layout(); fig.savefig(figs / "fig_pareto_score_cns.svg"); plt.close(fig)
    else:
        pareto_sc["cns_mpo"] = pd.NA
        _save_placeholder_svg(figs / "fig_pareto_score_cns.svg", "cns_mpo unavailable", style)
    pareto_sc.to_csv(figd / "pareto_score_cns.csv", index=False)

    score_ad = scored[[id_col, "score_mean"] + ([ad_col] if ad_col else [])].copy()
    if ad_col:
        score_ad = score_ad.rename(columns={ad_col: "ad_distance"})
        fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(score_ad["score_mean"], score_ad["ad_distance"], s=12, alpha=0.6)
        style_axis(ax, style, title="Score vs AD", xlabel="score_mean", ylabel="ad_distance")
        fig.tight_layout(); fig.savefig(figs / "fig_score_vs_ad.svg"); plt.close(fig)
    else:
        score_ad["ad_distance"] = pd.NA
        _save_placeholder_svg(figs / "fig_score_vs_ad.svg", "ad_distance unavailable", style)
    score_ad.to_csv(figd / "score_vs_ad.csv", index=False)

    scaf_plot = novelty.scaffold_novelty[["scaffold", "count", "seen_in_train"]].copy()
    scaf_plot.to_csv(figd / "scaffold_novelty_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    tmp = scaf_plot.head(20)
    colors = [style.palette[3] if x else style.palette[1] for x in tmp["seen_in_train"].fillna(False)]
    ax.bar(range(len(tmp)), tmp["count"], color=colors)
    style_axis(ax, style, title="Scaffold novelty (top 20)", xlabel="Scaffold rank", ylabel="Count")
    fig.tight_layout(); fig.savefig(figs / "fig_scaffold_novelty.svg"); plt.close(fig)

    cl_plot = diversity.clustering_summary[["cluster_id", "cluster_size"]].copy()
    cl_plot.to_csv(figd / "cluster_sizes_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(cl_plot["cluster_size"], bins=30, alpha=0.8)
    style_axis(ax, style, title="Cluster sizes", xlabel="cluster_size", ylabel="Count")
    fig.tight_layout(); fig.savefig(figs / "fig_cluster_sizes.svg"); plt.close(fig)

    trade = diversity.diversity_selection[[id_col, "score_mean", "cluster_size"]].copy()
    trade.to_csv(figd / "diversity_tradeoff_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(trade["cluster_size"], trade["score_mean"], s=15, alpha=0.7)
    style_axis(ax, style, title="Diversity tradeoff", xlabel="cluster_size", ylabel="score_mean")
    fig.tight_layout(); fig.savefig(figs / "fig_diversity_tradeoff.svg"); plt.close(fig)

    prop_cols = [c for c in ["MW", "TPSA", "HBD", "cLogP"] if c in scored.columns]
    prop_plot = pd.concat([
        scored[prop_cols].assign(set="all") if prop_cols else pd.DataFrame(),
        topk[prop_cols].assign(set=f"top{args.topk}") if prop_cols else pd.DataFrame(),
    ], ignore_index=True)
    prop_plot.to_csv(figd / "property_distributions_topk.csv", index=False)
    if prop_cols:
        fig, axes = plt.subplots(1, len(prop_cols), figsize=(4 * len(prop_cols), 4))
        axes = [axes] if len(prop_cols) == 1 else axes
        for ax, col in zip(axes, prop_cols):
            ax.hist(scored[col].dropna(), bins=30, alpha=0.5, label="all")
            ax.hist(topk[col].dropna(), bins=30, alpha=0.5, label=f"top{args.topk}")
            style_axis(ax, style, title=col, xlabel=col, ylabel="Count")
            ax.legend()
        fig.tight_layout(); fig.savefig(figs / "fig_property_distributions_topk.svg"); plt.close(fig)
    else:
        _save_placeholder_svg(figs / "fig_property_distributions_topk.svg", "No property columns found", style)

    triage = scored[[id_col] + (["inchikey"] if "inchikey" in scored.columns else []) + ["score_mean", unc_col] + ([ad_col] if ad_col else []) + ([cns_col] if cns_col else [])].copy()
    if "inchikey" not in triage.columns:
        triage["inchikey"] = pd.NA
    triage = triage.rename(columns={unc_col: "uncertainty_std", ad_col: "ad_distance" if ad_col else "", cns_col: "cns_mpo" if cns_col else ""})
    if "ad_distance" not in triage.columns:
        triage["ad_distance"] = pd.NA
    if "cns_mpo" not in triage.columns:
        triage["cns_mpo"] = pd.NA
    top50_ids = set(risk.head(50)[id_col].astype(str))
    triage["selected_flag"] = triage[id_col].astype(str).isin(top50_ids)
    triage.to_csv(figd / "triage_score_uncertainty_ad.csv", index=False)

    if triage["ad_distance"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5))
        color_vals = triage["ad_distance"] if args.triage_color_by == "ad_distance" else triage["cns_mpo"]
        size = triage["uncertainty_std"].fillna(0.0)
        s = 20 + 180 * (size - size.min()) / (size.max() - size.min() + 1e-9)
        sc = ax.scatter(triage["score_mean"], triage["ad_distance"], c=color_vals, s=s, alpha=0.7, cmap="viridis")
        ax.axvline(args.score_threshold, linestyle="--", color=style.palette[3], linewidth=1.2)
        ax.axhline(args.ad_threshold, linestyle="--", color=style.palette[0], linewidth=1.2)
        cb = fig.colorbar(sc, ax=ax); cb.set_label(args.triage_color_by)
        style_axis(ax, style, title="Triage: score / uncertainty / AD", xlabel="score_mean", ylabel="ad_distance")
        fig.tight_layout(); fig.savefig(figs / "fig_triage_score_uncertainty_ad.svg"); plt.close(fig)
    else:
        _save_placeholder_svg(figs / "fig_triage_score_uncertainty_ad.svg", "ad_distance unavailable", style)

    run_cfg = vars(args)
    (dirs["provenance"] / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    prov = {
        "target": args.target,
        "inputs": {
            "screen_scored": str(score_path),
            "screen_scored_sha256": _hash(score_path),
            "train_parquet": str(args.train_parquet),
            "train_parquet_sha256": _hash(Path(args.train_parquet)),
        },
        "counts": {"scored": len(scored), "topk": len(topk), "risk": len(risk), "diverse": len(top100_diverse)},
        "params": run_cfg,
    }
    (dirs["provenance"] / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    env_txt = f"python={platform.python_version()}\nplatform={platform.platform()}\n\n"
    env_txt += subprocess.getoutput("python -m pip freeze") + "\n"
    (dirs["provenance"] / "environment.txt").write_text(env_txt, encoding="utf-8")


if __name__ == "__main__":
    main()
