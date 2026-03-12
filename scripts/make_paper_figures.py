#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None
from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble paper-ready figure pack from Step 8 evaluation outputs.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--outdir", required=True)
    add_plot_style_args(parser)
    args = parser.parse_args()
    if pd is None:
        raise SystemExit("pandas is required to run make_paper_figures.py")

    eval_dir = Path(args.eval_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    style = style_from_args(args)
    configure_matplotlib(style, svg=True)
    import matplotlib.pyplot as plt

    perf_split = pd.read_csv(eval_dir / "reports" / "performance_by_split.csv") if (eval_dir / "reports" / "performance_by_split.csv").exists() else pd.DataFrame()
    perf_env = pd.read_csv(eval_dir / "reports" / "performance_by_env.csv") if (eval_dir / "reports" / "performance_by_env.csv").exists() else pd.DataFrame()
    envprobe = pd.read_csv(eval_dir / "reports" / "causal_sanity_envprobe.csv") if (eval_dir / "reports" / "causal_sanity_envprobe.csv").exists() else pd.DataFrame()
    zinv = pd.read_csv(eval_dir / "reports" / "zinv_stability.csv") if (eval_dir / "reports" / "zinv_stability.csv").exists() else pd.DataFrame()
    cf = pd.read_csv(eval_dir / "reports" / "counterfactual_consistency.csv") if (eval_dir / "reports" / "counterfactual_consistency.csv").exists() else pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    if not perf_split.empty and "rmse" in perf_split.columns:
        agg = perf_split.groupby("split")["rmse"].mean().sort_values()
        ax.bar(agg.index.astype(str), agg.values)
        ax.tick_params(axis="x", rotation=50)
    style_axis(ax, style, "A) Performance across splits", "Split", "RMSE")

    ax = axes[0, 1]
    if not perf_env.empty and "rmse" in perf_env.columns:
        agg = perf_env.groupby("env")["rmse"].mean().sort_values()
        ax.bar(agg.index.astype(str), agg.values)
        ax.tick_params(axis="x", rotation=50)
    style_axis(ax, style, "B) Performance by environment", "Environment", "RMSE")

    ax = axes[1, 0]
    if not envprobe.empty:
        ax.bar(envprobe["embedding"], envprobe["accuracy"])
    style_axis(ax, style, "C) Env probe accuracy", "Embedding", "Accuracy")

    ax = axes[1, 1]
    if not zinv.empty:
        ax.bar(zinv["env"].astype(str), zinv.get("within_var", pd.Series([0] * len(zinv))))
        ax.tick_params(axis="x", rotation=50)
    style_axis(ax, style, "D) z_inv stability", "Environment", "Within variance")

    fig.tight_layout()
    fig.savefig(outdir / "paper_figure_main.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not cf.empty:
        top = cf.groupby("rule_id")["invariance_score"].mean().sort_values(ascending=False).head(12)
        ax.bar(top.index.astype(str), top.values)
        ax.tick_params(axis="x", rotation=55)
    style_axis(ax, style, "Counterfactual invariance summary", "Rule", "Invariance score")
    fig.tight_layout()
    fig.savefig(outdir / "paper_figure_counterfactuals.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
