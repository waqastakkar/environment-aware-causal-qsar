#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from plot_style import PlotStyle, configure_matplotlib, parse_palette, style_axis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--dataset_parquet", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--split_names", required=True, help="comma-separated")
    p.add_argument("--outdir", required=True)
    p.add_argument("--task", required=True, choices=["regression", "classification"])
    p.add_argument("--label_col", required=True)
    p.add_argument("--env_col", required=True)
    p.add_argument("--seeds", default="42")
    p.add_argument("--ablations", default="full,no_adv,no_irm,no_dis")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--early_stopping_patience", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_rows", type=int, default=0, help="Optional cap on rows loaded from dataset_parquet (0 disables).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder", default="gine")
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--z_inv_dim", type=int, default=64)
    p.add_argument("--z_spu_dim", type=int, default=64)
    p.add_argument("--bbb_parquet", default=None)
    p.add_argument("--svg", action="store_true")
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--bold_text", action="store_true")
    p.add_argument("--palette", default="nature5")
    p.add_argument("--font_title", type=int, default=16)
    p.add_argument("--font_label", type=int, default=14)
    p.add_argument("--font_tick", type=int, default=12)
    p.add_argument("--font_legend", type=int, default=12)
    return p.parse_args()


def ablation_weights(name: str):
    if name == "full":
        return 0.5, 0.1, 0.1
    if name == "no_adv":
        return 0.0, 0.1, 0.1
    if name == "no_irm":
        return 0.5, 0.0, 0.1
    if name == "no_dis":
        return 0.5, 0.1, 0.0
    raise ValueError(name)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.split_names.split(",")]
    ablations = [a.strip() for a in args.ablations.split(",")]

    rows = []
    for split in splits:
        for seed in seeds:
            for ab in ablations:
                l_adv, l_irm, l_dis = ablation_weights(ab)
                run_id = f"bench_{split}_{ab}_seed{seed}"
                cmd = [
                    "python",
                    "scripts/train_causal_qsar.py",
                    "--target", args.target,
                    "--dataset_parquet", args.dataset_parquet,
                    "--splits_dir", args.splits_dir,
                    "--split_name", split,
                    "--outdir", args.outdir,
                    "--task", args.task,
                    "--label_col", args.label_col,
                    "--env_col", args.env_col,
                    "--encoder", args.encoder,
                    "--z_dim", str(args.z_dim),
                    "--z_inv_dim", str(args.z_inv_dim),
                    "--z_spu_dim", str(args.z_spu_dim),
                    "--lambda_adv", str(l_adv),
                    "--lambda_irm", str(l_irm),
                    "--lambda_dis", str(l_dis),
                    "--epochs", str(args.epochs),
                    "--early_stopping_patience", str(args.early_stopping_patience),
                    "--batch_size", str(args.batch_size),
                    "--max_rows", str(args.max_rows),
                    "--lr", str(args.lr),
                    "--seed", str(seed),
                    "--run_id", run_id,
                    "--font", args.font,
                    "--palette", args.palette,
                    "--font_title", str(args.font_title),
                    "--font_label", str(args.font_label),
                    "--font_tick", str(args.font_tick),
                    "--font_legend", str(args.font_legend),
                ]
                if args.svg:
                    cmd.append("--svg")
                if args.bold_text:
                    cmd.append("--bold_text")
                if args.bbb_parquet:
                    cmd += ["--bbb_parquet", args.bbb_parquet]
                subprocess.run(cmd, check=True)

                run_root = Path(args.outdir) / args.target / split / run_id
                ms = pd.read_csv(run_root / "reports/metrics_summary.csv").iloc[0].to_dict()
                row = {"split_name": split, "seed": seed, "ablation": ab, **ms}
                rows.append(row)

    all_df = pd.DataFrame(rows)
    out = Path(args.outdir) / args.target / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "ablation_table.csv", index=False)

    style = PlotStyle(
        font_family=args.font,
        font_title=args.font_title,
        font_label=args.font_label,
        font_tick=args.font_tick,
        font_legend=args.font_legend,
        bold_text=args.bold_text,
        palette=parse_palette(args.palette),
    )
    configure_matplotlib(style, svg=args.svg)

    fig, ax = plt.subplots(figsize=(8, 4))
    metric = "rmse" if args.task == "regression" else "auc"
    piv = all_df.groupby(["ablation"])[metric].mean().reset_index()
    ax.bar(piv["ablation"], piv[metric])
    style_axis(ax, style, "Ablation Summary", "Ablation", metric)
    fig.tight_layout(); fig.savefig(out / "fig_ablation_summary.svg")


if __name__ == "__main__":
    main()
