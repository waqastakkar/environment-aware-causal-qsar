#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

PALETTE_NATURE5 = ["#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build extraction summary tables and SVG diagnostics")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--svg", action="store_true", help="Retained for explicitness; output is always SVG")
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--bold_text", action="store_true", default=True, help="Retained for API compatibility; bold text is always enforced")
    p.add_argument("--palette", default="nature5", choices=["nature5"])
    p.add_argument("--font_title", type=float, default=16)
    p.add_argument("--font_label", type=float, default=14)
    p.add_argument("--font_tick", type=float, default=12)
    p.add_argument("--font_legend", type=float, default=12)
    return p.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_counts_csv(path: Path, name: str, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([name, "count"])
        for key, value in counts.most_common():
            writer.writerow([key, value])


def write_missingness_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("column,missing_count,missing_fraction\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    total = len(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["column", "missing_count", "missing_fraction"])
        for col in cols:
            missing = 0
            for row in rows:
                val = row.get(col)
                if val is None or str(val).strip() == "":
                    missing += 1
            writer.writerow([col, missing, f"{missing / total:.6f}"])


def apply_style(args: argparse.Namespace) -> list[str]:
    palette = PALETTE_NATURE5
    text_weight = "bold" if args.bold_text else "normal"
    plt.rcParams.update(
        {
            "savefig.format": "svg",
            "svg.fonttype": "none",
            "font.family": args.font,
            "font.weight": text_weight,
            "axes.titleweight": text_weight,
            "axes.labelweight": text_weight,
            "axes.titlesize": args.font_title,
            "axes.labelsize": args.font_label,
            "xtick.labelsize": args.font_tick,
            "ytick.labelsize": args.font_tick,
            "legend.fontsize": args.font_legend,
        }
    )
    return palette


def bar_plot(counts: Counter[str], title: str, xlabel: str, outpath: Path, palette: list[str]) -> None:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [palette[i % len(palette)] for i in range(len(labels))]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def value_distribution_log(qsar_rows: list[dict[str, str]], outpath: Path, palette: list[str]) -> None:
    groups: dict[str, list[float]] = {}
    for row in qsar_rows:
        val = row.get("standard_value")
        if val is None or str(val).strip() == "":
            continue
        try:
            numeric = float(val)
        except ValueError:
            continue
        if numeric <= 0:
            continue
        key = f"{row.get('standard_type','NA')}|{row.get('standard_units','NA')}"
        groups.setdefault(key, []).append(math.log10(numeric))

    fig, ax = plt.subplots(figsize=(9, 5))
    if groups:
        labels = list(groups.keys())
        data = [groups[k] for k in labels]
        bp = ax.boxplot(data, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(palette[i % len(palette)])
    else:
        labels = []
        ax.text(0.5, 0.5, "No positive standard_value values found", ha="center", va="center")

    ax.set_title("Value distribution (log10 standard_value)")
    ax.set_xlabel("Endpoint|Unit")
    ax.set_ylabel("log10(standard_value)")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def find_target_files(input_dir: Path) -> tuple[Path, Path]:
    raw_files = sorted(input_dir.glob("*_raw.csv"))
    qsar_files = sorted(input_dir.glob("*_qsar_ready.csv"))
    if not raw_files or not qsar_files:
        raise FileNotFoundError("Could not find both *_raw.csv and *_qsar_ready.csv in input_dir")
    return raw_files[0], qsar_files[0]


def main() -> None:
    args = parse_args()
    args.bold_text = True
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    summary_dir = outdir / "summary_tables"
    figures_dir = outdir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_path, qsar_path = find_target_files(input_dir)
    raw_rows = load_csv(raw_path)
    qsar_rows = load_csv(qsar_path)

    counts_type = Counter((r.get("standard_type") or "NA") for r in raw_rows)
    counts_units = Counter((r.get("standard_units") or "NA") for r in raw_rows)
    counts_relation = Counter((r.get("standard_relation") or "NA") for r in raw_rows)
    counts_conf = Counter((r.get("confidence_score") or "NA") for r in raw_rows)

    write_counts_csv(summary_dir / "counts_by_standard_type.csv", "standard_type", counts_type)
    write_counts_csv(summary_dir / "counts_by_units.csv", "standard_units", counts_units)
    write_counts_csv(summary_dir / "counts_by_relation.csv", "standard_relation", counts_relation)
    write_counts_csv(summary_dir / "counts_by_confidence.csv", "confidence_score", counts_conf)
    write_missingness_csv(summary_dir / "missingness_report.csv", raw_rows)

    palette = apply_style(args)
    bar_plot(
        counts_type,
        "Standard type distribution",
        "standard_type",
        figures_dir / "fig_standard_type_distribution.svg",
        palette,
    )
    bar_plot(
        counts_units,
        "Units distribution",
        "standard_units",
        figures_dir / "fig_units_distribution.svg",
        palette,
    )
    bar_plot(
        counts_conf,
        "Confidence distribution",
        "confidence_score",
        figures_dir / "fig_confidence_distribution.svg",
        palette,
    )
    value_distribution_log(qsar_rows, figures_dir / "fig_value_distribution_log.svg", palette)

    print(f"Report generated under {outdir}")


if __name__ == "__main__":
    main()
