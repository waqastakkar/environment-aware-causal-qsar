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
import pandas as pd
from matplotlib import font_manager

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_style import NATURE5, PlotStyle, configure_matplotlib, style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Step 2 report figures + diagnostics.")
    parser.add_argument("--input", help="Primary row-level Step2 CSV (row_level_primary.csv / row_level_with_pIC50.csv)")
    parser.add_argument("--input_dir", help="Step2 output directory")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--font", default="Times New Roman")
    parser.add_argument("--bold_text", action="store_true")
    parser.add_argument("--palette", default="nature5")
    parser.add_argument("--font_title", type=int, default=16)
    parser.add_argument("--font_label", type=int, default=14)
    parser.add_argument("--font_tick", type=int, default=12)
    parser.add_argument("--font_legend", type=int, default=12)
    return parser.parse_args()


def resolve_font_status(font_name: str) -> dict[str, str | bool]:
    fallback = font_manager.findfont(font_name, fallback_to_default=True)
    try:
        exact = font_manager.findfont(font_name, fallback_to_default=False)
        return {"requested": font_name, "available": True, "resolved_path": exact}
    except Exception:
        return {"requested": font_name, "available": False, "resolved_path": fallback}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def load_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_pic50_dist(row_df: pd.DataFrame, figures_dir: Path, style: PlotStyle) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    row_df["pIC50"].hist(bins=40, color=style.palette[0], ax=ax)
    style_axis(ax, style, title="Primary pIC50 distribution", xlabel="pIC50", ylabel="Row count")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_primary_pic50_distribution.svg")
    plt.close(fig)


def save_before_after(before: pd.DataFrame, after: pd.DataFrame, col: str, out: Path, style: PlotStyle) -> None:
    if before.empty or after.empty:
        return
    b = before.set_index(col)["count"]
    a = after.set_index(col)["count"]
    idx = b.index.union(a.index)
    plot_df = pd.DataFrame({"before": b.reindex(idx, fill_value=0), "after": a.reindex(idx, fill_value=0)})

    fig, ax = plt.subplots(figsize=(max(6, len(plot_df) * 0.8), 4.5))
    x = range(len(plot_df))
    ax.bar([i - 0.2 for i in x], plot_df["before"], width=0.4, label="before", color=style.palette[1])
    ax.bar([i + 0.2 for i in x], plot_df["after"], width=0.4, label="after primary", color=style.palette[3])
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
    style_axis(ax, style, title=f"{col} before vs after", ylabel="Rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input and not args.input_dir:
        raise SystemExit("Provide either --input or --input_dir")

    input_dir = Path(args.input_dir) if args.input_dir else Path(args.input).resolve().parent
    row_path = Path(args.input) if args.input else (input_dir / "row_level_primary.csv")
    if not row_path.exists():
        row_path = input_dir / "row_level_with_pIC50.csv"
    comp_props_path = input_dir / "compound_level_with_properties.csv"

    outdir = Path(args.outdir)
    figures_dir = outdir / "figures"
    provenance_dir = outdir / "provenance"
    tables_dir = outdir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    style = PlotStyle(
        font_family=args.font,
        font_title=args.font_title,
        font_label=args.font_label,
        font_tick=args.font_tick,
        font_legend=args.font_legend,
        palette=tuple(NATURE5),
    )
    configure_matplotlib(style)
    font_status = resolve_font_status(args.font)

    row_df = pd.read_csv(row_path)
    comp_df = pd.read_csv(comp_props_path)
    save_pic50_dist(row_df, figures_dir, style)

    for metric in ["standard_type", "standard_relation", "units"]:
        before = load_optional(input_dir / f"counts_before_{metric}.csv")
        after = load_optional(input_dir / f"counts_after_primary_{metric}.csv")
        col = before.columns[0] if not before.empty else (after.columns[0] if not after.empty else None)
        if col:
            save_before_after(before, after, col, figures_dir / f"fig_before_after_{metric}.svg", style)

    drop = load_optional(input_dir / "drop_reasons_primary.csv").sort_values("count", ascending=False)
    if not drop.empty:
        drop.to_csv(tables_dir / "top_drop_reasons_primary.csv", index=False)

    diagnostics = {
        "unique_molecules_primary": int(comp_df["molecule_chembl_id"].nunique()),
        "pIC50_min": float(row_df["pIC50"].min()),
        "pIC50_max": float(row_df["pIC50"].max()),
        "standard_type_unique": sorted(row_df["standard_type"].astype(str).unique().tolist()),
        "standard_relation_unique": sorted(row_df["standard_relation"].astype(str).unique().tolist()),
    }
    (tables_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line_args": sys.argv[1:],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "font_status": font_status,
        "git_commit_hash": git_commit(),
        "script_sha256": {
            "scripts/qsar_postprocess.py": sha256_file(Path("scripts/qsar_postprocess.py")),
            "scripts/qsar_postprocess_report.py": sha256_file(Path("scripts/qsar_postprocess_report.py")),
        },
    }
    (provenance_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote Step 2 report to {outdir}")


if __name__ == "__main__":
    main()
