#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from library_clean import clean_library
from library_io import parse_library


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


def _mkdirs(root: Path) -> None:
    for s in ["input", "processed", "provenance"]:
        (root / s).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 12a: library preparation")
    p.add_argument("--input_path", required=True)
    p.add_argument("--input_format", choices=["csv", "smi"], default="csv")
    p.add_argument("--outdir", required=True)
    p.add_argument("--screen_id", default=None)
    p.add_argument("--target", required=True)
    p.add_argument("--smi_layout", choices=["smiles_id", "smiles_name_id", "smiles_only"], default="smiles_id")
    p.add_argument("--header", default="auto")
    p.add_argument("--comment_prefix", default="#")
    p.add_argument("--allow_cpp_comments", default="true")
    p.add_argument("--name_is_rest", default="true")
    p.add_argument("--smi_quoted_name", default="false")
    p.add_argument("--sep", default=",")
    p.add_argument("--quotechar", default='"')
    p.add_argument("--smiles_col", default="smiles")
    p.add_argument("--id_col", default="compound_id")
    p.add_argument("--name_col", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"missing required file: {input_path}")

    screen_id = args.screen_id or datetime.now(timezone.utc).strftime("prepare_%Y%m%dT%H%M%SZ")
    out = Path(args.outdir) if Path(args.outdir).name == screen_id else Path(args.outdir) / args.target / screen_id
    _mkdirs(out)

    parsed, manifest = parse_library(
        input_path=str(input_path),
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
    (out / "input/input_fingerprint.json").write_text(json.dumps({"sha256": _sha256(input_path)}, indent=2), encoding="utf-8")

    clean, dedup, clean_report = clean_library(parsed)
    clean.to_parquet(out / "processed/library_clean.parquet", index=False)
    dedup.to_parquet(out / "processed/library_dedup.parquet", index=False)
    pd.DataFrame([{"metric": k, "value": v} for k, v in clean_report.items()]).to_csv(
        out / "processed/cleaning_report.csv", index=False
    )

    (out / "provenance/prepare_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
