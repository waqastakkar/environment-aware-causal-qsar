#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TYPES = ["IC50", "Ki", "Kd", "EC50", "AC50", "Potency"]
DEFAULT_UNITS = ["nM", "uM", "µM"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def fetch_rows(
    conn: sqlite3.Connection,
    target_chembl_id: str,
    types: list[str],
    units: list[str],
    confidence_min: int,
) -> list[dict[str, Any]]:
    query = """
    SELECT
        td.chembl_id AS target_chembl_id,
        md.chembl_id AS molecule_chembl_id,
        cs.canonical_smiles AS canonical_smiles,
        a.standard_type,
        a.standard_relation,
        a.standard_value,
        a.standard_units,
        a.pchembl_value,
        a.activity_comment,
        a.data_validity_comment,
        ass.confidence_score,
        ass.assay_type,
        docs.chembl_id AS document_chembl_id
    FROM activities a
    JOIN assays ass ON a.assay_id = ass.assay_id
    JOIN target_dictionary td ON ass.tid = td.tid
    JOIN molecule_dictionary md ON a.molregno = md.molregno
    LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
    LEFT JOIN docs ON a.doc_id = docs.doc_id
    WHERE td.chembl_id = ?
      AND a.standard_value IS NOT NULL
      AND a.standard_type IS NOT NULL
      AND a.standard_units IS NOT NULL
      AND ass.confidence_score >= ?
    """
    params: list[Any] = [target_chembl_id, confidence_min]

    if types:
        placeholders = ",".join(["?"] * len(types))
        query += f" AND a.standard_type IN ({placeholders})"
        params.extend(types)
    if units:
        placeholders = ",".join(["?"] * len(units))
        query += f" AND a.standard_units IN ({placeholders})"
        params.extend(units)

    query += " ORDER BY a.activity_id"

    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, params)
    return [dict(r) for r in cur.fetchall()]


def to_qsar_ready(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in raw_rows:
        smiles = (row.get("canonical_smiles") or "").strip()
        if not smiles:
            continue
        val = row.get("standard_value")
        if val is None:
            continue
        try:
            numeric = float(val)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        out.append(
            {
                "target_chembl_id": row.get("target_chembl_id"),
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "canonical_smiles": smiles,
                "standard_type": row.get("standard_type"),
                "standard_relation": row.get("standard_relation"),
                "standard_value": numeric,
                "standard_units": row.get("standard_units"),
                "pchembl_value": row.get("pchembl_value"),
                "confidence_score": row.get("confidence_score"),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--db")
    parser.add_argument("--target")
    parser.add_argument("--outdir")
    parser.add_argument("--types", default=",".join(DEFAULT_TYPES))
    parser.add_argument("--units", default=",".join(DEFAULT_UNITS))
    parser.add_argument("--confidence_min", type=int, default=7)
    return parser.parse_known_args()


def main() -> None:
    args, unknown = parse_args()
    if args.db and args.target and args.outdir:
        db_path = Path(args.db)
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        types = parse_csv_list(args.types)
        units = parse_csv_list(args.units)
        effective_config = {
            "db": str(db_path),
            "target": args.target,
            "outdir": str(outdir),
            "types": types,
            "units": units,
            "confidence_min": args.confidence_min,
        }

        with sqlite3.connect(str(db_path)) as conn:
            raw_rows = fetch_rows(conn, args.target, types, units, args.confidence_min)

        qsar_rows = to_qsar_ready(raw_rows)
        raw_csv = outdir / f"{args.target}_raw.csv"
        qsar_csv = outdir / f"{args.target}_qsar_ready.csv"
        write_csv(raw_csv, raw_rows)
        write_csv(qsar_csv, qsar_rows)

        script_path = Path(__file__).resolve()
        provenance = {
            "cli_args": sys.argv[1:],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "git_commit_hash": git_commit_hash(),
            "script_sha256": sha256_file(script_path),
            "db_path": str(db_path),
            "db_sha256": sha256_file(db_path) if db_path.exists() else None,
            "target_id": args.target,
        }
        write_json(outdir / "extraction_config.json", effective_config)
        write_json(outdir / "provenance.json", provenance)
        print(f"Wrote {raw_csv}")
        print(f"Wrote {qsar_csv}")
        print(f"Wrote {outdir / 'extraction_config.json'}")
        print(f"Wrote {outdir / 'provenance.json'}")
        return

    # Backward-compatible default behavior when pipeline invokes stub-style mode.
    print(f"Executed {__file__} with config={args.config} extra={unknown}")


if __name__ == "__main__":
    main()
