#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

SQL = """
SELECT
    md.chembl_id AS molecule_chembl_id,
    cs.canonical_smiles,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.standard_type,
    act.assay_id,
    act.doc_id
FROM activities act
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
JOIN molecule_dictionary md ON act.molregno = md.molregno
LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
WHERE td.chembl_id = ?
  AND act.standard_type = 'Inhibition'
  AND act.standard_units = '%'
  AND act.standard_value IS NOT NULL
  AND cs.canonical_smiles IS NOT NULL
"""


EXPECTED_COLUMNS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_relation",
    "standard_value",
    "standard_units",
    "standard_type",
    "assay_id",
    "doc_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract external inhibition data from a local ChEMBL SQLite database."
    )
    parser.add_argument("--target", required=True, help="Target ChEMBL ID (e.g., CHEMBL335)")
    parser.add_argument("--chembl_sqlite", required=True, help="Path to local ChEMBL SQLite database")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.chembl_sqlite)
    out_csv = Path(args.out_csv)

    if not db_path.is_file():
        raise SystemExit(f"Missing local ChEMBL SQLite database: {db_path}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(SQL, (args.target,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []

    if not rows:
        raise SystemExit(
            f"No inhibition records found for target {args.target} in database: {db_path}"
        )

    if columns != EXPECTED_COLUMNS:
        raise SystemExit(
            f"Unexpected query columns: {columns}. Expected: {EXPECTED_COLUMNS}"
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"Extracted {len(rows)} inhibition records from ChEMBL database")


if __name__ == "__main__":
    main()
