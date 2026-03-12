#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    out_path = Path("outputs/step3/multienv_compound_level.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O", "CCCO", "CC(C)O", "CC(C)N", "CCOC", "CCS", "CCBr"]
    pvals = [5.10, 5.65, 6.35, 6.90, 7.20, 4.85, 5.55, 6.15, 6.05, 5.40, 5.80, 6.75]
    envs = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    for i, (smi, pic50, env) in enumerate(zip(smiles, pvals, envs), start=1):
        rows.append(
            {
                "molecule_id": f"MOL_{i:03d}",
                "canonical_smiles": smi,
                "pIC50": pic50,
                "activity_label": int(pic50 >= 6.0),
                "env_id": env,
                "env_id_manual": env,
                "publication_year": 2010 + (i % 6),
                "assay_type": "cell-based" if i % 3 == 0 else "biochemical",
                "series_id": f"SERIES_{i:03d}",
                "document_id": f"DOC_{i:03d}",
                "MW": 250 + i,
                "LogP": 1.0 + (i * 0.1),
                "TPSA": 35 + i,
                "HBD": i % 3,
                "HBA": 2 + (i % 4),
                "RotB": i % 5,
                "Rings": 1 + (i % 3),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)

    print(f"Wrote tiny step3 parquet: {out_path} ({len(df)} rows)")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
