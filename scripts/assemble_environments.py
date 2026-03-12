#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bbb_rules import add_bbb_metrics
from property_calc import compute_properties

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover
    Chem = None
    MurckoScaffold = None


DEFAULT_ENV_KEYS = ["assay_type", "species", "readout", "publication", "chemistry_regime", "series"]
BBB_ANNOTATION_COLUMNS = [
    "molecule_id",
    "canonical_smiles",
    "is_bbb_like",
    "MW",
    "TPSA",
    "HBD",
    "HBA",
    "RB",
    "LogP",
    "cns_mpo",
    "bbb_rule_version",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble multi-environment datasets from Step 2 artifacts.")
    p.add_argument("--target", required=True)
    p.add_argument("--row_level_csv", required=True)
    p.add_argument("--compound_level_csv", required=True)
    p.add_argument("--raw_extract_csv")
    p.add_argument("--outdir", required=True)
    p.add_argument("--env_keys", nargs="+", default=DEFAULT_ENV_KEYS)
    p.add_argument("--bbb_rules", required=True)
    p.add_argument("--series_rules")
    return p.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def normalize_assay_type(v: Any) -> str:
    s = str(v).strip().lower()
    if not s or s == "nan":
        return "unknown"
    if "cell" in s:
        return "cell-based"
    if "bio" in s or "enzyme" in s:
        return "biochemical"
    return s


def normalize_species(v: Any) -> str:
    s = str(v).strip().lower()
    if not s or s == "nan":
        return "unknown"
    if "human" in s or s == "homo sapiens":
        return "human"
    return s


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def ensure_molecule_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_col = pick_col(out, ["molecule_id", "molecule_chembl_id", "chembl_molecule_id", "compound_id", "mol_id", "id"])
    if source_col is None:
        raise ValueError(
            "Could not determine primary key for compound-level data. "
            "Expected one of: molecule_id, molecule_chembl_id, chembl_molecule_id, compound_id, mol_id, id"
        )
    if source_col != "molecule_id":
        logging.warning("Creating canonical molecule_id from source column '%s'", source_col)
    out["molecule_id"] = out[source_col]

    out["molecule_id"] = out["molecule_id"].astype(str)
    dup_count = int(out["molecule_id"].duplicated(keep=False).sum())
    if dup_count > 0:
        raise ValueError(
            f"compound-level primary key molecule_id must be unique; found {dup_count} duplicate rows"
        )
    return out


def compute_scaffold(smiles: Any) -> str:
    s = str(smiles).strip()
    if not s or s == "nan":
        return "unknown"
    if Chem is None or MurckoScaffold is None:
        return f"nosdk::{s[:20]}"
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return "unknown"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold or "unknown"


def compute_chemistry_regime(df: pd.DataFrame, rules: dict[str, Any]) -> pd.Series:
    col_map = {
        "MW": ["MW", "mw", "molecular_weight"],
        "TPSA": ["TPSA", "tpsa"],
        "HBD": ["HBD", "hbd"],
        "LogP": ["LogP", "logp", "alogp", "xlogp"],
        "RotB": ["RotB", "rotb", "rotatable_bonds"],
    }

    cols: dict[str, str | None] = {k: pick_col(df, v) for k, v in col_map.items()}

    def classify(row: pd.Series) -> str:
        for need in ["MW", "TPSA", "HBD", "LogP"]:
            col = cols[need]
            if col is None or pd.isna(row[col]):
                return "unknown"
        checks = []
        checks.append(float(row[cols["MW"]]) <= float(rules.get("MW_max", 450)))
        checks.append(float(row[cols["TPSA"]]) <= float(rules.get("TPSA_max", 90)))
        checks.append(float(row[cols["HBD"]]) <= float(rules.get("HBD_max", 2)))
        logp = float(row[cols["LogP"]])
        checks.append(float(rules.get("LogP_min", -1)) <= logp <= float(rules.get("LogP_max", 5)))
        rotb_col = cols.get("RotB")
        if rules.get("use_RotB", True) and rotb_col is not None and not pd.isna(row[rotb_col]):
            checks.append(float(row[rotb_col]) <= float(rules.get("RotB_max", 8)))
        return "bbb-like" if all(checks) else "non-bbb-like"

    return df.apply(classify, axis=1)


def ensure_bbb_input_properties(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    out = df.copy()
    needed = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB"]
    if all(c in out.columns for c in needed):
        return out

    calc_base = out[[smiles_col]].copy()
    calc_base = calc_base.rename(columns={smiles_col: "canonical_smiles"})
    calculated = compute_properties(calc_base, smiles_col="canonical_smiles")
    for c in needed:
        if c not in out.columns:
            out[c] = calculated[c]
    return out


def build_bbb_annotations(df: pd.DataFrame, rules: dict[str, Any], smiles_col: str) -> pd.DataFrame:
    bbb_df = ensure_bbb_input_properties(df, smiles_col=smiles_col)
    bbb_df = add_bbb_metrics(bbb_df)

    bbb_df["is_bbb_like"] = compute_chemistry_regime(bbb_df, rules).eq("bbb-like")
    bbb_df["RB"] = pd.to_numeric(bbb_df.get("RotB"), errors="coerce")
    bbb_df["bbb_rule_version"] = f"bbb_rules::{json.dumps(rules, sort_keys=True)}"

    if smiles_col != "canonical_smiles":
        bbb_df["canonical_smiles"] = bbb_df[smiles_col]

    for col in BBB_ANNOTATION_COLUMNS:
        if col not in bbb_df.columns:
            bbb_df[col] = pd.NA

    return bbb_df[BBB_ANNOTATION_COLUMNS].sort_values("molecule_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    row_df = pd.read_csv(args.row_level_csv)
    comp_df = pd.read_csv(args.compound_level_csv)
    raw_df = pd.read_csv(args.raw_extract_csv) if args.raw_extract_csv and Path(args.raw_extract_csv).exists() else None

    rules = _load_yaml(Path(args.bbb_rules))

    # canonical compound identifier
    comp_df = ensure_molecule_id(comp_df)
    id_col = "molecule_id"

    smiles_col = pick_col(comp_df, ["canonical_smiles", "smiles"])
    if smiles_col is None:
        smiles_col = "canonical_smiles"
        comp_df[smiles_col] = "unknown"

    comp_df["scaffold_id"] = comp_df[smiles_col].map(compute_scaffold)

    standard_type_col = pick_col(comp_df, ["standard_type", "readout"])
    if standard_type_col is None:
        comp_df["readout"] = "unknown"
    else:
        comp_df["readout"] = comp_df[standard_type_col].fillna("unknown").astype(str)
    comp_df["chemistry_regime"] = compute_chemistry_regime(comp_df, rules)

    comp_df["publication"] = "unknown"
    comp_df["assay_type"] = "unknown"
    comp_df["species"] = "unknown"

    if raw_df is not None:
        raw_id_col = pick_col(raw_df, ["molecule_chembl_id", "compound_id", "molecule_id", "mol_id"])
        publication_col = pick_col(raw_df, ["document_chembl_id", "publication", "doc_id"])
        assay_col = pick_col(raw_df, ["assay_type", "assay_category", "assay_description"])
        species_col = pick_col(raw_df, ["target_organism", "species", "organism"])

        if raw_id_col is not None:
            keep_cols = [raw_id_col]
            rename_map = {}
            if publication_col:
                keep_cols.append(publication_col)
                rename_map[publication_col] = "publication"
            if assay_col:
                keep_cols.append(assay_col)
                rename_map[assay_col] = "assay_type"
            if species_col:
                keep_cols.append(species_col)
                rename_map[species_col] = "species"

            agg = raw_df[keep_cols].copy().rename(columns=rename_map)
            agg = agg.groupby(raw_id_col, as_index=True).first()
            comp_df = comp_df.merge(agg, left_on=id_col, right_index=True, how="left", suffixes=("", "_raw"))
            for c in ["publication", "assay_type", "species"]:
                raw_c = f"{c}_raw"
                if raw_c in comp_df.columns:
                    comp_df[c] = comp_df[raw_c].combine_first(comp_df[c])

    comp_df["assay_type"] = comp_df["assay_type"].map(normalize_assay_type)
    comp_df["species"] = comp_df["species"].map(normalize_species)
    comp_df["publication"] = comp_df["publication"].fillna("unknown").astype(str)

    comp_df["series"] = comp_df["publication"].astype(str) + "::" + comp_df["scaffold_id"].astype(str)

    env_keys = args.env_keys
    for k in env_keys:
        if k not in comp_df.columns:
            comp_df[k] = "unknown"
    comp_df["env_id"] = comp_df[env_keys].astype(str).agg("|".join, axis=1)
    try:
        comp_df["env_id_manual"] = comp_df[env_keys].astype(str).agg("|".join, axis=1)
    except Exception as exc:  # pragma: no cover - defensive fallback
        comp_df["env_id_manual"] = comp_df["env_id"]
        logging.warning("Could not create env_id_manual from env keys, falling back to env_id (%s)", exc)

    if comp_df["env_id_manual"].isna().any():
        logging.warning("env_id_manual contains missing values; filling missing entries from env_id")
        comp_df["env_id_manual"] = comp_df["env_id_manual"].fillna(comp_df["env_id"])

    row_id_col = pick_col(row_df, ["molecule_chembl_id", "compound_id", "molecule_id", "mol_id"])
    if row_id_col is None:
        row_id_col = id_col
    attach_cols = [id_col, "env_id_manual", "env_id", *env_keys, "scaffold_id"]
    attach_cols = [c for c in attach_cols if c in comp_df.columns]
    merged_row = row_df.merge(comp_df[attach_cols].drop_duplicates(id_col), left_on=row_id_col, right_on=id_col, how="left")

    merged_row.to_parquet(outdir / "multienv_row_level.parquet", index=False)
    comp_df.to_parquet(outdir / "multienv_compound_level.parquet", index=False)

    data_outdir = outdir / "data"
    data_outdir.mkdir(parents=True, exist_ok=True)
    bbb_annotations = build_bbb_annotations(comp_df, rules, smiles_col=smiles_col)
    bbb_annotations.to_parquet(data_outdir / "bbb_annotations.parquet", index=False)

    env_counts = comp_df.groupby("env_id", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    env_counts.to_csv(outdir / "env_counts.csv", index=False)

    comp_df[[id_col, "publication", "scaffold_id", "series", "env_id"]].to_csv(outdir / "series_assignments.csv", index=False)

    env_schema = {
        "env_tuple": env_keys,
        "description": {
            "assay_type": "biochemical vs cell-based from raw extract where available; else unknown",
            "species": "species bucket from raw extract where available; else unknown",
            "readout": "standard_type-like readout",
            "publication": "document/publication family",
            "chemistry_regime": "bbb-like vs non-bbb-like from configurable property thresholds",
            "series": "publication x scaffold",
        },
        "env_id_delimiter": "|",
    }
    (outdir / "env_vector_schema.json").write_text(json.dumps(env_schema, indent=2), encoding="utf-8")

    env_def = {
        "target": args.target,
        "env_keys": env_keys,
        "bbb_rules": rules,
        "series_rules": args.series_rules,
        "unknown_policy": "use_unknown_bucket_not_drop",
        "source_files": {
            "row_level_csv": args.row_level_csv,
            "compound_level_csv": args.compound_level_csv,
            "raw_extract_csv": args.raw_extract_csv,
        },
    }
    (outdir / "env_definitions.json").write_text(json.dumps(env_def, indent=2), encoding="utf-8")

    print(f"Assembled environments: {len(comp_df)} compounds, {merged_row.shape[0]} rows, {env_counts.shape[0]} environments")


if __name__ == "__main__":
    main()
