#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ENV_METADATA_CANDIDATES = [
    "assay_chembl_id",
    "assay_type",
    "document_chembl_id",
    "publication",
    "src_id",
    "target_chembl_id",
    "target_organism",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2 QSAR post-processing: generate clean endpoint datasets.")
    parser.add_argument("--config", help="Optional pipeline config path")
    parser.add_argument("--input", required=True, help="Input CSV from Step 1")
    parser.add_argument("--outdir", required=True, help="Output data directory")
    parser.add_argument("--target", help="Optional target identifier (compatibility)")
    parser.add_argument("--endpoint", default=None, help="Primary endpoint, defaults to config postprocess.primary_endpoint or IC50")
    parser.add_argument("--threshold", type=float, default=None, help="pIC50 activity threshold (default 6.0)")
    parser.add_argument("--units_keep", nargs="+", default=None)
    parser.add_argument("--relation_keep", nargs="+", default=None)
    parser.add_argument("--aggregate", choices=["best", "median", "mean"], default=None)
    parser.add_argument("--secondary_endpoints", nargs="+", default=None, help="Optional secondary endpoints (e.g., Ki)")
    parser.add_argument("--max_value_nM", type=float, default=None)
    parser.add_argument("--svg", action="store_true", help="Retained for interface compatibility")
    return parser.parse_args()


def parse_args_compat() -> tuple[argparse.Namespace, list[str]]:
    argv = sys.argv[1:]
    has_input = any(token == "--input" or token.startswith("--input=") for token in argv)
    has_outdir = any(token == "--outdir" or token.startswith("--outdir=") for token in argv)
    if has_input or has_outdir:
        return parse_args(), []

    parser = argparse.ArgumentParser(description="Step 2 QSAR post-processing: generate clean endpoint datasets.")
    parser.add_argument("--config", help="Optional pipeline config path (stub compatibility mode)")
    return parser.parse_known_args()


def _require_column(df: pd.DataFrame, name: str) -> str:
    mapping = {c.lower(): c for c in df.columns}
    if name.lower() not in mapping:
        raise ValueError(f"Required column missing: {name}")
    return mapping[name.lower()]


def _optional_column(df: pd.DataFrame, name: str) -> str | None:
    mapping = {c.lower(): c for c in df.columns}
    return mapping.get(name.lower())


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _to_nm(value: float, unit: str) -> float:
    unit_u = str(unit).strip().upper()
    if unit_u == "NM":
        return float(value)
    if unit_u == "UM":
        return float(value) * 1000.0
    return math.nan


def compute_properties(smiles: str) -> dict[str, float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return {"mw": np.nan, "logp": np.nan, "hbd": np.nan, "hba": np.nan, "tpsa": np.nan, "rotatable_bonds": np.nan, "aromatic_rings": np.nan}

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {"mw": np.nan, "logp": np.nan, "hbd": np.nan, "hba": np.nan, "tpsa": np.nan, "rotatable_bonds": np.nan, "aromatic_rings": np.nan}

    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "aromatic_rings": Lipinski.RingCount(mol),
    }


def resolve_settings(args: argparse.Namespace) -> dict[str, object]:
    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    post = cfg.get("postprocess", {}) if isinstance(cfg.get("postprocess"), dict) else {}
    training = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}

    return {
        "primary_endpoint": str(args.endpoint or post.get("primary_endpoint", "IC50")),
        "threshold": float(args.threshold if args.threshold is not None else post.get("threshold", 6.0)),
        "units_primary": [str(x) for x in (args.units_keep or post.get("units_primary", ["nM", "uM"]))],
        "relations_primary": [str(x) for x in (args.relation_keep or post.get("allowed_relations_primary", ["="]))],
        "aggregate": str(args.aggregate or post.get("aggregate", "median")),
        "secondary_endpoints": [str(x) for x in (args.secondary_endpoints if args.secondary_endpoints is not None else post.get("secondary_endpoints", ["Ki"]))],
        "max_value_nM": float(args.max_value_nM if args.max_value_nM is not None else post.get("max_value_nM", 1e9)),
        "label_col": str(training.get("label_col", "pIC50")),
    }


def process_endpoint(df: pd.DataFrame, endpoint: str, settings: dict[str, object], summary_prefix: str) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    col_endpoint = _require_column(df, "standard_type")
    col_units = _require_column(df, "standard_units")
    col_relation = _require_column(df, "standard_relation")
    col_value = _require_column(df, "standard_value")

    out = df.copy()
    out["_endpoint_norm"] = _normalize_text(out[col_endpoint]).str.upper()
    out["_unit_norm"] = _normalize_text(out[col_units]).str.upper()
    out["_rel_norm"] = _normalize_text(out[col_relation])
    out["_value_num"] = pd.to_numeric(out[col_value], errors="coerce")

    drop_reasons = []
    is_endpoint = out["_endpoint_norm"] == endpoint.upper()
    drop_reasons.append(("wrong_standard_type", ~is_endpoint))

    is_unit_allowed = out["_unit_norm"].isin({u.upper() for u in settings["units_primary"]})
    drop_reasons.append(("unsupported_units", is_endpoint & ~is_unit_allowed))

    has_value = out["_value_num"].notna()
    drop_reasons.append(("missing_numeric_value", is_endpoint & is_unit_allowed & ~has_value))

    out["IC50_nM"] = out.apply(lambda r: _to_nm(r["_value_num"], r["_unit_norm"]) if pd.notna(r["_value_num"]) else math.nan, axis=1)
    positive = out["IC50_nM"] > 0
    drop_reasons.append(("non_positive_value", is_endpoint & is_unit_allowed & has_value & ~positive))

    rel_ok = out["_rel_norm"].isin(set(settings["relations_primary"]))
    drop_reasons.append(("censored_relation", is_endpoint & is_unit_allowed & has_value & positive & ~rel_ok))

    within_max = out["IC50_nM"] <= float(settings["max_value_nM"])
    drop_reasons.append(("extreme_value", is_endpoint & is_unit_allowed & has_value & positive & rel_ok & ~within_max))

    keep = is_endpoint & is_unit_allowed & has_value & positive & rel_ok & within_max
    filtered = out.loc[keep].copy()
    filtered["standard_units"] = "nM"
    filtered["standard_value"] = filtered["IC50_nM"]
    filtered["pIC50"] = 9.0 - np.log10(filtered["IC50_nM"])

    counts = {
        f"{summary_prefix}_rows_before": int(len(df)),
        f"{summary_prefix}_rows_after_filtering": int(len(filtered)),
    }

    for key, mask in drop_reasons:
        counts[f"{summary_prefix}_dropped_{key}"] = int(mask.sum())

    dropped_rows = []
    for key, mask in drop_reasons:
        if mask.any():
            dropped_rows.append(pd.DataFrame({"reason": key, "count": [int(mask.sum())]}))
    drop_df = pd.concat(dropped_rows, ignore_index=True) if dropped_rows else pd.DataFrame(columns=["reason", "count"])

    return filtered.drop(columns=[c for c in filtered.columns if c.startswith("_")], errors="ignore"), counts, drop_df


def build_compound_level(row_df: pd.DataFrame, threshold: float, aggregate: str) -> pd.DataFrame:
    col_compound = _require_column(row_df, "molecule_chembl_id")
    col_smiles = _require_column(row_df, "canonical_smiles")

    agg_func = {"best": "max", "median": "median", "mean": "mean"}[aggregate]
    grouped = row_df.groupby([col_compound, col_smiles], dropna=False)["pIC50"].agg([agg_func, "count"]).reset_index()
    grouped.columns = ["molecule_chembl_id", "canonical_smiles", "pIC50", "n_measurements"]
    grouped["smiles"] = grouped["canonical_smiles"]
    grouped["activity_label"] = (grouped["pIC50"] >= threshold).astype(int)

    keep_meta = [c for c in ENV_METADATA_CANDIDATES if c in row_df.columns]
    if keep_meta:
        meta = row_df[[col_compound, *keep_meta]].drop_duplicates(subset=[col_compound], keep="first")
        grouped = grouped.merge(meta, on="molecule_chembl_id", how="left")

    props = grouped[["molecule_chembl_id", "canonical_smiles"]].copy()
    props_df = props["canonical_smiles"].apply(compute_properties).apply(pd.Series)
    return pd.concat([grouped, props_df], axis=1)


def main() -> None:
    args, unknown = parse_args_compat()
    if not getattr(args, "input", None) or not getattr(args, "outdir", None):
        print(f"Executed {__file__} with config={getattr(args, 'config', None)} extra={unknown}")
        return

    settings = resolve_settings(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    type_col = _require_column(df, "standard_type")
    rel_col = _require_column(df, "standard_relation")
    unit_col = _require_column(df, "standard_units")

    before_type = df[type_col].astype(str).value_counts(dropna=False).reset_index()
    before_type.columns = ["standard_type", "count"]
    before_rel = df[rel_col].astype(str).value_counts(dropna=False).reset_index()
    before_rel.columns = ["standard_relation", "count"]
    before_units = df[unit_col].astype(str).value_counts(dropna=False).reset_index()
    before_units.columns = ["standard_units", "count"]

    primary_row, primary_counts, primary_drop = process_endpoint(df, str(settings["primary_endpoint"]), settings, "primary")
    primary_comp = build_compound_level(primary_row, float(settings["threshold"]), str(settings["aggregate"]))

    (outdir / "row_level_primary.csv").write_text(primary_row.to_csv(index=False), encoding="utf-8")
    (outdir / "compound_level_primary.csv").write_text(primary_comp[["molecule_chembl_id", "canonical_smiles", "pIC50", "n_measurements", "activity_label"]].to_csv(index=False), encoding="utf-8")
    primary_comp.to_csv(outdir / "compound_level_primary_with_properties.csv", index=False)

    # Backward-compatible aliases used by step03 and existing reports.
    primary_row.to_csv(outdir / "row_level_with_pIC50.csv", index=False)
    primary_comp[["molecule_chembl_id", "canonical_smiles", "pIC50", "n_measurements", "activity_label"]].to_csv(outdir / "compound_level_pIC50.csv", index=False)
    primary_comp.to_csv(outdir / "compound_level_with_properties.csv", index=False)

    secondary_outputs = []
    for endpoint in settings["secondary_endpoints"]:
        if str(endpoint).upper() == str(settings["primary_endpoint"]).upper():
            continue
        sec_df, sec_counts, sec_drop = process_endpoint(df, str(endpoint), settings, f"secondary_{endpoint}")
        if sec_df.empty:
            continue
        sec_comp = build_compound_level(sec_df, float(settings["threshold"]), str(settings["aggregate"]))
        stem = f"secondary_{str(endpoint).lower()}"
        sec_df.to_csv(outdir / f"row_level_{stem}.csv", index=False)
        sec_comp.to_csv(outdir / f"compound_level_{stem}_with_properties.csv", index=False)
        sec_drop.to_csv(outdir / f"drop_reasons_{stem}.csv", index=False)
        secondary_outputs.append({"endpoint": endpoint, "rows": len(sec_df), **sec_counts})

    after_type = primary_row[type_col].astype(str).value_counts(dropna=False).reset_index()
    after_type.columns = ["standard_type", "count"]
    after_rel = primary_row[rel_col].astype(str).value_counts(dropna=False).reset_index()
    after_rel.columns = ["standard_relation", "count"]
    after_units = primary_row[unit_col].astype(str).value_counts(dropna=False).reset_index()
    after_units.columns = ["standard_units", "count"]

    before_type.to_csv(outdir / "counts_before_standard_type.csv", index=False)
    before_rel.to_csv(outdir / "counts_before_standard_relation.csv", index=False)
    before_units.to_csv(outdir / "counts_before_units.csv", index=False)
    after_type.to_csv(outdir / "counts_after_primary_standard_type.csv", index=False)
    after_rel.to_csv(outdir / "counts_after_primary_standard_relation.csv", index=False)
    after_units.to_csv(outdir / "counts_after_primary_units.csv", index=False)
    primary_drop.to_csv(outdir / "drop_reasons_primary.csv", index=False)

    pic50_min = float(primary_row["pIC50"].min()) if not primary_row.empty else math.nan
    pic50_max = float(primary_row["pIC50"].max()) if not primary_row.empty else math.nan

    summary_metrics = {
        "rows_before_filtering": int(len(df)),
        "rows_after_primary": int(len(primary_row)),
        "n_compounds_primary": int(primary_comp["molecule_chembl_id"].nunique()),
        "n_actives_primary": int(primary_comp["activity_label"].sum()),
        "n_inactives_primary": int((primary_comp["activity_label"] == 0).sum()),
        "pIC50_min_primary": pic50_min,
        "pIC50_max_primary": pic50_max,
        **primary_counts,
    }
    pd.DataFrame({"metric": list(summary_metrics.keys()), "value": list(summary_metrics.values())}).to_csv(outdir / "summary.csv", index=False)

    run_config = {
        "primary_endpoint": settings["primary_endpoint"],
        "allowed_relations_primary": settings["relations_primary"],
        "units_primary": settings["units_primary"],
        "max_value_nM": settings["max_value_nM"],
        "threshold": settings["threshold"],
        "aggregate": settings["aggregate"],
        "secondary_endpoints": settings["secondary_endpoints"],
        "secondary_outputs": secondary_outputs,
    }
    (outdir / "postprocess_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Wrote Step 2 primary outputs in {outdir}")


if __name__ == "__main__":
    main()
