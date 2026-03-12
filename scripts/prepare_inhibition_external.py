#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import ks_2samp

from plot_style import PlotStyle, configure_matplotlib, parse_palette, style_axis


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def resolve_col(df: pd.DataFrame, options: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower:
            return lower[opt.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {options}; available={list(df.columns)}")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare inhibition external set with strict leakage removal.")
    p.add_argument("--target", default="CHEMBL335")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--ic50_parquet", required=True)
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--split_name", default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--inhib_threshold", type=float, default=50.0)
    p.add_argument("--svg", action="store_true", default=True)
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--bold_text", action="store_true", default=True)
    p.add_argument("--palette", default="nature5")
    p.add_argument("--font_title", type=int, default=16)
    p.add_argument("--font_label", type=int, default=14)
    p.add_argument("--font_tick", type=int, default=12)
    p.add_argument("--font_legend", type=int, default=12)
    return p.parse_args()


def add_mol_fields(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    out = df.copy()
    mols = []
    can = []
    ik = []
    valid = []
    mw = []
    logp = []
    tpsa = []
    for s in out[smiles_col].astype(str):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            mols.append(None)
            can.append(None)
            ik.append(None)
            valid.append(False)
            mw.append(np.nan)
            logp.append(np.nan)
            tpsa.append(np.nan)
        else:
            mols.append(mol)
            can.append(Chem.MolToSmiles(mol, canonical=True))
            ik.append(Chem.MolToInchiKey(mol))
            valid.append(True)
            mw.append(float(Descriptors.MolWt(mol)))
            logp.append(float(Descriptors.MolLogP(mol)))
            tpsa.append(float(Descriptors.TPSA(mol)))
    out["smiles_canonical"] = can
    out["inchikey"] = ik
    out["smiles_valid"] = valid
    out["MW"] = mw
    out["LogP"] = logp
    out["TPSA"] = tpsa
    return out


def save_environment_txt(path: Path) -> None:
    text = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def read_external_csv(path: str) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(path, sep=";", low_memory=False)
    detected_delimiter = ";"
    if len(raw.columns) == 1 and "," in str(raw.columns[0]):
        raw = pd.read_csv(path, sep=",", low_memory=False)
        detected_delimiter = ","
    return raw, detected_delimiter


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    data_dir = outdir / "data"
    reports_dir = outdir / "reports"
    figs_dir = outdir / "figures"
    prov_dir = outdir / "provenance"
    for d in [data_dir, reports_dir, figs_dir, prov_dir]:
        d.mkdir(parents=True, exist_ok=True)

    style = PlotStyle(
        font_family=args.font,
        font_title=args.font_title,
        font_label=args.font_label,
        font_tick=args.font_tick,
        font_legend=args.font_legend,
        bold_text=True,
        palette=parse_palette(args.palette),
    )
    configure_matplotlib(style, svg=True)

    raw, detected_delimiter = read_external_csv(args.input_csv)
    print(f"[prepare_inhibition_external] detected delimiter: '{detected_delimiter}'")
    smi_col = resolve_col(raw, ["Smiles", "smiles", "canonical_smiles"])
    print(f"[prepare_inhibition_external] resolved smiles column: {smi_col}")
    type_col = resolve_col(raw, ["Standard Type", "standard_type"])
    unit_col = resolve_col(raw, ["Standard Units", "standard_units"])
    rel_col = resolve_col(raw, ["Standard Relation", "standard_relation"])
    value_col = resolve_col(raw, ["Standard Value", "standard_value"])
    pchembl_col = resolve_col(raw, ["pChEMBL Value"], required=False)
    mol_id_col = resolve_col(raw, ["Molecule ChEMBL ID", "molecule_chembl_id", "molecule_id"], required=False)

    parsed = raw.copy()
    parsed["standard_relation_norm"] = parsed[rel_col].astype(str).str.strip()
    parsed["inhib_pct_raw"] = pd.to_numeric(parsed[value_col], errors="coerce")
    parsed = add_mol_fields(parsed, smi_col)
    parsed.to_parquet(data_dir / "inhibition_raw_parsed.parquet", index=False)

    relation_allowed = parsed["standard_relation_norm"].isin(["=", ">", ">="])
    keep = (
        parsed[smi_col].notna()
        & (parsed[type_col].astype(str).str.strip().str.lower().str.contains("inhibition"))
        & (parsed[unit_col].astype(str).str.strip() == "%")
        & parsed["inhib_pct_raw"].notna()
        & parsed["inhib_pct_raw"].between(0, 100, inclusive="both")
        & relation_allowed
        & parsed["smiles_valid"]
    )
    clean = parsed[keep].copy()
    clean["pct_out_of_range"] = (clean["inhib_pct_raw"] < 0) | (clean["inhib_pct_raw"] > 100)
    thresh = float(args.inhib_threshold)
    clean["y_inhib_active"] = 0
    clean.loc[
        clean["standard_relation_norm"].isin([">", ">="]) & (clean["inhib_pct_raw"] > thresh),
        "y_inhib_active",
    ] = 1
    clean.loc[
        (clean["standard_relation_norm"] == "=") & (clean["inhib_pct_raw"] >= thresh),
        "y_inhib_active",
    ] = 1
    clean["y_inhib_active"] = clean["y_inhib_active"].astype(int)
    if mol_id_col and mol_id_col in clean.columns:
        clean["molecule_id"] = clean[mol_id_col].astype(str)
    clean.to_parquet(data_dir / "inhibition_clean.parquet", index=False)

    if pchembl_col and pchembl_col in clean.columns:
        clean["_has_pchembl"] = clean[pchembl_col].notna().astype(int)
        dedup = clean.sort_values(["_has_pchembl"], ascending=False).drop_duplicates(subset=["inchikey"], keep="first").copy()
        dedup.drop(columns=["_has_pchembl"], inplace=True)
    else:
        dedup = clean.drop_duplicates(subset=["inchikey"], keep="first").copy()
    dedup.to_parquet(data_dir / "inhibition_dedup_internal.parquet", index=False)

    ic50 = pd.read_parquet(args.ic50_parquet)
    if "inchikey" not in ic50.columns:
        if "smiles" in ic50.columns:
            ic50 = add_mol_fields(ic50, "smiles")
        else:
            raise ValueError("IC50 parquet must contain inchikey or smiles")
    ic50_keys = set(ic50["inchikey"].dropna().astype(str))
    ic50_smiles = set(ic50["smiles"].dropna().astype(str)) if "smiles" in ic50.columns else set()

    dedup["overlap_ic50"] = dedup["inchikey"].astype(str).isin(ic50_keys) | dedup["smiles_canonical"].astype(str).isin(ic50_smiles)
    external = dedup[~dedup["overlap_ic50"]].copy()
    external.drop(columns=["overlap_ic50"], inplace=True)
    external.to_parquet(data_dir / "inhibition_external_final.parquet", index=False)

    parsing_summary = pd.DataFrame(
        [
            {"stage": "raw_rows", "count": len(raw)},
            {"stage": "after_endpoint_unit_value_smiles_filters", "count": len(clean)},
            {"stage": "after_internal_dedup", "count": len(dedup)},
            {"stage": "after_ic50_leakage_removal", "count": len(external)},
        ]
    )
    parsing_summary.to_csv(reports_dir / "parsing_summary.csv", index=False)

    value_sanity = pd.DataFrame(
        [
            {"metric": "raw_missing_standard_value", "count": int(parsed["inhib_pct_raw"].isna().sum())},
            {"metric": "raw_below_0", "count": int((parsed["inhib_pct_raw"] < 0).sum())},
            {"metric": "raw_above_100", "count": int((parsed["inhib_pct_raw"] > 100).sum())},
            {"metric": "clean_out_of_range_flagged", "count": int(clean["pct_out_of_range"].sum())},
            {"metric": "raw_disallowed_relation", "count": int((~relation_allowed).sum())},
        ]
    )
    value_sanity.to_csv(reports_dir / "value_sanity.csv", index=False)

    relation_counts = (
        parsed["standard_relation_norm"]
        .where(parsed["standard_relation_norm"].isin(["=", ">", ">=", "<", "<="]), "other")
        .value_counts(dropna=False)
        .reindex(["=", ">", ">=", "<", "<="], fill_value=0)
        .rename_axis("standard_relation")
        .reset_index(name="count")
    )
    relation_counts.to_csv(reports_dir / "counts_by_relation.csv", index=False)

    overlap = pd.DataFrame(
        [
            {"metric": "dedup_total", "count": len(dedup)},
            {"metric": "overlap_removed", "count": int((dedup["inchikey"].astype(str).isin(ic50_keys)).sum())},
            {"metric": "external_remaining", "count": len(external)},
        ]
    )
    overlap.to_csv(reports_dir / "overlap_with_ic50.csv", index=False)

    split_rows = []
    if args.splits_dir and args.split_name:
        split_base = Path(args.splits_dir) / args.split_name
        for part in ["train", "val", "test"]:
            f = split_base / f"{part}_ids.csv"
            if not f.exists():
                continue
            ids = pd.read_csv(f)
            col = "molecule_id" if "molecule_id" in ids.columns else ids.columns[0]
            split_ids = set(ids[col].astype(str))
            in_part = external[external.get("molecule_id", pd.Series(dtype=str)).astype(str).isin(split_ids)] if "molecule_id" in external.columns else pd.DataFrame()
            split_rows.append({"split": part, "join_key": "molecule_id", "overlap_count": int(len(in_part))})
            if "inchikey" in ic50.columns:
                ic50_part = ic50[ic50.get("molecule_id", pd.Series(dtype=str)).astype(str).isin(split_ids)] if "molecule_id" in ic50.columns else pd.DataFrame()
                part_keys = set(ic50_part.get("inchikey", pd.Series(dtype=str)).dropna().astype(str))
                split_rows.append(
                    {
                        "split": part,
                        "join_key": "inchikey_vs_internal_split",
                        "overlap_count": int(external["inchikey"].astype(str).isin(part_keys).sum()),
                    }
                )
    if not split_rows:
        split_rows = [{"split": "NA", "join_key": "not_provided", "overlap_count": 0}]
    pd.DataFrame(split_rows).to_csv(reports_dir / "overlap_by_split_membership.csv", index=False)

    # Shift report vs IC50 train if available else full IC50
    compare_base = ic50.copy()
    if args.splits_dir and args.split_name and "molecule_id" in ic50.columns:
        train_f = Path(args.splits_dir) / args.split_name / "train_ids.csv"
        if train_f.exists():
            ids = pd.read_csv(train_f)
            col = "molecule_id" if "molecule_id" in ids.columns else ids.columns[0]
            compare_base = ic50[ic50["molecule_id"].astype(str).isin(set(ids[col].astype(str)))].copy()

    props = [p for p in ["MW", "LogP", "TPSA"] if p in compare_base.columns]
    if not props:
        compare_base = add_mol_fields(compare_base, "smiles")
        props = ["MW", "LogP", "TPSA"]

    shift_rows = []
    for p in props:
        a = external[p].dropna().astype(float)
        b = compare_base[p].dropna().astype(float)
        if len(a) and len(b):
            ks = ks_2samp(a, b)
            shift_rows.append(
                {
                    "property": p,
                    "n_external": len(a),
                    "n_ic50_train": len(b),
                    "ks_stat": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "external_mean": float(a.mean()),
                    "ic50_train_mean": float(b.mean()),
                }
            )
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(reports_dir / "shift_vs_ic50_train.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean["inhib_pct_raw"], bins=30, alpha=0.85)
    style_axis(ax, style, "Inhibition value distribution", "Inhibition (%)", "Count")
    fig.tight_layout(); fig.savefig(figs_dir / "fig_inhibition_value_distribution.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["dedup", "removed", "external"], [len(dedup), len(dedup) - len(external), len(external)])
    style_axis(ax, style, "Overlap with IC50", "Stage", "Count")
    fig.tight_layout(); fig.savefig(figs_dir / "fig_overlap_breakdown.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if not shift_df.empty:
        ax.bar(shift_df["property"], shift_df["ks_stat"])
    style_axis(ax, style, "Shift vs IC50 train", "Property", "KS statistic")
    fig.tight_layout(); fig.savefig(figs_dir / "fig_shift_vs_ic50_train.svg"); plt.close(fig)

    run_config = vars(args)
    (prov_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": run_config,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "script_sha256": sha256_file(Path(__file__)),
        "input_hashes": {
            "input_csv": sha256_file(Path(args.input_csv)),
            "ic50_parquet": sha256_file(Path(args.ic50_parquet)),
        },
        "counts": {row["stage"]: int(row["count"]) for _, row in parsing_summary.iterrows()},
    }
    (prov_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    save_environment_txt(prov_dir / "environment.txt")


if __name__ == "__main__":
    main()
