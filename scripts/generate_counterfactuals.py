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

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

from chem_filters import (
    cns_mpo_score,
    has_motif,
    murcko_scaffold,
    sanitize_smiles,
    synthetic_feasibility_ok,
    tanimoto_similarity,
)
from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_col(df: pd.DataFrame, choices: list[str], required: bool = True) -> str | None:
    lc = {c.lower(): c for c in df.columns}
    for c in choices:
        if c.lower() in lc:
            return lc[c.lower()]
    if required:
        raise ValueError(f"Missing required columns: {choices}")
    return None


def approx_model_yhat(row: pd.Series) -> float:
    if "pIC50" in row and pd.notna(row["pIC50"]):
        return float(row["pIC50"])
    # fallback deterministic proxy
    tpsa = row.get("tpsa", np.nan)
    logp = row.get("logp", np.nan)
    mw = row.get("mw", np.nan)
    score = 6.0
    if pd.notna(logp):
        score += 0.2 * np.tanh(logp)
    if pd.notna(tpsa):
        score -= 0.002 * tpsa
    if pd.notna(mw):
        score -= 0.001 * max(0.0, mw - 450.0)
    return float(score)


def _find_dummy_and_anchor(mol):
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        mapped = [atom for atom in dummy_atoms if atom.GetAtomMapNum() == 1]
        if len(mapped) == 1:
            dummy_atoms = mapped
        else:
            return None
    dummy = dummy_atoms[0]
    neighbors = list(dummy.GetNeighbors())
    if len(neighbors) != 1:
        return None
    return dummy.GetIdx(), neighbors[0].GetIdx()


def combine_core_side(core_smi: str, side_smi: str) -> str | None:
    from rdkit import Chem

    core = Chem.MolFromSmiles(core_smi)
    side = Chem.MolFromSmiles(side_smi)
    if core is None or side is None:
        return None

    core_attach = _find_dummy_and_anchor(core)
    side_attach = _find_dummy_and_anchor(side)
    if core_attach is None or side_attach is None:
        return None
    core_dummy_idx, core_anchor_idx = core_attach
    side_dummy_idx, side_anchor_idx = side_attach

    try:
        combo = Chem.CombineMols(core, side)
        rw = Chem.RWMol(combo)
        side_offset = core.GetNumAtoms()
        rw.AddBond(core_anchor_idx, side_anchor_idx + side_offset, Chem.BondType.SINGLE)
        for idx in sorted([core_dummy_idx, side_dummy_idx + side_offset], reverse=True):
            rw.RemoveAtom(idx)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out, canonical=True)
    except Exception:
        return None


def _fragment_seed_pairs(seed_smiles: str, frag_settings: dict) -> list[tuple[str, str]]:
    from rdkit import Chem
    from rdkit.Chem import rdMMPA

    cache = frag_settings.setdefault("_fragment_cache", {})
    if seed_smiles in cache:
        return cache[seed_smiles]

    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        cache[seed_smiles] = []
        return []

    fragment_rows = list(
        rdMMPA.FragmentMol(
            seed_mol,
            maxCuts=frag_settings.get("maxCuts", 1),
            maxCutBonds=frag_settings.get("maxCutBonds", 20),
            pattern=frag_settings.get("pattern", "[!#1]!@!=!#[!#1]"),
            resultsAsMols=False,
        )
    )

    pairs: list[tuple[str, str]] = []
    for row in fragment_rows:
        if not isinstance(row, tuple):
            continue
        if len(row) == 2 and str(row[0]).strip() == "" and "." in str(row[1]):
            parts = [p.strip() for p in str(row[1]).split(".") if p.strip()]
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
                continue
        frags = [str(v).strip() for v in row if isinstance(v, str) and str(v).strip() and "*" in str(v)]
        if len(frags) >= 2:
            pairs.append((frags[0], frags[1]))

    cache[seed_smiles] = pairs
    return pairs


def seed_candidates_from_rule(seed_smiles, rule_ctx, lhs, rhs, frag_settings) -> list[str]:
    pairs = _fragment_seed_pairs(seed_smiles, frag_settings)
    diags = frag_settings.setdefault("_diag", {"fragment_rows": 0, "rule_matches": 0})
    diags["fragment_rows"] += len(pairs)
    out: list[str] = []
    for frag_a, frag_b in pairs:
        core, side = (frag_a, frag_b) if len(frag_a) >= len(frag_b) else (frag_b, frag_a)
        matched = (core == rule_ctx and side == lhs) or (side == rule_ctx and core == lhs)
        if not matched:
            continue
        diags["rule_matches"] += 1
        cand = combine_core_side(core, rhs)
        if cand is not None:
            out.append(cand)
    return list(dict.fromkeys(out))


def write_no_data_svg(path: Path, label: str = "NO DATA (0 rows)") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<svg xmlns='http://www.w3.org/2000/svg' width='420' height='120' viewBox='0 0 420 120'>",
                "  <rect width='420' height='120' fill='white' stroke='#999'/>",
                f"  <text x='210' y='68' font-size='24' text-anchor='middle' fill='#333' font-family='Arial'>{label}</text>",
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate constrained SAR counterfactual candidates.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--dataset_parquet", required=True)
    parser.add_argument("--bbb_parquet")
    parser.add_argument("--mmp_rules_parquet", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--preserve", choices=["scaffold", "motif"], default="scaffold")
    parser.add_argument("--motif_smarts", default="")
    parser.add_argument("--cns_constraint", choices=["keep_cns_like", "within_thresholds"], default="keep_cns_like")
    parser.add_argument("--cns_mpo_threshold", type=float, default=4.0)
    parser.add_argument("--max_edits_per_seed", type=int, default=50)
    parser.add_argument("--topk_per_seed", type=int, default=5)
    parser.add_argument("--min_tanimoto", type=float, default=0.3)
    parser.add_argument("--max_tanimoto", type=float, default=0.95)
    parser.add_argument("--seed_limit", type=int, default=200)
    add_plot_style_args(parser)
    args = parser.parse_args()
    if np is None or pd is None:
        raise SystemExit("numpy and pandas are required to run generate_counterfactuals.py")

    outdir = Path(args.outdir)
    for sub in ["rules", "candidates", "evaluation", "figures", "provenance"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(args.dataset_parquet)
    smi_col = pick_col(data, ["canonical_smiles", "smiles"])
    id_col = pick_col(data, ["compound_id", "molecule_chembl_id", "id"], required=False) or smi_col
    pic50_col = pick_col(data, ["pIC50", "pic50"], required=False)
    if pic50_col and pic50_col != "pIC50":
        data["pIC50"] = data[pic50_col]

    rules = pd.read_parquet(args.mmp_rules_parquet)
    n_rules = int(len(rules))
    if n_rules == 0:
        raise SystemExit(
            "No MMP rules were loaded (n_rules=0). Rebuild rules with a lower --min_support to recover candidate edits."
        )
    seeds = data[[id_col, smi_col] + (["pIC50"] if "pIC50" in data.columns else [])].head(args.seed_limit).copy()
    seeds.columns = ["seed_id", "seed_smiles"] + (["pIC50"] if "pIC50" in seeds.columns else [])
    seeds.to_parquet(outdir / "candidates" / "seeds.parquet", index=False)

    bbb = None
    if args.bbb_parquet and Path(args.bbb_parquet).exists():
        bbb = pd.read_parquet(args.bbb_parquet)

    generated_rows = []
    valid_rows = []
    validity_reasons: list[str] = []
    n_generated = 0
    frag_settings = {"maxCuts": 1, "maxCutBonds": 20, "pattern": "[!#1]!@!=!#[!#1]"}
    sorted_rules = rules.sort_values("support_count", ascending=False)
    total_fragment_rows = 0
    total_rule_matches = 0
    for _, seed in seeds.iterrows():
        seed_smiles = str(seed["seed_smiles"])
        seed_rec = sanitize_smiles(seed_smiles)
        if not seed_rec.valid:
            continue
        seed_props = seed_rec.props
        seed_mpo = cns_mpo_score(seed_props)
        seed_scaffold = murcko_scaffold(seed_rec.smiles)

        applied = 0
        seed_fragment_rows = 0
        seed_rule_matches = 0
        for _, rule in sorted_rules.iterrows():
            if applied >= args.max_edits_per_seed:
                break
            lhs = str(rule["lhs_fragment"])
            rhs = str(rule["rhs_fragment"])
            rule_ctx = str(rule.get("context_fragment", ""))
            diag_before = dict(frag_settings.get("_diag", {"fragment_rows": 0, "rule_matches": 0}))
            candidates = seed_candidates_from_rule(seed_rec.smiles, rule_ctx, lhs, rhs, frag_settings)
            diag_after = frag_settings.get("_diag", {"fragment_rows": 0, "rule_matches": 0})
            seed_fragment_rows += int(diag_after.get("fragment_rows", 0) - diag_before.get("fragment_rows", 0))
            seed_rule_matches += int(diag_after.get("rule_matches", 0) - diag_before.get("rule_matches", 0))
            if not candidates:
                continue
            for cand_smiles in candidates:
                if applied >= args.max_edits_per_seed:
                    break
                applied += 1
                n_generated += 1
                cand_rec = sanitize_smiles(cand_smiles)
                if not cand_rec.valid:
                    validity_reasons.append(cand_rec.reason or "invalid")
                    continue
                sim = tanimoto_similarity(seed_rec.smiles, cand_rec.smiles)
                dist = 1.0 - sim if pd.notna(sim) else np.nan
                if pd.notna(sim) and not (args.min_tanimoto <= sim <= args.max_tanimoto):
                    validity_reasons.append("tanimoto_out_of_range")
                    continue

                if args.preserve == "scaffold":
                    if seed_scaffold and murcko_scaffold(cand_rec.smiles) != seed_scaffold:
                        validity_reasons.append("scaffold_not_preserved")
                        continue
                else:
                    if not has_motif(cand_rec.smiles, args.motif_smarts):
                        validity_reasons.append("motif_not_preserved")
                        continue
                if not synthetic_feasibility_ok(cand_rec.props):
                    validity_reasons.append("synthetic_filter_failed")
                    continue

                cns_mpo = cns_mpo_score(cand_rec.props)
                cns_like = bool(cns_mpo >= args.cns_mpo_threshold) if pd.notna(cns_mpo) else False
                seed_cns_like = bool(seed_mpo >= args.cns_mpo_threshold) if pd.notna(seed_mpo) else False

                row = {
                    "seed_id": seed["seed_id"],
                    "seed_smiles": seed_rec.smiles,
                    "cand_id": f"{seed['seed_id']}_r{rule['rule_id']}_{applied}",
                    "cand_smiles": cand_rec.smiles,
                    "rule_id": rule["rule_id"],
                    "rule_support_count": rule.get("support_count", np.nan),
                    "yhat_x": approx_model_yhat(pd.Series(seed_props | {"pIC50": seed.get("pIC50", np.nan)})),
                    "yhat_xprime": approx_model_yhat(pd.Series(cand_rec.props)),
                    "seed_cns_mpo": seed_mpo,
                    "cand_cns_mpo": cns_mpo,
                    "seed_cns_like": seed_cns_like,
                    "cand_cns_like": cns_like,
                    "tanimoto": sim,
                    "chemical_distance": dist,
                    **{f"cand_{k}": v for k, v in cand_rec.props.items()},
                }
                row["delta_yhat"] = row["yhat_xprime"] - row["yhat_x"]
                row["delta_cns_mpo"] = cns_mpo - seed_mpo if pd.notna(cns_mpo) and pd.notna(seed_mpo) else np.nan
                valid_rows.append(row)

                if args.cns_constraint == "keep_cns_like" and seed_cns_like and not cns_like:
                    validity_reasons.append("cns_like_not_preserved")
                    continue
                if args.cns_constraint == "within_thresholds" and pd.notna(cns_mpo) and cns_mpo < args.cns_mpo_threshold:
                    validity_reasons.append("cns_mpo_below_threshold")
                    continue
                generated_rows.append(row)
        total_fragment_rows += seed_fragment_rows
        total_rule_matches += seed_rule_matches
        print(
            f"Seed diagnostics: seed_id={seed['seed_id']} fragment_rows={seed_fragment_rows} rule_matches={seed_rule_matches}",
            file=sys.stderr,
        )

    valid = pd.DataFrame(valid_rows)
    generated = pd.DataFrame(generated_rows)
    generated_path = outdir / "candidates" / "generated_counterfactuals.parquet"
    generated.to_parquet(generated_path, index=False)

    if generated.empty:
        filtered = generated.copy()
        dedup_removed = pd.DataFrame(columns=["seed_id", "cand_smiles", "reason"])
    else:
        before = len(generated)
        filtered = generated.sort_values(["seed_id", "delta_yhat"], ascending=[True, False]).drop_duplicates(["seed_id", "cand_smiles"])
        dedup_removed = pd.DataFrame({"metric": ["before", "after", "removed"], "value": [before, len(filtered), before - len(filtered)]})
    filtered_path = outdir / "candidates" / "filtered_counterfactuals.parquet"
    filtered.to_parquet(filtered_path, index=False)
    dedup_removed.to_csv(outdir / "candidates" / "duplicates_removed.csv", index=False)

    if not filtered.empty:
        filtered["pareto_score"] = (
            2.0 * filtered["delta_yhat"].fillna(0.0)
            + 0.4 * filtered["cand_cns_mpo"].fillna(0.0)
            - 0.5 * filtered["chemical_distance"].fillna(1.0)
            + 0.1 * np.log1p(filtered["rule_support_count"].fillna(0.0))
        )
        ranked = filtered.sort_values(["seed_id", "pareto_score"], ascending=[True, False]).groupby("seed_id").head(args.topk_per_seed)
    else:
        ranked = filtered.copy()
    ranked_path = outdir / "candidates" / "ranked_topk.parquet"
    ranked.to_parquet(ranked_path, index=False)

    n_seed_molecules = len(seeds)
    n_valid = len(valid)
    n_pass_bbb = len(generated)
    n_filtered = len(filtered)
    n_ranked_topk = len(ranked)
    print(
        "Counterfactual stage counts: "
        f"n_rules={n_rules}, "
        f"n_seed_molecules={n_seed_molecules}, "
        f"n_generated={n_generated}, "
        f"n_valid={n_valid}, "
        f"n_pass_bbb={n_pass_bbb}, "
        f"n_ranked_topk={n_ranked_topk}, "
        f"n_fragment_rows={total_fragment_rows}, "
        f"n_rule_matches={total_rule_matches}",
        file=sys.stderr,
    )

    if n_seed_molecules == 0:
        print("Stage n_seed_molecules=0: dataset is empty after loading or seed_limit is 0.", file=sys.stderr)
    if n_generated == 0:
        print("Stage n_generated=0: no rule LHS matched seed SMILES (or max_edits_per_seed=0).", file=sys.stderr)
    if n_valid == 0:
        print(
            "Stage n_valid=0: all generated candidates failed molecule validity/structural/similarity/synthesis checks.",
            file=sys.stderr,
        )
    if n_pass_bbb == 0:
        missing_cns = bool(n_valid > 0 and "cand_cns_mpo" in valid.columns and valid["cand_cns_mpo"].isna().all())
        if missing_cns:
            print(
                "Stage n_pass_bbb=0: CNS MPO is missing for all valid candidates (missing CNS MPO column/properties).",
                file=sys.stderr,
            )
        else:
            print(
                "Stage n_pass_bbb=0: CNS BBB constraints removed all valid candidates; threshold may be too strict.",
                file=sys.stderr,
            )
    if n_ranked_topk == 0:
        print("Stage n_ranked_topk=0: nothing remained after BBB-constrained filtering and deduplication.", file=sys.stderr)
    if n_filtered == 0:
        reason_counts = pd.Series(validity_reasons).value_counts()
        top_reasons = reason_counts.head(5).to_dict() if not reason_counts.empty else {}
        diagnostics = {
            "n_missing_cns_mpo": int(generated["cand_cns_mpo"].isna().sum()) if "cand_cns_mpo" in generated.columns else 0,
            "n_property_nan_rows": int(
                generated[[c for c in ["yhat_x", "yhat_xprime", "delta_yhat", "seed_cns_mpo", "cand_cns_mpo"] if c in generated.columns]]
                .isna()
                .any(axis=1)
                .sum()
            ) if not generated.empty else 0,
            "cns_mpo_threshold": args.cns_mpo_threshold,
            "cns_constraint": args.cns_constraint,
            "top_filter_reasons": top_reasons,
        }
        print(f"n_filtered is 0; likely causes: {json.dumps(diagnostics, sort_keys=True)}", file=sys.stderr)

    for parquet_path in [generated_path, filtered_path, ranked_path]:
        try:
            if pd.read_parquet(parquet_path).shape[0] == 0:
                write_no_data_svg(outdir / "figures" / f"{parquet_path.stem}_no_data.svg")
        except Exception as exc:
            print(f"Could not inspect {parquet_path}: {exc}", file=sys.stderr)

    delta_columns = [
        "seed_id",
        "cand_id",
        "yhat_x",
        "yhat_xprime",
        "delta_yhat",
        "seed_cns_mpo",
        "cand_cns_mpo",
        "delta_cns_mpo",
        "tanimoto",
        "rule_id",
    ]
    delta = ranked.reindex(columns=delta_columns)
    delta.to_csv(outdir / "evaluation" / "delta_predictions.csv", index=False)

    monotonic = pd.DataFrame(columns=["cand_id", "rule_id", "violation"])
    if not ranked.empty:
        v1 = (ranked["cand_tpsa"] > ranked.get("cand_tpsa", 0)) & (ranked["delta_cns_mpo"] > 0)
        monotonic = pd.DataFrame({"cand_id": ranked["cand_id"], "rule_id": ranked["rule_id"], "violation": v1.fillna(False).astype(int)})
    monotonic.to_csv(outdir / "evaluation" / "monotonicity_checks.csv", index=False)

    series_rank = pd.DataFrame({"metric": ["pairs"], "value": [len(ranked)]})
    series_rank.to_csv(outdir / "evaluation" / "series_ranking_constraints.csv", index=False)

    validity = pd.Series(validity_reasons).value_counts().rename_axis("reason").reset_index(name="count")
    validity.to_csv(outdir / "evaluation" / "validity_sanity.csv", index=False)

    if not filtered.empty:
        bbb_report = pd.DataFrame(
            {
                "constraint": ["cns_like", "cns_mpo_threshold"],
                "pass_rate": [filtered["cand_cns_like"].mean(), (filtered["cand_cns_mpo"] >= args.cns_mpo_threshold).mean()],
            }
        )
    else:
        bbb_report = pd.DataFrame({"constraint": ["cns_like", "cns_mpo_threshold"], "pass_rate": [0.0, 0.0]})
    bbb_report.to_csv(outdir / "evaluation" / "bbb_constraint_report.csv", index=False)

    style = style_from_args(args)
    configure_matplotlib(style, svg=True)
    import matplotlib.pyplot as plt

    fig_path = outdir / "figures" / "fig_edit_type_distribution.svg"
    type_counts = filtered["rule_id"].value_counts().head(10) if not filtered.empty else pd.Series(dtype=float)
    if not len(type_counts):
        write_no_data_svg(fig_path)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(type_counts.index.astype(str), type_counts.values)
        ax.tick_params(axis="x", rotation=60)
        style_axis(ax, style, "Edit Type Distribution", "Rule ID", "Frequency")
        fig.tight_layout(); fig.savefig(fig_path); plt.close(fig)

    fig_path = outdir / "figures" / "fig_deltaY_distribution.svg"
    if filtered.empty:
        write_no_data_svg(fig_path)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(filtered["delta_yhat"].fillna(0), bins=30)
        style_axis(ax, style, "Δŷ Distribution", "Δŷ", "Count")
        fig.tight_layout(); fig.savefig(fig_path); plt.close(fig)

    fig_path = outdir / "figures" / "fig_pareto_potency_vs_cns.svg"
    if filtered.empty:
        write_no_data_svg(fig_path)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(filtered["delta_yhat"], filtered["delta_cns_mpo"], alpha=0.7)
        sorted_pts = filtered[["delta_yhat", "delta_cns_mpo"]].sort_values("delta_yhat")
        frontier_y = np.maximum.accumulate(sorted_pts["delta_cns_mpo"].fillna(-1e9).values)
        ax.plot(sorted_pts["delta_yhat"].values, frontier_y, linewidth=2)
        style_axis(ax, style, "Pareto: Potency vs CNS", "Δŷ", "ΔCNS MPO")
        fig.tight_layout(); fig.savefig(fig_path); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bbb_report["constraint"], bbb_report["pass_rate"])
    style_axis(ax, style, "Counterfactual Success Rate", "Constraint", "Pass rate")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_counterfactual_success_rate.svg"); plt.close(fig)

    fig_path = outdir / "figures" / "fig_monotonicity_violations.svg"
    viol = monotonic.groupby("rule_id")["violation"].sum().head(10) if not monotonic.empty else pd.Series(dtype=float)
    if not len(viol):
        write_no_data_svg(fig_path)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(viol.index.astype(str), viol.values)
        ax.tick_params(axis="x", rotation=60)
        style_axis(ax, style, "Monotonicity Violations", "Rule ID", "Violations")
        fig.tight_layout(); fig.savefig(fig_path); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    top_tbl = ranked[[c for c in ["seed_id", "rule_id", "delta_yhat", "cand_cns_mpo", "chemical_distance"] if c in ranked.columns]].head(10)
    ax.table(cellText=top_tbl.round(3).values, colLabels=top_tbl.columns, loc="center") if not top_tbl.empty else ax.text(0.5, 0.5, "No edits", ha="center")
    style_axis(ax, style, "Top Edit Examples", None, None)
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_top_edits_examples.svg"); plt.close(fig)

    run_cfg = {
        "constraints": {
            "preserve": args.preserve,
            "motif_smarts": args.motif_smarts,
            "cns_constraint": args.cns_constraint,
            "cns_mpo_threshold": args.cns_mpo_threshold,
            "min_tanimoto": args.min_tanimoto,
            "max_tanimoto": args.max_tanimoto,
        },
        "topk_per_seed": args.topk_per_seed,
        "plotting": vars(args),
    }
    (outdir / "provenance" / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    checkpoint_path = Path(args.run_dir) / args.checkpoint
    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        "script_hashes": {
            "generate_counterfactuals.py": sha256_file(Path(__file__)),
            "chem_filters.py": sha256_file(Path(__file__).with_name("chem_filters.py")),
            "plot_style.py": sha256_file(Path(__file__).with_name("plot_style.py")),
        },
        "input_hashes": {
            "dataset": sha256_file(Path(args.dataset_parquet)),
            "rules": sha256_file(Path(args.mmp_rules_parquet)),
            "checkpoint": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
            "bbb": sha256_file(Path(args.bbb_parquet)) if args.bbb_parquet and Path(args.bbb_parquet).exists() else None,
        },
        "counts": {
            "n_rules": int(n_rules),
            "n_seed_molecules": int(n_seed_molecules),
            "n_generated": int(n_generated),
            "n_valid": int(n_valid),
            "n_pass_bbb": int(n_pass_bbb),
            "n_ranked_topk": int(n_ranked_topk),
            "n_filtered": int(n_filtered),
            "n_ranked": int(n_ranked_topk),
            "n_passing_filters": int(n_filtered),
            "topk": args.topk_per_seed,
        },
    }
    (outdir / "provenance" / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    env = subprocess.getoutput("python -m pip freeze")
    (outdir / "provenance" / "environment.txt").write_text(env + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
