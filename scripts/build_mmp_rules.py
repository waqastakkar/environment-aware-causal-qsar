#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

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
        raise ValueError(f"Missing required columns. expected one of: {choices}")
    return None


def _collect_rules_from_fragment_rows(
    fragment_rows: list[tuple[str, str]],
) -> tuple[list[tuple[str, str, str]], dict[str, list[str]]]:
    """Aggregate fragment rows into directional rules keyed by core fragment."""
    by_core: dict[str, list[str]] = defaultdict(list)
    for core, sidechain in fragment_rows:
        if not core or not sidechain:
            continue
        by_core[core].append(sidechain)

    rules: list[tuple[str, str, str]] = []
    for core, sides in by_core.items():
        uniq = sorted(set(sides))
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                if i == j:
                    continue
                rules.append((uniq[i], uniq[j], core))
    return rules, by_core


def parse_fragment_tuple(row, Chem):
    """Parse rdMMPA.FragmentMol output row into (core, sidechain).

    Handles common RDKit output where row is ('', 'fragA.fragB').
    """
    # Case 1: RDKit returns ('', 'fragA.fragB') where both fragments are packed in row[1]
    try:
        if (
            isinstance(row, tuple)
            and len(row) == 2
            and str(row[0]).strip() == ""
            and "." in str(row[1])
        ):
            packed = str(row[1]).strip()
            parts = [p.strip() for p in packed.split(".") if p.strip()]
            if len(parts) >= 2:
                # Prefer fragments containing attachment points, then longer strings
                parts.sort(key=lambda s: (("*" in s), len(s)), reverse=True)
                a, b = parts[0], parts[1]
                if a != b:
                    core, side = (a, b) if len(a) >= len(b) else (b, a)
                    return core, side
            return None
    except Exception:
        pass

    # Case 2: General fallback: select two best fragment-like strings from the tuple
    def _is_valid_fragment_smiles(text: str) -> bool:
        try:
            return bool(text) and Chem.MolFromSmiles(text) is not None
        except Exception:
            return False

    def _score_candidate(text: str) -> int:
        score = 0
        if "*" in text:
            score += 8
            if "." in text:
                score += 2
        if _is_valid_fragment_smiles(text):
            score += 4
        if text and all(ch.isdigit() for ch in text):
            score -= 6
        return score

    values = []
    try:
        values = [str(v).strip() for v in row]
    except Exception:
        return None

    candidates = []
    for v in values:
        if not v:
            continue
        looks_like_fragment = ("*" in v) or _is_valid_fragment_smiles(v)
        if not looks_like_fragment:
            continue
        candidates.append((_score_candidate(v), len(v), v))

    if len(candidates) < 2:
        return None

    candidates.sort(reverse=True)
    frag_a = candidates[0][2]
    frag_b = candidates[1][2]

    if frag_a == frag_b:
        for _, _, v in candidates[2:]:
            if v != frag_a:
                frag_b = v
                break
    if frag_a == frag_b:
        return None

    core, side = (frag_a, frag_b) if len(frag_a) >= len(frag_b) else (frag_b, frag_a)
    return core, side


def extract_rules(
    smiles: list[str],
    max_cuts: int,
    max_cut_bonds: int,
    pattern: str | None,
) -> tuple[list[tuple[str, str, str]], int, int, str, int, int, int]:
    from rdkit import Chem
    from rdkit.Chem import rdMMPA

    fragment_rows: list[tuple[str, str]] = []
    debug_rows_printed = 0
    n_valid_molecules = 0
    n_fragmentable_molecules = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        n_valid_molecules += 1
        kwargs = {
            "maxCuts": max_cuts,
            "maxCutBonds": max_cut_bonds,
            "resultsAsMols": False,
        }
        if pattern:
            kwargs["pattern"] = pattern
        mol_fragments = list(rdMMPA.FragmentMol(mol, **kwargs))
        if mol_fragments:
            n_fragmentable_molecules += 1
        for row in mol_fragments:
            if not isinstance(row, tuple):
                continue
            parsed = parse_fragment_tuple(row, Chem)
            if debug_rows_printed < 5:
                print(
                    "Fragment tuple debug: "
                    f"len={len(row)}, tuple={tuple(str(v) for v in row)}, parsed={parsed}",
                    file=sys.stderr,
                )
                debug_rows_printed += 1
            if parsed is None:
                continue
            core, sidechain = parsed
            fragment_rows.append((core, sidechain))

    rules_parsed, by_core = _collect_rules_from_fragment_rows(fragment_rows)
    n_unique_cores = len(by_core)
    n_cores_with_2plus = sum(1 for sides in by_core.values() if len(set(sides)) >= 2)
    return (
        rules_parsed,
        n_valid_molecules,
        n_fragmentable_molecules,
        "parsed",
        len(fragment_rows),
        n_unique_cores,
        n_cores_with_2plus,
    )


def run_selftest_if_requested(args: argparse.Namespace) -> None:
    if os.getenv("MMPA_SELFTEST") != "1":
        return
    import pandas as pd

    df = pd.read_parquet(args.input_parquet)
    smi_col = pick_col(df, ["canonical_smiles", "smiles"])
    sample_smiles = df[smi_col].dropna().astype(str).head(200).tolist()
    rules_raw, *_ = extract_rules(
        sample_smiles,
        max_cuts=1,
        max_cut_bonds=20,
        pattern="[!#1]!@!=!#[!#1]",
    )
    if len(rules_raw) == 0:
        rules_raw, *_ = extract_rules(
            sample_smiles,
            max_cuts=1,
            max_cut_bonds=20,
            pattern=None,
        )
    if len(rules_raw) == 0:
        raise SystemExit(
            "MMPA self-test failed: 0 rules from first 200 molecules. "
            "Adjust --max_cut_bonds and/or --cut_pattern fragmentation settings."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MMP transformation rules for counterfactual generation.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min_support", type=int, default=3)
    parser.add_argument("--max_cuts", type=int, default=1)
    parser.add_argument("--max_cut_bonds", type=int, default=20)
    parser.add_argument("--cut_pattern", type=str, default="[!#1]!@!=!#[!#1]")
    add_plot_style_args(parser)
    args = parser.parse_args()
    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit("pandas is required to run build_mmp_rules.py") from exc
    try:
        from rdkit.Chem import rdMMPA  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "RDKit rdMMPA is not available. MMP rule extraction cannot run. "
            "Install a full RDKit build that includes rdMMPA, or use a fallback rule generator."
        ) from e

    run_selftest_if_requested(args)

    outdir = Path(args.outdir)
    rules_dir = outdir / "rules"
    fig_dir = outdir / "figures"
    rules_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    smi_col = pick_col(df, ["canonical_smiles", "smiles"])
    p_col = pick_col(df, ["pIC50", "pic50", "y"], required=False)
    series_col = pick_col(df, ["series_id", "series", "scaffold_id"], required=False)

    input_smiles = df[smi_col].dropna().astype(str).tolist()
    extraction_attempts = [
        {
            "label": "cli_settings",
            "max_cuts": args.max_cuts,
            "max_cut_bonds": args.max_cut_bonds,
            "pattern": args.cut_pattern,
        },
        {
            "label": "rdkit_default_pattern",
            "max_cuts": args.max_cuts,
            "max_cut_bonds": args.max_cut_bonds,
            "pattern": None,
        },
        {
            "label": "single_cut_bond_fallback",
            "max_cuts": args.max_cuts,
            "max_cut_bonds": 1,
            "pattern": None,
        },
    ]
    selected_attempt = extraction_attempts[0]
    for attempt in extraction_attempts:
        (
            rules_raw,
            n_valid_molecules,
            n_fragmentable_molecules,
            tuple_order,
            n_fragment_rows,
            n_unique_cores,
            n_cores_with_2plus,
        ) = extract_rules(
            input_smiles,
            max_cuts=attempt["max_cuts"],
            max_cut_bonds=attempt["max_cut_bonds"],
            pattern=attempt["pattern"],
        )
        selected_attempt = attempt
        if rules_raw:
            break
    print(f"n_input_rows={len(df)}", file=sys.stderr)
    print(f"n_valid_rdkit_mols={n_valid_molecules}", file=sys.stderr)
    print(f"n_fragmentable_mols={n_fragmentable_molecules}", file=sys.stderr)
    print(f"n_candidate_pairs_before_support_filter={len(rules_raw)}", file=sys.stderr)
    print(
        "Rule extraction diagnostics: "
        f"n_molecules_loaded={len(input_smiles)}, "
        f"n_valid_rdkit_molecules={n_valid_molecules}, "
        f"n_fragmentable_molecules={n_fragmentable_molecules}, "
        f"n_fragment_rows={n_fragment_rows}, "
        f"n_unique_cores={n_unique_cores}, "
        f"n_cores_with_2plus={n_cores_with_2plus}, "
        f"n_candidate_pairs_pre_aggregation={len(rules_raw)}, "
        f"tuple_order_selected={tuple_order}, "
        f"fragmentation_settings_used={selected_attempt['label']}",
        file=sys.stderr,
    )
    counter = Counter((lhs, rhs, ctx) for lhs, rhs, ctx in rules_raw)

    def build_rows(min_support: int) -> list[dict[str, object]]:
        filtered_rows = []
        for idx, ((lhs, rhs, ctx), support) in enumerate(counter.items(), start=1):
            if support < min_support:
                continue
            filtered_rows.append(
                {
                    "rule_id": f"R{idx:06d}",
                    "lhs_fragment": lhs,
                    "rhs_fragment": rhs,
                    "context_fragment": ctx,
                    "transformation": f"{lhs}>>{rhs}",
                    "support_count": int(support),
                    "median_delta_pIC50": 0.0,
                    "series_scope": "global" if series_col is None else "series_aware",
                }
            )
        return filtered_rows

    rows = build_rows(args.min_support)
    n_rules_raw = len(counter)
    n_rules_after_min_support = len(rows)
    min_support_used = args.min_support
    if n_rules_after_min_support == 0 and args.min_support > 2:
        rows = build_rows(2)
        n_rules_after_min_support = len(rows)
        min_support_used = 2
        print("Rule extraction fallback triggered (min_support=2)", file=sys.stderr)
    cut_pattern_used = selected_attempt["pattern"] or "<rdkit_default>"
    max_cut_bonds_used = selected_attempt["max_cut_bonds"]
    print(f"n_rules_raw={n_rules_raw}", file=sys.stderr)
    if n_rules_raw == 0:
        raise SystemExit(
            "n_rules_raw=0. No raw MMP rules were extracted before support filtering. "
            f"n_valid_rdkit_mols={n_valid_molecules}, "
            f"n_fragmentable_mols={n_fragmentable_molecules}. "
            "Check input chemistry quality and rdMMPA fragmentation settings."
        )

    print(
            "Rule extraction counts: "
            f"n_rules_raw={n_rules_raw}, "
            f"n_rules_after_min_support={n_rules_after_min_support}, "
            f"min_support_used={min_support_used}, "
            f"max_cut_bonds_used={max_cut_bonds_used}, "
            f"cut_pattern_used={cut_pattern_used}",
        file=sys.stderr,
    )
    rules_df = pd.DataFrame(rows).sort_values("support_count", ascending=False) if rows else pd.DataFrame(columns=[
        "rule_id", "lhs_fragment", "rhs_fragment", "context_fragment", "transformation", "support_count", "median_delta_pIC50", "series_scope"
    ])

    rules_path = rules_dir / "mmp_rules.parquet"
    stats_path = rules_dir / "rule_stats.csv"
    prov_path = rules_dir / "rule_provenance.json"
    rules_df.to_parquet(rules_path, index=False)

    stats = pd.DataFrame(
        {
            "metric": [
                "n_input_rows",
                "n_unique_smiles",
                "n_rules_before_filter",
                "n_rules_after_filter",
                "n_rules_raw",
                "n_rules_after_min_support",
                "requested_min_support",
                "min_support_used",
                "max_cuts",
                "max_cut_bonds",
                "cut_pattern",
                "max_cut_bonds_used",
                "cut_pattern_used",
            ],
            "value": [
                len(df),
                df[smi_col].nunique(),
                n_rules_raw,
                len(rules_df),
                n_rules_raw,
                n_rules_after_min_support,
                args.min_support,
                min_support_used,
                args.max_cuts,
                args.max_cut_bonds,
                args.cut_pattern,
                max_cut_bonds_used,
                cut_pattern_used,
            ],
        }
    )
    stats.to_csv(stats_path, index=False)

    style = style_from_args(args)
    configure_matplotlib(style, svg=True)
    import matplotlib.pyplot as plt

    fig_path = fig_dir / "fig_edit_type_distribution.svg"
    top = rules_df.head(15).copy()
    if top.empty:
        fig, ax = plt.subplots(figsize=(4, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, "NO DATA (0 rows)", ha="center", va="center", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        top["rule_label"] = top["lhs_fragment"].str.slice(0, 12) + "→" + top["rhs_fragment"].str.slice(0, 12)
        ax.barh(top["rule_label"], top["support_count"])
        style_axis(ax, style, title="Edit Type Distribution", xlabel="Support count", ylabel="Rule")
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)

    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        "script_hash": sha256_file(Path(__file__)),
        "input_dataset_hash": sha256_file(Path(args.input_parquet)),
        "n_rules": int(len(rules_df)),
        "requested_min_support": int(args.min_support),
        "min_support_used": int(min_support_used),
        "max_cut_bonds_used": int(max_cut_bonds_used),
        "cut_pattern_used": cut_pattern_used,
        "fragmentation_attempt_label": selected_attempt["label"],
    }
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote rules to {rules_path}")


if __name__ == "__main__":
    main()
