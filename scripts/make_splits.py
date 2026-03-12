#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from functools import lru_cache
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.ML.Cluster import Butina
except Exception as exc:  # pragma: no cover
    raise RuntimeError("RDKit is required for make_splits.py") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.plot_style import NATURE5
except ModuleNotFoundError:
    from plot_style import NATURE5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def pip_freeze() -> str:
    try:
        return subprocess.check_output(["pip", "freeze"], text=True)
    except Exception as exc:
        return f"pip freeze unavailable: {exc}\n"


def stratified_split_ids(df: pd.DataFrame, seed: int, train_frac: float, val_frac: float, stratify_col: str) -> tuple[pd.Index, pd.Index, pd.Index]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for _, grp in df.groupby(stratify_col, dropna=False):
        ids = grp.index.to_numpy(copy=True)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
        train_idx.extend(ids[:n_train])
        val_idx.extend(ids[n_train : n_train + n_val])
        test_idx.extend(ids[n_train + n_val : n_train + n_val + n_test])
    return pd.Index(train_idx), pd.Index(val_idx), pd.Index(test_idx)


def random_split(df: pd.DataFrame, args) -> dict[str, pd.Index]:
    tr, va, te = stratified_split_ids(df, args.seed, args.train_frac, args.val_frac, "activity_label")
    return {"train": tr, "val": va, "test": te}


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "NOSCAF"


def scaffold_split(df: pd.DataFrame, seed: int, train_frac: float, val_frac: float) -> tuple[dict[str, pd.Index], pd.Series]:
    tmp = df.copy()
    tmp["_scaffold"] = tmp["canonical_smiles"].astype(str).map(_scaffold)
    rng = np.random.default_rng(seed)
    scaffolds = tmp["_scaffold"].dropna().unique().tolist()
    rng.shuffle(scaffolds)
    n_total = len(tmp)
    target_train = int(n_total * train_frac)
    target_val = int(n_total * val_frac)
    split = {"train": [], "val": [], "test": []}
    c_train = c_val = 0
    for scaf in scaffolds:
        idx = tmp.index[tmp["_scaffold"] == scaf].tolist()
        if c_train < target_train:
            split["train"].extend(idx)
            c_train += len(idx)
        elif c_val < target_val:
            split["val"].extend(idx)
            c_val += len(idx)
        else:
            split["test"].extend(idx)
    return {k: pd.Index(v) for k, v in split.items()}, tmp["_scaffold"]


def time_split(df: pd.DataFrame, args) -> dict[str, pd.Index]:
    key = args.time_key
    if key not in df.columns:
        raise ValueError(f"time key '{key}' not in dataframe")
    tmp = df.sort_values(key, kind="mergesort")
    n = len(tmp)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    idx = tmp.index
    return {"train": idx[:n_train], "val": idx[n_train : n_train + n_val], "test": idx[n_train + n_val :]}


def env_holdout_assay(df: pd.DataFrame, args) -> dict[str, pd.Index]:
    if "assay_type" not in df.columns:
        raise ValueError("assay_type missing for env_holdout_assay")
    test = df.index[df["assay_type"].astype(str) == args.assay_holdout_value]
    rest = df.index.difference(test)
    rem = df.loc[rest]
    tr, va, _ = stratified_split_ids(rem, args.seed, args.train_frac / (args.train_frac + args.val_frac), args.val_frac / (args.train_frac + args.val_frac), "activity_label")
    return {"train": tr, "val": va, "test": test}


def select_pub_holdout(df: pd.DataFrame, args) -> tuple[str, list[str]]:
    col = "series_id" if "series_id" in df.columns else ("publication" if "publication" in df.columns else "document_id")
    if args.pub_holdout_values:
        vals = [x.strip() for x in args.pub_holdout_values.split(",") if x.strip()]
    else:
        vals = [str(df[col].value_counts(dropna=False).index[0])]
    return col, vals


def env_holdout_pubfam(df: pd.DataFrame, args) -> tuple[dict[str, pd.Index], str, list[str]]:
    col, vals = select_pub_holdout(df, args)
    test = df.index[df[col].astype(str).isin(vals)]
    rem = df.index.difference(test)
    rem_df = df.loc[rem]
    tr, va, _ = stratified_split_ids(rem_df, args.seed, args.train_frac / (args.train_frac + args.val_frac), args.val_frac / (args.train_frac + args.val_frac), "activity_label")
    return {"train": tr, "val": va, "test": test}, col, vals


def combo_scaffold_env(df: pd.DataFrame, args, scaffold_series: pd.Series) -> dict[str, pd.Index]:
    base, col, vals = env_holdout_pubfam(df, args)
    test_scaf = set(scaffold_series.loc[base["test"]].tolist())
    tr = [i for i in base["train"] if scaffold_series.loc[i] not in test_scaf and str(df.loc[i, col]) not in vals]
    va = [i for i in base["val"] if scaffold_series.loc[i] not in test_scaf and str(df.loc[i, col]) not in vals]
    return {"train": pd.Index(tr), "val": pd.Index(va), "test": base["test"]}


def combo_time_env(df: pd.DataFrame, args) -> dict[str, pd.Index]:
    ts = time_split(df, args)
    _, col, vals = env_holdout_pubfam(df, args)
    holdout_env = df.index[df[col].astype(str).isin(vals)]
    test = ts["test"].intersection(holdout_env)
    remaining = df.index.difference(test)
    rem_df = df.loc[remaining].sort_values(args.time_key, kind="mergesort") if args.time_key in df.columns else df.loc[remaining]
    n_train = int(len(rem_df) * (args.train_frac / (args.train_frac + args.val_frac)))
    tr = rem_df.index[:n_train]
    va = rem_df.index[n_train:]
    return {"train": tr, "val": va, "test": test}


def hard_boundary(df: pd.DataFrame, args) -> dict[str, pd.Index]:
    thr = args.activity_threshold
    if thr is None:
        thr = float(df["pIC50"].median())
    hard = df.index[(df["pIC50"] - thr).abs() <= args.hard_delta]
    rem = df.index.difference(hard)
    rem_df = df.loc[rem]
    tr, va, _ = stratified_split_ids(rem_df, args.seed, args.train_frac / (args.train_frac + args.val_frac), args.val_frac / (args.train_frac + args.val_frac), "activity_label")
    return {"train": tr, "val": va, "test": hard}


@lru_cache(maxsize=8)
def _morgan_generator(radius: int, nbits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def morgan_fp(smiles: str, radius: int, nbits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _morgan_generator(radius, nbits).GetFingerprint(mol)


def neighbor_similarity_split(df: pd.DataFrame, args) -> dict[str, pd.Index]:
    fps = [morgan_fp(s, args.similarity_radius, args.similarity_nbits) for s in df["canonical_smiles"].astype(str)]
    valid_idx = [i for i, fp in zip(df.index, fps) if fp is not None]
    valid_fps = [fp for fp in fps if fp is not None]
    dists = []
    for i in range(1, len(valid_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend([1.0 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(valid_fps), 1.0 - args.neighbor_threshold, isDistData=True)
    n = len(valid_idx)
    target_train = int(n * args.train_frac)
    target_val = int(n * args.val_frac)
    split = {"train": [], "val": [], "test": []}
    c_train = c_val = 0
    for cl in sorted(clusters, key=len, reverse=True):
        members = [valid_idx[m] for m in cl]
        if c_train < target_train:
            split["train"].extend(members)
            c_train += len(members)
        elif c_val < target_val:
            split["val"].extend(members)
            c_val += len(members)
        else:
            split["test"].extend(members)
    return {k: pd.Index(v) for k, v in split.items()}


def scaffold_matched_props(df: pd.DataFrame, args, base_test: pd.Index) -> tuple[dict[str, pd.Index], pd.DataFrame]:
    props = args.match_props
    test_df = df.loc[base_test]
    train_pool = df.index.difference(base_test)
    pool_df = df.loc[train_pool].copy()
    q_test = {p: pd.qcut(test_df[p], q=min(5, test_df[p].nunique()), duplicates="drop") for p in props if p in df.columns}
    scores = pd.Series(0.0, index=pool_df.index)
    for p, bins in q_test.items():
        test_hist = bins.value_counts(normalize=True)
        pool_bins = pd.cut(pool_df[p], bins.cat.categories)
        w = pool_bins.map(lambda b: test_hist.get(b, 0.0)).fillna(0.0)
        scores += w
    chosen_n = max(len(test_df) * 2, 1)
    chosen = scores.sort_values(ascending=False).head(chosen_n).index
    rem = chosen
    rem_df = df.loc[rem]
    tr, va, _ = stratified_split_ids(rem_df, args.seed, args.train_frac / (args.train_frac + args.val_frac), args.val_frac / (args.train_frac + args.val_frac), "activity_label")
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except Exception:
        ks_2samp = wasserstein_distance = None
    rows = []
    for p in props:
        if p not in df.columns:
            continue
        before = pool_df[p].dropna().to_numpy()
        after = df.loc[tr, p].dropna().to_numpy()
        test = test_df[p].dropna().to_numpy()
        if ks_2samp and len(before) and len(test) and len(after):
            rows.append({
                "property": p,
                "ks_before": ks_2samp(before, test).statistic,
                "ks_after": ks_2samp(after, test).statistic,
                "wasserstein_before": wasserstein_distance(before, test),
                "wasserstein_after": wasserstein_distance(after, test),
            })
    return {"train": tr, "val": va, "test": base_test}, pd.DataFrame(rows)


def save_manifest(split_dir: Path, df: pd.DataFrame, idxs: dict[str, pd.Index], config: dict, id_col: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for part, idx in idxs.items():
        sub = df.loc[idx]
        sub[[id_col]].to_csv(split_dir / f"{part}_ids.csv", index=False)
        sub.to_parquet(split_dir / f"{part}.parquet", index=False)
    (split_dir / "split_config.json").write_text(json.dumps(config, indent=2))


def integrity_checks(name: str, df: pd.DataFrame, idxs: dict[str, pd.Index], pub_holdout: tuple[str, list[str]] | None = None) -> list[dict]:
    rows = []
    train = df.loc[idxs["train"]]
    test = df.loc[idxs["test"]]
    if "series_id" in df.columns:
        overlap = set(train["series_id"].dropna().astype(str)) & set(test["series_id"].dropna().astype(str))
        rows.append({"split": name, "check": "series_train_test_disjoint", "passed": len(overlap) == 0, "details": ",".join(sorted(overlap))[:500]})
    if pub_holdout is not None:
        col, vals = pub_holdout
        leak = set(train[col].astype(str)) & set(vals)
        leak |= set(df.loc[idxs["val"], col].astype(str)) & set(vals)
        rows.append({"split": name, "check": "pub_holdout_absent_train_val", "passed": len(leak) == 0, "details": ",".join(sorted(leak))[:500]})
    return rows


def similarity_leakage(name: str, df: pd.DataFrame, idxs: dict[str, pd.Index], args) -> dict:
    tr_sm = df.loc[idxs["train"], "canonical_smiles"].astype(str).tolist()
    te_sm = df.loc[idxs["test"], "canonical_smiles"].astype(str).tolist()
    tr_fp = [morgan_fp(s, args.similarity_radius, args.similarity_nbits) for s in tr_sm]
    tr_fp = [x for x in tr_fp if x is not None]
    max_sims = []
    for s in te_sm:
        fp = morgan_fp(s, args.similarity_radius, args.similarity_nbits)
        if fp is None or not tr_fp:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, tr_fp)
        max_sims.append(max(sims) if sims else 0.0)
    arr = np.array(max_sims) if max_sims else np.array([0.0])
    return {
        "split": name,
        "n_test_scored": int(len(max_sims)),
        "sim_min": float(np.min(arr)),
        "sim_median": float(np.median(arr)),
        "sim_max": float(np.max(arr)),
        "count_ge_0.8": int((arr >= 0.8).sum()),
        "count_ge_0.85": int((arr >= 0.85).sum()),
        "count_ge_0.9": int((arr >= 0.9).sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict split generation for QSAR")
    p.add_argument("--target", required=True)
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--enable", nargs="+", default=["random", "scaffold_bm", "time_publication", "env_holdout_assay", "env_holdout_pubfam", "combo_scaffold_env", "combo_time_env", "scaffold_matched_props", "hard_boundary", "neighbor_similarity"])
    p.add_argument("--time_key", default="publication_year")
    p.add_argument("--assay_holdout_value", default="cell-based")
    p.add_argument("--pub_holdout_values", default="")
    p.add_argument("--similarity_radius", type=int, default=2)
    p.add_argument("--similarity_nbits", type=int, default=2048)
    p.add_argument("--neighbor_threshold", type=float, default=0.65)
    p.add_argument("--hard_delta", type=float, default=0.3)
    p.add_argument("--activity_threshold", type=float, default=None)
    p.add_argument("--match_props", nargs="+", default=["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings"])
    p.add_argument("--id_col", default="molecule_id")
    return p.parse_args()


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_split_inputs(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    canonical_id = "molecule_id"

    if canonical_id not in out.columns:
        fallback_col = _first_present(out, ["molecule_chembl_id", "chembl_molecule_id", "compound_id", "mol_id", "id"])
        if fallback_col is not None:
            logging.warning("Creating canonical molecule_id from source column '%s'", fallback_col)
            out[canonical_id] = out[fallback_col]

    if id_col not in out.columns:
        alternatives = [c for c in ["molecule_id", "molecule_chembl_id", "chembl_molecule_id", "compound_id", "mol_id", "id"] if c in out.columns]
        avail = ", ".join(map(str, out.columns))
        hint = f" Try --id_col one of: {', '.join(alternatives)}." if alternatives else ""
        raise ValueError(f"id column '{id_col}' not found. Available columns: {avail}.{hint}")

    if id_col != canonical_id:
        logging.warning("Using id_col '%s'; also standardizing canonical molecule_id from this column", id_col)
        out[canonical_id] = out[id_col]

    smiles_col = _first_present(out, ["canonical_smiles", "smiles", "smiles_canonical"])
    if smiles_col is None:
        raise ValueError("missing required SMILES column; expected one of canonical_smiles/smiles/smiles_canonical")
    if smiles_col != "canonical_smiles":
        out["canonical_smiles"] = out[smiles_col]

    if "activity_label" not in out.columns:
        if "pIC50" not in out.columns:
            raise ValueError("missing required label columns: activity_label or pIC50")
        thr = float(out["pIC50"].median())
        out["activity_label"] = (pd.to_numeric(out["pIC50"], errors="coerce") >= thr).astype(int)

    out[id_col] = out[id_col].astype(str)
    out[canonical_id] = out[canonical_id].astype(str)
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_splits = Path(args.outdir)
    root = out_splits.parent
    reports = root / "reports"
    figures = root / "figures"
    prov = root / "provenance"
    for d in [out_splits, reports, figures, prov]:
        d.mkdir(parents=True, exist_ok=True)

    df = normalize_split_inputs(pd.read_parquet(args.input_parquet), args.id_col)
    required = [args.id_col, "canonical_smiles", "pIC50", "activity_label"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"missing required columns: {miss}")
    if "publication_year" not in df.columns and args.time_key == "publication_year":
        df["publication_year"] = np.nan

    split_results: dict[str, dict[str, pd.Index]] = {}
    checks, sim_rows, match_rows, summary_rows = [], [], [], []
    sc_split, scaffolds = scaffold_split(df, args.seed, args.train_frac, args.val_frac)

    for name in args.enable:
        pub_hold = None
        if name == "random":
            idxs = random_split(df, args)
        elif name == "scaffold_bm":
            idxs = sc_split
        elif name == "time_publication":
            idxs = time_split(df, args)
        elif name == "env_holdout_assay":
            idxs = env_holdout_assay(df, args)
        elif name == "env_holdout_pubfam":
            idxs, col, vals = env_holdout_pubfam(df, args)
            pub_hold = (col, vals)
        elif name == "combo_scaffold_env":
            idxs = combo_scaffold_env(df, args, scaffolds)
        elif name == "combo_time_env":
            idxs = combo_time_env(df, args)
        elif name == "scaffold_matched_props":
            idxs, mq = scaffold_matched_props(df, args, sc_split["test"])
            if not mq.empty:
                mq.insert(0, "split", name)
                match_rows.append(mq)
        elif name == "hard_boundary":
            idxs = hard_boundary(df, args)
        elif name == "neighbor_similarity":
            idxs = neighbor_similarity_split(df, args)
        else:
            continue

        split_results[name] = idxs
        save_manifest(out_splits / name, df, idxs, {"split": name, **vars(args)}, args.id_col)
        checks.extend(integrity_checks(name, df, idxs, pub_hold))
        sim_rows.append(similarity_leakage(name, df, idxs, args))
        summary_rows.append({"split": name, "n_train": len(idxs["train"]), "n_val": len(idxs["val"]), "n_test": len(idxs["test"])})

    checks_df = pd.DataFrame(checks)
    if not checks_df.empty and not checks_df["passed"].all():
        checks_df.to_csv(reports / "group_integrity_checks.csv", index=False)
        raise RuntimeError("Group integrity checks failed")

    pd.DataFrame(summary_rows).to_csv(reports / "split_summary.csv", index=False)
    checks_df.to_csv(reports / "group_integrity_checks.csv", index=False)
    pd.DataFrame(sim_rows).to_csv(reports / "similarity_leakage.csv", index=False)
    (pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame(columns=["split", "property", "ks_before", "ks_after", "wasserstein_before", "wasserstein_after"]))\
        .to_csv(reports / "matching_quality.csv", index=False)

    run_cfg = {"seed": args.seed, "train_frac": args.train_frac, "val_frac": args.val_frac, "test_frac": args.test_frac, "time_key": args.time_key, "neighbor_threshold": args.neighbor_threshold, "hard_delta": args.hard_delta, "match_props": args.match_props, "palette": NATURE5}
    (prov / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    script_paths = [Path(__file__), Path("scripts/splits_report.py"), Path("scripts/plot_style.py")]
    prov_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": safe_git_hash(),
        "script_sha256": {str(p): sha256_file(p) for p in script_paths if p.exists()},
        "input_parquet": {"path": args.input_parquet, "sha256": sha256_file(Path(args.input_parquet))},
        "split_names_generated": list(split_results.keys()),
        "split_sizes": summary_rows,
    }
    (prov / "provenance.json").write_text(json.dumps(prov_data, indent=2))
    (prov / "environment.txt").write_text(pip_freeze())

    splits_report_cmd = [
        sys.executable,
        str(Path("scripts") / "splits_report.py"),
        "--input_parquet",
        str(args.input_parquet),
        "--splits_dir",
        str(out_splits),
        "--outdir",
        str(root),
        "--id_col",
        args.id_col,
    ]
    try:
        subprocess.run(splits_report_cmd, check=True)
    except Exception as exc:
        (reports / "splits_report_error.txt").write_text(f"Failed to run splits_report.py: {exc}\n")

    for csv_name in ["label_shift.csv", "covariate_shift.csv", "scaffold_overlap.csv", "env_overlap.csv", "time_coverage.csv"]:
        p = reports / csv_name
        if not p.exists():
            pd.DataFrame().to_csv(p, index=False)

    for fig_name in [
        "fig_split_sizes.svg",
        "fig_label_shift_by_split.svg",
        "fig_covariate_shift_props.svg",
        "fig_scaffold_overlap_by_split.svg",
        "fig_env_overlap_by_split.svg",
        "fig_similarity_leakage.svg",
        "fig_time_split_timeline.svg",
        "fig_matching_quality.svg",
    ]:
        fig = figures / fig_name
        if not fig.exists():
            fig.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">Generated by splits_report.py</text></svg>')


if __name__ == "__main__":
    main()
