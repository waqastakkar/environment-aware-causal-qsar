#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None
from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, q=(2.5, 97.5)) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        means.append(float(np.mean(values[idx])))
    lo, hi = np.percentile(means, q)
    return float(lo), float(hi)


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(np.square(err)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    spearman = float(y_true.corr(y_pred, method="spearman"))
    pearson = float(y_true.corr(y_pred, method="pearson"))
    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman, "pearson": pearson}


def classification_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if y_true.nunique() > 1 else np.nan,
        "f1": float(f1_score(y_true, y_pred)) if y_true.nunique() > 1 else np.nan,
        "balanced_acc": float(((y_pred[y_true == 1] == 1).mean() + (y_pred[y_true == 0] == 0).mean()) / 2),
    }
    return out


def expected_calibration_error(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> tuple[float, pd.DataFrame]:
    df = pd.DataFrame({"y": y_true.astype(float), "p": y_prob.astype(float)}).dropna()
    if df.empty:
        return np.nan, pd.DataFrame(columns=["bin", "acc", "conf", "count"])
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grp = df.groupby("bin", observed=False)
    cal = grp.agg(acc=("y", "mean"), conf=("p", "mean"), count=("y", "size")).reset_index()
    cal["gap"] = (cal["acc"] - cal["conf"]).abs()
    ece = float((cal["gap"] * cal["count"] / max(1, cal["count"].sum())).sum())
    return ece, cal


def scan_runs(runs_root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(runs_root.glob("**/predictions/test_predictions.parquet")):
        run_dir = p.parent.parent
        split = run_dir.parent.name
        run_id = run_dir.name
        rows.append({"split": split, "run_id": run_id, "run_dir": str(run_dir), "pred_path": str(p)})
    return pd.DataFrame(rows)


def ensure_cols(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    pred_col = None
    for c in ["y_pred", "prediction", "yhat", "pred", label_col + "_pred"]:
        if c in df.columns:
            pred_col = c
            break
    if pred_col is None and label_col in df.columns:
        pred_col = label_col
    if pred_col is None:
        raise ValueError("Could not find prediction column in predictions file.")
    out = df.copy()
    # Preserve canonical columns when they already exist in prediction artifacts.
    # Some training outputs save `y_true`/`y_pred` only (without raw `label_col`),
    # and overwriting `y_true` with NaN would make all downstream metrics empty.
    if "y_true" not in out.columns and label_col in out.columns:
        out["y_true"] = out[label_col]
    elif "y_true" not in out.columns:
        out["y_true"] = np.nan
    out["y_pred"] = out[pred_col]
    return out




def canonicalize_molecule_id(df: pd.DataFrame, context: str) -> pd.DataFrame:
    out = df.copy()
    if "molecule_id" in out.columns:
        out["molecule_id"] = out["molecule_id"].astype(str)
        return out

    for col in ["molecule_chembl_id", "chembl_molecule_id", "compound_id", "mol_id", "id"]:
        if col in out.columns:
            logging.warning("%s: creating canonical molecule_id from source column '%s'", context, col)
            out["molecule_id"] = out[col].astype(str)
            return out

    logging.warning("%s: no molecule identifier columns found", context)
    return out


def resolve_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    for key in ["molecule_id", "molecule_chembl_id", "chembl_molecule_id", "compound_id"]:
        if key in left.columns and key in right.columns:
            return key
    return None


def detect_counterfactual_dir(outputs_root: Path, target: str) -> tuple[Path | None, list[Path]]:
    checked = [
        outputs_root / "step7" / "candidates",
        outputs_root / "step7" / target / "candidates",
    ]
    for candidate in checked:
        if candidate.is_dir():
            return candidate, checked

    ranked_paths = sorted((outputs_root / "step7").glob("**/ranked_topk.parquet"))
    if len(ranked_paths) == 1:
        return ranked_paths[0].parent, checked + ranked_paths
    if len(ranked_paths) > 1:
        target_ranked = [p for p in ranked_paths if target in p.as_posix()]
        if len(target_ranked) == 1:
            return target_ranked[0].parent, checked + ranked_paths
        msg = "\n  - ".join(str(p.parent) for p in ranked_paths)
        raise SystemExit(
            "Ambiguous counterfactual directory auto-detection under step7. "
            "Provide --counterfactual_dir explicitly.\n"
            f"  - {msg}"
        )
    return None, checked


def resolve_counterfactual_source(cf_dir: Path) -> tuple[Path | None, str | None, list[Path]]:
    ranked = cf_dir / "ranked_topk.parquet"
    filtered = cf_dir / "filtered_counterfactuals.parquet"
    generated = cf_dir / "generated_counterfactuals.parquet"
    checked = [ranked, filtered, generated]
    for path, label in [(ranked, "ranked_topk"), (filtered, "filtered_counterfactuals"), (generated, "generated_counterfactuals")]:
        if path.exists() and path.is_file():
            return path, label, checked
    return None, None, checked


def build_cf_metrics_from_candidates(cf_path: Path) -> list[dict[str, float | str]]:
    df = pd.read_parquet(cf_path)
    if df.empty:
        return []

    delta_col = next((c for c in ["delta_yhat", "delta_pred", "delta_prediction"] if c in df.columns), None)
    if delta_col is None:
        return []

    rows = []
    if "rule_id" in df.columns:
        groups = df.groupby("rule_id")
    else:
        groups = [("all", df)]
    for rid, g in groups:
        vals = pd.to_numeric(g[delta_col], errors="coerce").dropna()
        if vals.empty:
            continue
        frac_pos = float((vals > 0).mean())
        std = float(vals.std()) if len(vals) > 1 else 0.0
        rows.append(
            {
                "run_id": cf_path.parent.name,
                "rule_id": rid,
                "fraction_positive": frac_pos,
                "delta_std": std,
                "invariance_score": frac_pos - 0.1 * std,
                "mean_delta": float(vals.mean()),
            }
        )
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8 evaluation suite")
    parser.add_argument("--target", required=True)
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--splits_dir", required=True)
    parser.add_argument("--dataset_parquet", required=True)
    parser.add_argument("--bbb_parquet")
    parser.add_argument("--counterfactual_dir", "--cf_dir", dest="counterfactual_dir", default=None)
    parser.add_argument("--counterfactuals_root")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--label_col", default="pIC50")
    parser.add_argument("--env_col", default="env_id_manual")
    parser.add_argument("--compute_envprobe", action="store_true")
    parser.add_argument("--compute_zinv_stability", action="store_true")
    parser.add_argument("--compute_cf_consistency", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    add_plot_style_args(parser)
    args = parser.parse_args()
    if np is None or pd is None:
        raise SystemExit("numpy and pandas are required to run evaluate_runs.py")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outdir = Path(args.outdir)
    outputs_root = outdir.resolve().parent

    auto_bbb_candidates = [
        outputs_root / "step3" / "data" / "bbb_annotations.parquet",
        outputs_root / "step3" / "bbb_annotations.parquet",
    ]
    if not args.bbb_parquet:
        for candidate in auto_bbb_candidates:
            if candidate.exists():
                args.bbb_parquet = str(candidate)
                logging.info("Auto-enabled CNS subset metrics with bbb_parquet=%s", candidate)
                break

    auto_cf_candidates = [
        outputs_root / "step7" / "candidates" / "ranked_topk.parquet",
        outputs_root / "step7" / args.target / "candidates" / "ranked_topk.parquet",
    ]
    if not args.compute_cf_consistency:
        if any(p.exists() for p in auto_cf_candidates):
            args.compute_cf_consistency = True
            logging.info("Auto-enabled counterfactual consistency from step7 ranked_topk.parquet")
    for sub in ["inputs_snapshot", "predictions/per_split_predictions", "reports", "figures", "provenance"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    runs = scan_runs(Path(args.runs_root))
    runs.to_csv(outdir / "inputs_snapshot" / "runs_index.csv", index=False)
    split_files = pd.DataFrame({"file": [str(p) for p in sorted(Path(args.splits_dir).glob("*.csv"))]})
    split_files.to_csv(outdir / "inputs_snapshot" / "splits_index.csv", index=False)

    dataset = canonicalize_molecule_id(pd.read_parquet(args.dataset_parquet), "dataset")
    summary_status: dict[str, tuple[str, str]] = {}
    env_col_effective = args.env_col
    if env_col_effective not in dataset.columns and env_col_effective == "env_id_manual" and "env_id" in dataset.columns:
        print("WARNING: env_col 'env_id_manual' not found in dataset; falling back to 'env_id'.")
        env_col_effective = "env_id"
    elif env_col_effective not in dataset.columns:
        print(f"WARNING: env_col '{env_col_effective}' not found in dataset and no fallback available.")
    fp = {
        "dataset_path": args.dataset_parquet,
        "rows": int(len(dataset)),
        "cols": list(dataset.columns),
        "sha256": sha256_file(Path(args.dataset_parquet)),
    }
    (outdir / "inputs_snapshot" / "dataset_fingerprint.json").write_text(json.dumps(fp, indent=2), encoding="utf-8")

    merged = []
    perf_rows = []
    perf_split_rows = []
    perf_env_rows = []
    cal_split_rows = []
    for _, r in runs.iterrows():
        pred = canonicalize_molecule_id(pd.read_parquet(r["pred_path"]), f"predictions run {r['run_id']}")
        pred = ensure_cols(pred, args.label_col)
        pred["split"] = r["split"]
        pred["run_id"] = r["run_id"]
        if env_col_effective not in pred.columns and env_col_effective == "env_id_manual" and "env_id" in pred.columns:
            print(f"WARNING: env_col 'env_id_manual' not found in predictions for run {r['run_id']}; falling back to 'env_id'.")
            env_col_effective = "env_id"
        if env_col_effective not in pred.columns and env_col_effective in dataset.columns:
            key = resolve_join_key(pred, dataset)
            if key:
                pred = pred.merge(dataset[[key, env_col_effective]], on=key, how="left")
            else:
                logging.warning("Could not backfill env column for run %s: no shared identifier column", r["run_id"])
        pred.to_parquet(outdir / "predictions" / "per_split_predictions" / f"{r['split']}__{r['run_id']}.parquet", index=False)
        merged.append(pred)

        clean = pred[["y_true", "y_pred"]].dropna()
        if clean.empty:
            continue
        if args.task == "regression":
            m = regression_metrics(clean["y_true"], clean["y_pred"])
            ci_lo, ci_hi = bootstrap_ci(np.abs(clean["y_pred"] - clean["y_true"]).values, n_boot=args.bootstrap)
            m.update({"metric": "mae", "ci_lo": ci_lo, "ci_hi": ci_hi})
        else:
            m = classification_metrics(clean["y_true"].astype(int), clean["y_pred"])
        perf_split_rows.append({"split": r["split"], "run_id": r["run_id"], **m})

        if args.task == "classification":
            ece, _ = expected_calibration_error(clean["y_true"].astype(int), clean["y_pred"])
            cal_split_rows.append({"split": r["split"], "run_id": r["run_id"], "ece": ece})

        if env_col_effective in pred.columns:
            for env, g in pred.dropna(subset=["y_true", "y_pred"]).groupby(env_col_effective):
                if len(g) < 5:
                    continue
                vals = regression_metrics(g["y_true"], g["y_pred"]) if args.task == "regression" else classification_metrics(g["y_true"].astype(int), g["y_pred"])
                perf_env_rows.append({"split": r["split"], "run_id": r["run_id"], "env": env, **vals})

    merged_df = pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()
    merged_df.to_parquet(outdir / "predictions" / "merged_test_predictions.parquet", index=False)

    perf_by_split = pd.DataFrame(perf_split_rows)
    perf_by_split.to_csv(outdir / "reports" / "performance_by_split.csv", index=False)

    if not perf_by_split.empty:
        numeric_cols = [c for c in perf_by_split.columns if c not in {"split", "run_id", "metric"}]
        performance_overall = perf_by_split[numeric_cols].mean(numeric_only=True).to_frame("value").reset_index().rename(columns={"index": "metric"})
    else:
        performance_overall = pd.DataFrame(columns=["metric", "value"])
    performance_overall.to_csv(outdir / "reports" / "performance_overall.csv", index=False)

    perf_by_env = pd.DataFrame(perf_env_rows)
    perf_by_env.to_csv(outdir / "reports" / "performance_by_env.csv", index=False)

    if args.task == "classification":
        cal_overall, cal_curve = expected_calibration_error(merged_df["y_true"].astype(int), merged_df["y_pred"])
        pd.DataFrame({"metric": ["ece"], "value": [cal_overall]}).to_csv(outdir / "reports" / "calibration_overall.csv", index=False)
        pd.DataFrame(cal_split_rows).to_csv(outdir / "reports" / "calibration_by_split.csv", index=False)
        cal_curve.to_csv(outdir / "reports" / "calibration_by_cns.csv", index=False)
    else:
        abs_err = (merged_df["y_pred"] - merged_df["y_true"]).abs() if not merged_df.empty else pd.Series(dtype=float)
        pd.DataFrame({"metric": ["mae"], "value": [abs_err.mean() if len(abs_err) else np.nan]}).to_csv(outdir / "reports" / "calibration_overall.csv", index=False)
        pd.DataFrame(columns=["split", "run_id", "ece"]).to_csv(outdir / "reports" / "calibration_by_split.csv", index=False)
        pd.DataFrame(columns=["bin", "acc", "conf", "count"]).to_csv(outdir / "reports" / "calibration_by_cns.csv", index=False)

    cns_metrics = []
    if args.bbb_parquet and Path(args.bbb_parquet).exists() and not merged_df.empty:
        bbb = canonicalize_molecule_id(pd.read_parquet(args.bbb_parquet), "bbb_parquet")
        key = resolve_join_key(merged_df, bbb)
        if key:
            joined = merged_df.merge(bbb, on=key, how="left")
            cns_col = "cns_like" if "cns_like" in joined.columns else None
            mpo_col = "cns_mpo" if "cns_mpo" in joined.columns else None
            if cns_col:
                for grp, g in joined.dropna(subset=[cns_col, "y_true", "y_pred"]).groupby(cns_col):
                    vals = regression_metrics(g["y_true"], g["y_pred"]) if args.task == "regression" else classification_metrics(g["y_true"].astype(int), g["y_pred"])
                    cns_metrics.append({"subset": f"cns_like={grp}", **vals})
            if mpo_col:
                bins = pd.cut(joined[mpo_col], [-np.inf, 2, 4, 6, np.inf])
                for grp, g in joined.dropna(subset=["y_true", "y_pred"]).groupby(bins):
                    if len(g) < 5:
                        continue
                    vals = regression_metrics(g["y_true"], g["y_pred"]) if args.task == "regression" else classification_metrics(g["y_true"].astype(int), g["y_pred"])
                    cns_metrics.append({"subset": f"mpo_bin={grp}", **vals})
        summary_status["cns_subset_metrics"] = ("RAN", "computed from bbb_parquet")
    else:
        reason = "bbb_parquet missing/unreadable" if not args.bbb_parquet or not Path(args.bbb_parquet).exists() else "no merged predictions"
        print(f"WARNING: CNS subset metrics skipped ({reason}).")
        cns_metrics = [{"status": "SKIPPED", "reason": reason}]
        summary_status["cns_subset_metrics"] = ("SKIPPED", reason)
    pd.DataFrame(cns_metrics).to_csv(outdir / "reports" / "cns_subset_metrics.csv", index=False)

    envprobe_rows = []
    if args.compute_envprobe and env_col_effective in merged_df.columns:
        zinv_cols = [c for c in merged_df.columns if c.startswith("z_inv")]
        zspu_cols = [c for c in merged_df.columns if c.startswith("z_spu")]
        h_cols = [c for c in merged_df.columns if c.startswith("h_")]
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        for label, cols in [("z_inv", zinv_cols), ("z_spu", zspu_cols), ("h", h_cols)]:
            if not cols:
                envprobe_rows.append({"embedding": label, "accuracy": np.nan, "n_features": 0})
                continue
            probe_df = merged_df.dropna(subset=cols + [env_col_effective]).copy()
            if probe_df.empty or probe_df[env_col_effective].nunique() < 2:
                envprobe_rows.append({"embedding": label, "accuracy": np.nan, "n_features": len(cols)})
                continue
            X_train, X_test, y_train, y_test = train_test_split(probe_df[cols].values, probe_df[env_col_effective].values, test_size=0.25, random_state=42, stratify=probe_df[env_col_effective].values)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            envprobe_rows.append({"embedding": label, "accuracy": float(clf.score(X_test, y_test)), "n_features": len(cols)})
        summary_status["causal_sanity_envprobe"] = ("RAN", "computed")
    elif not args.compute_envprobe:
        summary_status["causal_sanity_envprobe"] = ("SKIPPED", "flag --compute_envprobe not set")
    else:
        reason = f"environment column '{env_col_effective}' unavailable"
        print(f"WARNING: Envprobe skipped ({reason}).")
        summary_status["causal_sanity_envprobe"] = ("SKIPPED", reason)
    pd.DataFrame(envprobe_rows).to_csv(outdir / "reports" / "causal_sanity_envprobe.csv", index=False)

    zinv_rows = []
    if args.compute_zinv_stability:
        zinv_cols = [c for c in merged_df.columns if c.startswith("z_inv")]
        if zinv_cols and env_col_effective in merged_df.columns:
            for env, g in merged_df.dropna(subset=zinv_cols + [env_col_effective]).groupby(env_col_effective):
                mean_vec = g[zinv_cols].mean().values
                zinv_rows.append({"env": env, "n": len(g), "mean_norm": float(np.linalg.norm(mean_vec)), "within_var": float(g[zinv_cols].var().mean())})
            summary_status["zinv_stability"] = ("RAN", "computed")
        else:
            reason = "missing z_inv features or environment column"
            print(f"WARNING: z_inv stability skipped ({reason}).")
            summary_status["zinv_stability"] = ("SKIPPED", reason)
        between = float(pd.DataFrame(zinv_rows)["mean_norm"].var()) if zinv_rows else np.nan
        for r in zinv_rows:
            r["between_env_var_mean_norm"] = between
    else:
        summary_status["zinv_stability"] = ("SKIPPED", "flag --compute_zinv_stability not set")
    pd.DataFrame(zinv_rows).to_csv(outdir / "reports" / "zinv_stability.csv", index=False)

    cf_rows = []
    checked_cf_paths: list[Path] = []
    if args.compute_cf_consistency:
        cf_dir: Path | None = Path(args.counterfactual_dir) if args.counterfactual_dir else None
        if cf_dir is None:
            outputs_root = Path(args.outdir).resolve().parent
            cf_dir, checked_cf_paths = detect_counterfactual_dir(outputs_root, args.target)
            if cf_dir is not None:
                logging.info("Counterfactual directory auto-detected: %s", cf_dir)
        else:
            logging.info("Counterfactual directory provided via CLI: %s", cf_dir)

        if cf_dir is not None:
            cf_file, cf_source, checked_files = resolve_counterfactual_source(cf_dir)
            checked_cf_paths.extend(checked_files)
            if cf_file is None:
                checked_msg = ", ".join(str(p) for p in checked_files)
                if args.counterfactual_dir:
                    raise SystemExit(
                        "counterfactual_dir was provided but no supported counterfactual files were found. "
                        f"Checked: {checked_msg}"
                    )
                reason = "counterfactual metrics unavailable"
                print(f"WARNING: Counterfactual consistency skipped ({reason}). Checked paths: {checked_msg}")
                cf_rows = [{"status": "SKIPPED", "reason": reason, "checked_paths": checked_msg}]
                summary_status["counterfactual_consistency"] = ("SKIPPED", reason)
            else:
                logging.info("Counterfactual consistency source: %s (%s)", cf_file, cf_source)
                cf_rows = build_cf_metrics_from_candidates(cf_file)
                if cf_rows:
                    summary_status["counterfactual_consistency"] = ("RAN", f"computed from {cf_source}")
                else:
                    reason = f"{cf_source} available but missing delta prediction columns"
                    print(f"WARNING: Counterfactual consistency skipped ({reason}). Source file: {cf_file}")
                    cf_rows = [{"status": "SKIPPED", "reason": reason, "source_file": str(cf_file)}]
                    summary_status["counterfactual_consistency"] = ("SKIPPED", reason)
        elif args.counterfactuals_root:
            for p in sorted(Path(args.counterfactuals_root).glob("*/evaluation/delta_predictions.csv")):
                df = pd.read_csv(p)
                if "rule_id" in df.columns and "delta_yhat" in df.columns:
                    for rid, g in df.groupby("rule_id"):
                        frac_pos = float((g["delta_yhat"] > 0).mean())
                        std = float(g["delta_yhat"].std()) if len(g) > 1 else 0.0
                        cf_rows.append({"run_id": p.parent.parent.name, "rule_id": rid, "fraction_positive": frac_pos, "delta_std": std, "invariance_score": frac_pos - 0.1 * std, "mean_delta": float(g["delta_yhat"].mean())})
            if cf_rows:
                summary_status["counterfactual_consistency"] = ("RAN", "computed from counterfactuals_root")
            else:
                reason = "counterfactual metrics unavailable"
                checked_msg = ", ".join(str(p) for p in checked_cf_paths) if checked_cf_paths else "(none)"
                print(f"WARNING: Counterfactual consistency skipped ({reason}). Checked paths: {checked_msg}")
                cf_rows = [{"status": "SKIPPED", "reason": reason, "checked_paths": checked_msg}]
                summary_status["counterfactual_consistency"] = ("SKIPPED", reason)
        else:
            reason = "counterfactual metrics unavailable"
            checked_msg = ", ".join(str(p) for p in checked_cf_paths) if checked_cf_paths else "(none)"
            print(f"WARNING: Counterfactual consistency skipped ({reason}). Checked paths: {checked_msg}")
            cf_rows = [{"status": "SKIPPED", "reason": reason, "checked_paths": checked_msg}]
            summary_status["counterfactual_consistency"] = ("SKIPPED", reason)
    elif not args.compute_cf_consistency:
        summary_status["counterfactual_consistency"] = ("SKIPPED", "flag --compute_cf_consistency not set")
    pd.DataFrame(cf_rows).to_csv(outdir / "reports" / "counterfactual_consistency.csv", index=False)

    pd.DataFrame(columns=["test", "effect_size", "p_value", "p_adjusted"]).to_csv(outdir / "reports" / "statistical_tests.csv", index=False)
    main_table = perf_by_split.head(20).copy()
    main_table.to_csv(outdir / "reports" / "paper_table_main.csv", index=False)

    style = style_from_args(args)
    configure_matplotlib(style, svg=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if not perf_by_split.empty and "rmse" in perf_by_split.columns:
        agg = perf_by_split.groupby("split")["rmse"].mean().sort_values()
        ax.bar(agg.index.astype(str), agg.values)
        ax.tick_params(axis="x", rotation=55)
    style_axis(ax, style, "Main Performance Across Splits", "Split", "RMSE")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_main_perf_across_splits.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not perf_by_env.empty and "rmse" in perf_by_env.columns:
        env_agg = perf_by_env.groupby("env")["rmse"].mean().sort_values()
        ax.bar(env_agg.index.astype(str), env_agg.values)
        ax.tick_params(axis="x", rotation=55)
    style_axis(ax, style, "Performance by Environment", "Environment", "RMSE")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_perf_by_env.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if args.task == "classification" and not merged_df.empty:
        _, cal = expected_calibration_error(merged_df["y_true"].astype(int), merged_df["y_pred"])
        if not cal.empty:
            ax.plot(cal["conf"], cal["acc"], marker="o")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        summary_status["fig_calibration_reliability"] = ("RAN", "classification reliability diagram")
    else:
        reason = "task is regression" if args.task == "regression" else "no merged predictions"
        print(f"WARNING: fig_calibration_reliability skipped ({reason}).")
        ax.text(0.5, 0.5, f"SKIPPED: {reason}", ha="center", va="center", transform=ax.transAxes)
        summary_status["fig_calibration_reliability"] = ("SKIPPED", reason)
    style_axis(ax, style, "Calibration Reliability", "Confidence", "Accuracy")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_calibration_reliability.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if not merged_df.empty:
        df = merged_df[["y_true", "y_pred"]].dropna().copy()
        df["conf"] = (1.0 / (1.0 + np.abs(df["y_pred"] - df["y_pred"].median())))
        rows = []
        for q in np.linspace(0.1, 1.0, 10):
            thr = df["conf"].quantile(1 - q)
            keep = df[df["conf"] >= thr]
            err = float(np.sqrt(np.mean((keep["y_pred"] - keep["y_true"]) ** 2))) if len(keep) else np.nan
            rows.append((q, err))
        rr = pd.DataFrame(rows, columns=["coverage", "error"])
        ax.plot(rr["coverage"], rr["error"], marker="o")
    style_axis(ax, style, "Selective Prediction", "Coverage", "Error")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_selective_prediction.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cns_df = pd.DataFrame(cns_metrics)
    if not cns_df.empty and "rmse" in cns_df.columns:
        ax.bar(cns_df["subset"], cns_df["rmse"])
        ax.tick_params(axis="x", rotation=45)
    style_axis(ax, style, "CNS-Stratified Performance", "Subset", "RMSE")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_cns_stratified_perf.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cf_df = pd.DataFrame(cf_rows)
    pareto_xlabel = "Fraction positive Δŷ"
    if cf_df is None or cf_df.empty or "status" in cf_df.columns:
        print("WARNING: fig_pareto_potency_vs_cns skipped (counterfactual metrics unavailable).")
    elif "delta_std" not in cf_df.columns:
        print("WARNING: fig_pareto_potency_vs_cns skipped (missing column: delta_std).")
    elif args.task == "regression":
        safe_x_candidates = ["mean_delta", "delta_mean", "delta_yhat_mean"]
        safe_x_col = next((c for c in safe_x_candidates if c in cf_df.columns), None)
        if safe_x_col is None:
            print(
                "WARNING: fig_pareto_potency_vs_cns skipped "
                "(regression task with no safe x-axis column)."
            )
        else:
            pareto_xlabel = safe_x_col
            ax.scatter(cf_df[safe_x_col], cf_df["delta_std"], alpha=0.7)
            logging.info("fig_pareto_potency_vs_cns: RAN (source metric column: %s)", safe_x_col)
    else:
        required = ["fraction_positive", "delta_std"]
        missing = [c for c in required if c not in cf_df.columns]
        if missing:
            print(
                "WARNING: fig_pareto_potency_vs_cns skipped "
                f"(missing columns: {', '.join(missing)})."
            )
        else:
            ax.scatter(cf_df["fraction_positive"], cf_df["delta_std"], alpha=0.7)
            logging.info("fig_pareto_potency_vs_cns: RAN (classification metrics)")
    style_axis(ax, style, "Pareto: Potency vs CNS", pareto_xlabel, "Δŷ std")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_pareto_potency_vs_cns.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ep = pd.DataFrame(envprobe_rows)
    if not ep.empty:
        ax.bar(ep["embedding"], ep["accuracy"])
    style_axis(ax, style, "Environment Probe Accuracy", "Embedding", "Accuracy")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_envprobe_accuracy.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    zs = pd.DataFrame(zinv_rows)
    if not zs.empty:
        ax.bar(zs["env"].astype(str), zs["within_var"])
        ax.tick_params(axis="x", rotation=45)
    style_axis(ax, style, "z_inv Stability", "Environment", "Within-env variance")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_zinv_stability.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if not cf_df.empty and {"rule_id", "invariance_score"}.issubset(cf_df.columns):
        top = cf_df.groupby("rule_id")["invariance_score"].mean().sort_values(ascending=False).head(10)
        ax.bar(top.index.astype(str), top.values)
        ax.tick_params(axis="x", rotation=55)
    elif not cf_df.empty:
        missing = [c for c in ["rule_id", "invariance_score"] if c not in cf_df.columns]
        print(
            "WARNING: fig_cf_consistency_across_envs skipped "
            f"(missing columns: {', '.join(missing)})."
        )
    style_axis(ax, style, "CF Consistency Across Environments", "Rule", "Invariance score")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_cf_consistency_across_envs.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not perf_by_split.empty:
        for col in ["rmse", "mae", "r2"]:
            if col in perf_by_split.columns:
                ax.plot(perf_by_split.groupby("split")[col].mean().index.astype(str), perf_by_split.groupby("split")[col].mean().values, marker="o", label=col)
        ax.tick_params(axis="x", rotation=55)
        ax.legend()
    style_axis(ax, style, "Ablation Summary", "Split", "Metric")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "fig_ablation_summary.svg"); plt.close(fig)

    run_cfg = vars(args)
    (outdir / "provenance" / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    prov = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        "inputs": {
            "dataset": args.dataset_parquet,
            "runs_root": args.runs_root,
            "splits_dir": args.splits_dir,
            "bbb": args.bbb_parquet,
            "counterfactuals_root": args.counterfactuals_root,
        },
    }
    (outdir / "provenance" / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (outdir / "provenance" / "environment.txt").write_text(subprocess.getoutput("python -m pip freeze") + "\n", encoding="utf-8")

    print("\n=== Step 8 analysis summary ===")
    for name in [
        "cns_subset_metrics",
        "causal_sanity_envprobe",
        "zinv_stability",
        "counterfactual_consistency",
        "fig_calibration_reliability",
    ]:
        status, reason = summary_status.get(name, ("SKIPPED", "not evaluated"))
        print(f" - {name}: {status} ({reason})")


if __name__ == "__main__":
    main()
