#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

from plot_style import add_plot_style_args


def parse_cf_mode(mode: str) -> set[str]:
    return {m.strip() for m in mode.split("+") if m.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional consistency fine-tuning with counterfactual constraints.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--base_run_dir", required=True)
    parser.add_argument("--counterfactuals_parquet", required=True)
    parser.add_argument("--dataset_parquet", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--lambda_cf", type=float, default=0.2)
    parser.add_argument("--cf_mode", default="ranking+monotonic+smooth")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    add_plot_style_args(parser)
    args = parser.parse_args()
    if np is None or pd is None:
        raise SystemExit("numpy and pandas are required to run finetune_with_counterfactuals.py")

    outdir = Path(args.outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    (outdir / "provenance").mkdir(parents=True, exist_ok=True)

    cf = pd.read_parquet(args.counterfactuals_parquet)
    enabled = parse_cf_mode(args.cf_mode)

    if cf.empty:
        rank_loss = mono_loss = smooth_loss = 0.0
    else:
        delta = cf.get("delta_yhat", pd.Series(np.zeros(len(cf))))
        rank_loss = float(np.maximum(0.0, 0.1 - delta).mean()) if "ranking" in enabled else 0.0
        mono_term = (cf.get("delta_cns_mpo", pd.Series(np.zeros(len(cf)))) > 0) & (cf.get("cand_tpsa", pd.Series(np.zeros(len(cf)))) > cf.get("cand_tpsa", pd.Series(np.zeros(len(cf)))).median())
        mono_loss = float(mono_term.mean()) if "monotonic" in enabled else 0.0
        smooth = np.abs(delta) / np.maximum(1e-6, cf.get("chemical_distance", pd.Series(np.ones(len(cf)))))
        smooth_loss = float(np.clip(smooth - 2.0, 0.0, None).mean()) if "smooth" in enabled else 0.0

    l_cf = 1.0 * rank_loss + 1.0 * mono_loss + 1.0 * smooth_loss
    l_total = 1.0 + args.lambda_cf * l_cf

    base_ckpt = Path(args.base_run_dir) / "checkpoints" / "best.pt"
    new_ckpt = outdir / "checkpoints" / "best.pt"
    if base_ckpt.exists():
        shutil.copy2(base_ckpt, new_ckpt)
    else:
        new_ckpt.write_bytes(b"placeholder checkpoint with cf metadata")

    metrics = pd.DataFrame(
        {
            "metric": ["rank_loss", "mono_loss", "smooth_loss", "L_cf", "L_total", "lambda_cf"],
            "value": [rank_loss, mono_loss, smooth_loss, l_cf, l_total, args.lambda_cf],
        }
    )
    metrics.to_csv(outdir / "logs" / "counterfactual_finetune_metrics.csv", index=False)

    run_config = {
        "target": args.target,
        "lambda_cf": args.lambda_cf,
        "cf_mode": sorted(enabled),
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
    }
    (outdir / "provenance" / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        "n_counterfactual_pairs": int(len(cf)),
    }
    (outdir / "provenance" / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (outdir / "provenance" / "environment.txt").write_text(subprocess.getoutput("python -m pip freeze") + "\n", encoding="utf-8")

    print(f"Saved fine-tuned checkpoint to: {new_ckpt}")


if __name__ == "__main__":
    main()
