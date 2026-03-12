from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


POINTER_CANDIDATES = ("latest_screen.json", "run_pointer.json")


def _read_pointer(path: Path) -> Path | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    run_dir = data.get("run_dir")
    if not run_dir:
        return None
    return Path(run_dir).resolve()


def _is_screen_run_dir(path: Path) -> bool:
    return any(
        (path / rel).exists()
        for rel in (
            "predictions/scored_with_uncertainty.parquet",
            "predictions/predictions_ensemble.parquet",
            "predictions/scored_single_model.parquet",
            "ranking/ranked_all.parquet",
        )
    )


def resolve_step12_screen_outputs(step12_or_screen_dir: Path, explicit_screen_dir: str | None = None) -> dict[str, Any]:
    """Resolve Step12 screen outputs with new->old compatibility order."""
    if explicit_screen_dir:
        screen_dir = Path(explicit_screen_dir).resolve()
        if _is_screen_run_dir(screen_dir):
            return {"screen_dir": screen_dir, "source": "explicit", "workflow": "explicit"}
        if screen_dir.exists() and screen_dir.is_dir():
            return resolve_step12_screen_outputs(screen_dir, explicit_screen_dir=None)
        raise SystemExit(
            f"explicit screen_dir does not match accepted Step12 layouts: {screen_dir}; "
            "expected a screen run directory or a step12 root with latest_screen.json/run_pointer.json"
        )

    base = Path(step12_or_screen_dir).resolve()
    if _is_screen_run_dir(base):
        return {"screen_dir": base, "source": "direct", "workflow": "direct"}

    if not base.exists():
        raise SystemExit(
            f"missing Step12 directory: {base}; accepted layouts are new pointers "
            "(outputs/step12/latest_screen.json or outputs/step12/run_pointer.json) "
            "or old direct outputs/step12 with ranking/predictions artifacts"
        )

    for pointer in POINTER_CANDIDATES:
        pointer_path = base / pointer
        target = _read_pointer(pointer_path)
        if target is not None and _is_screen_run_dir(target):
            wf = "new" if pointer == "latest_screen.json" else "compat"
            return {"screen_dir": target, "source": str(pointer_path), "workflow": wf}

    # Old workflow fallback: single unambiguous screen dir under step12.
    candidates = [p for p in sorted(base.glob("screening/*/*"), reverse=True) if _is_screen_run_dir(p)]
    if not candidates:
        candidates = [p for p in sorted(base.glob("*/*"), reverse=True) if _is_screen_run_dir(p)]
    if len(candidates) == 1:
        return {"screen_dir": candidates[0], "source": "old-layout", "workflow": "old"}
    if len(candidates) > 1:
        raise SystemExit(
            "ambiguous Step12 outputs without pointer files; found multiple screen directories "
            f"under {base}. Provide --screen_dir or write outputs/step12/latest_screen.json"
        )

    if _is_screen_run_dir(base / "screening"):
        return {"screen_dir": base / "screening", "source": "old-layout", "workflow": "old"}

    raise SystemExit(
        "no Step12 screening outputs found. Accepted layouts:\n"
        "- New: outputs/step12/latest_screen.json or outputs/step12/run_pointer.json pointing to a screen run dir\n"
        "- Old: outputs/step12 directory (or child) containing ranking/ranked_all.parquet and prediction tables"
    )


def normalize_screening_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    out = df.copy()
    score_col = None
    for c in ("pred_mean", "score_mean", "prediction_mean", "pIC50_hat", "prediction"):
        if c in out.columns:
            score_col = c
            break
    if score_col is None:
        raise SystemExit(
            "screening table is missing score columns. Expected one of: "
            "pred_mean, score_mean, prediction_mean, pIC50_hat, prediction"
        )
    if "canonical_score" not in out.columns:
        out["canonical_score"] = out[score_col]
    if "score_mean" not in out.columns:
        out["score_mean"] = out["canonical_score"]

    rank_col = None
    for c in ("rank_by_mean", "rank", "rank_score", "rank_by_score"):
        if c in out.columns:
            rank_col = c
            break
    if rank_col is None:
        out["rank_by_mean"] = out["canonical_score"].rank(method="first", ascending=False).astype(int)
        rank_col = "rank_by_mean"

    # Preserve uncertainty/model-count if present and expose canonical names.
    if "pred_std" in out.columns and "uncertainty_std" not in out.columns:
        out["uncertainty_std"] = out["pred_std"]
    if "prediction_std" in out.columns and "uncertainty_std" not in out.columns:
        out["uncertainty_std"] = out["prediction_std"]

    return out, score_col, rank_col


def load_screening_tables(screen_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_candidates = [
        screen_dir / "predictions/scored_with_uncertainty.parquet",
        screen_dir / "predictions/scored_ensemble.parquet",
        screen_dir / "predictions/predictions_ensemble.parquet",
        screen_dir / "predictions/scored_single_model.parquet",
        screen_dir / "predictions/predictions_seed1.parquet",
        screen_dir / "ranking/ranked_all.parquet",
    ]
    score_path = next((p for p in score_candidates if p.exists()), None)
    if score_path is None:
        raise SystemExit(
            f"missing Step12 prediction table in {screen_dir}; expected one of: "
            "predictions/scored_with_uncertainty.parquet, predictions/predictions_ensemble.parquet, "
            "or predictions/scored_single_model.parquet"
        )
    scored = pd.read_parquet(score_path)

    rank_all_path = screen_dir / "ranking/ranked_all.parquet"
    if rank_all_path.exists():
        ranked_all = pd.read_parquet(rank_all_path)
    else:
        ranked_all = scored.copy()

    rank_cns_candidates = [
        screen_dir / "ranking/ranked_cns_like_in_domain.parquet",
        screen_dir / "ranking/ranked_cns_like.parquet",
    ]
    rank_cns_path = next((p for p in rank_cns_candidates if p.exists()), None)
    ranked_cns = pd.read_parquet(rank_cns_path) if rank_cns_path else ranked_all.copy()

    scored, _, _ = normalize_screening_columns(scored)
    ranked_all, _, _ = normalize_screening_columns(ranked_all)
    ranked_cns, _, _ = normalize_screening_columns(ranked_cns)
    return scored, ranked_all, ranked_cns
