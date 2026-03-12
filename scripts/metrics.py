from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] < 2:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
    }


def classification_metrics(y_true, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    metrics = {}
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    records = []
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        frac = mask.mean()
        ece += abs(acc - conf) * frac
        records.append({"bin": i, "count": int(mask.sum()), "confidence": conf, "accuracy": acc, "gap": abs(acc - conf)})
    return float(ece), pd.DataFrame(records)


def regression_calibration(y_true, y_pred, n_bins: int = 10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    qs = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    idx = np.digitize(y_pred, qs[1:-1], right=True)
    records = []
    for i in range(n_bins):
        m = idx == i
        if m.sum() == 0:
            continue
        records.append(
            {
                "bin": i,
                "count": int(m.sum()),
                "pred_mean": float(y_pred[m].mean()),
                "obs_mean": float(y_true[m].mean()),
                "abs_gap": float(abs(y_pred[m].mean() - y_true[m].mean())),
            }
        )
    return pd.DataFrame(records)


def per_environment_metrics(df: pd.DataFrame, task: str, env_col: str, label_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for env, group in df.groupby(env_col):
        if task == "regression":
            m = regression_metrics(group[label_col], group[pred_col])
        else:
            m = classification_metrics(group[label_col], group[pred_col])
        m[env_col] = env
        m["n"] = len(group)
        rows.append(m)
    return pd.DataFrame(rows)


def linear_probe_env_predictability(z_inv: np.ndarray, env: np.ndarray) -> float:
    if len(np.unique(env)) < 2:
        return float("nan")
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=0),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        probe.fit(z_inv, env)
    return float(probe.score(z_inv, env))
