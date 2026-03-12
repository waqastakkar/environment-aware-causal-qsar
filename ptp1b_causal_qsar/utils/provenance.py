from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return proc.stdout.strip()
    except Exception:
        return None


def write_environment_txt(path: str | Path) -> None:
    proc = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True)
    Path(path).write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")


def collect_provenance(
    *,
    config_sha: str,
    config: dict[str, Any],
    executed_commands: list[dict[str, Any]],
) -> dict[str, Any]:
    tracked_files = {
        "chembl_sqlite": config.get("paths", {}).get("chembl_sqlite"),
        "dataset_parquet": config.get("paths", {}).get("dataset_parquet"),
        "checkpoint": config.get("paths", {}).get("checkpoint"),
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "config_sha256": config_sha,
        "referenced_file_sha256": {k: sha256_file(v) for k, v in tracked_files.items() if v},
        "executed_commands": executed_commands,
    }
