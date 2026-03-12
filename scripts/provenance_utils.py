from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def git_commit_or_unknown(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def safe_load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}
    try:
        if p.suffix.lower() == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        if p.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # type: ignore

            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"value": data}
    except Exception:
        return {}
    return {}


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def collect_tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ["pandas", "openpyxl", "yaml"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not-installed"
    return versions


def write_environment_txt(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = subprocess.check_output(["python", "-m", "pip", "freeze"], stderr=subprocess.STDOUT)
        p.write_text(out.decode("utf-8"), encoding="utf-8")
    except Exception as exc:
        p.write_text(f"pip freeze unavailable: {exc}\n", encoding="utf-8")


def try_read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def basename_run_id(run_dir: str | Path) -> str:
    return Path(run_dir).name


def file_size(path: str | Path) -> int:
    p = Path(path)
    return p.stat().st_size if p.exists() and p.is_file() else 0


def find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def recursive_collect_by_suffix(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    found: list[Path] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(suffixes):
                found.append(Path(dirpath) / fn)
    return found
