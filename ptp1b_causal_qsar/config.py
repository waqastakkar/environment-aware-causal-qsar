from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL = ["paper_id", "target", "paths", "style", "training", "robustness", "screening"]
REQUIRED_PATH_KEYS = ["chembl_sqlite", "data_root", "outputs_root"]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    if "," in raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def parse_overrides(override_items: list[str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        dotted_key, value = item.split("=", 1)
        cursor = updates
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce_value(value)
    return updates


def resolve_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["_config_path"] = str(Path(config_path).resolve())
    config_parent = Path.cwd()
    path_cfg = cfg.get("paths", {})
    for key in REQUIRED_PATH_KEYS:
        if key in path_cfg:
            val = Path(path_cfg[key])
            if not val.is_absolute():
                val = (config_parent / val).resolve()
            path_cfg[key] = str(val)
    outputs_root = Path(path_cfg.get("outputs_root", (config_parent / "outputs").resolve()))
    outputs_root.mkdir(parents=True, exist_ok=True)
    cfg["paths"] = path_cfg
    cfg["paths"]["outputs_root"] = str(outputs_root)
    return cfg


def validate_minimum_schema(config: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in config:
            issues.append(f"Missing top-level key: {key}")
    for key in REQUIRED_PATH_KEYS:
        if key not in config.get("paths", {}):
            issues.append(f"Missing paths.{key}")
    return issues


def config_sha256(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
