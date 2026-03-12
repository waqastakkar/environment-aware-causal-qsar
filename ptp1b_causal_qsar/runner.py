from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

import yaml

from ptp1b_causal_qsar.config import config_sha256
from ptp1b_causal_qsar.steps_registry import STEPS_REGISTRY
from ptp1b_causal_qsar.utils.logging import dump_json
from ptp1b_causal_qsar.utils.provenance import collect_provenance, write_environment_txt


@dataclass
class RunnerResult:
    run_dir: Path
    steps: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def create_pipeline_run_dir(outputs_root: str | Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}"
    run_dir = Path(outputs_root) / "pipeline_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_config_token(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _resolve_default_output_path(config: dict[str, Any], template: str | None) -> Path | None:
    if not template:
        return None

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = _resolve_config_token(config, key)
        return str(value) if value is not None else match.group(0)

    resolved = re.sub(r"\{([^{}]+)\}", _replace, template)
    return Path(resolved)


def _has_global_step_completion(config: dict[str, Any], step_number: int, logger: Any) -> bool:
    meta = STEPS_REGISTRY[step_number]
    output_path = _resolve_default_output_path(config, meta.get("default_output_path"))
    if output_path is None:
        logger.info("Prerequisite step %s has no configured output path to verify.", step_number)
        return False

    if output_path.exists() and (output_path.is_file() or any(output_path.iterdir())):
        logger.info(
            "Prerequisite step %s accepted via global completion check at: %s",
            step_number,
            output_path,
        )
        return True

    logger.info(
        "Prerequisite step %s not satisfied by global completion check at: %s",
        step_number,
        output_path,
    )
    return False


def execute_steps(
    *,
    config: dict[str, Any],
    steps: list[int],
    run_dir: Path,
    logger: Any,
    continue_on_error: bool = False,
    dry_run: bool = False,
    overrides: dict[str, Any] | None = None,
) -> RunnerResult:
    overrides = overrides or {}
    step_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    resolved_config_path = run_dir / "pipeline_config_resolved.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed_steps: set[int] = set()

    for step in steps:
        if step == 0:
            logger.info("Step 0 selected (CLI), no script execution required.")
            continue

        meta = STEPS_REGISTRY[step]
        missing_dependencies = [
            dep
            for dep in meta.get("depends_on", [])
            if dep not in completed_steps and not _has_global_step_completion(config, dep, logger)
        ]
        if missing_dependencies:
            checked_steps_file = run_dir / "pipeline_steps_executed.json"
            expected_outputs = {
                dep: str(_resolve_default_output_path(config, STEPS_REGISTRY[dep].get("default_output_path")))
                for dep in missing_dependencies
            }
            blocked_record = {
                "step_number": step,
                "name": meta["name"],
                "cmd": None,
                "start_time": None,
                "end_time": None,
                "return_code": None,
                "status": "blocked",
                "blocked_by": missing_dependencies,
                "pipeline_run_id": run_dir.name,
                "checked_steps_file": str(checked_steps_file),
                "expected_prerequisite_outputs": expected_outputs,
                "output_paths": [meta.get("default_output_path")],
            }
            step_records.append(blocked_record)
            err = {
                "step_number": step,
                "name": meta["name"],
                "command": None,
                "traceback": (
                    f"Step {step} blocked: missing completed prerequisite step(s) "
                    f"{missing_dependencies}. pipeline_run_id={run_dir.name}; "
                    f"checked step completion file={checked_steps_file}; "
                    f"expected prerequisite output path(s)={expected_outputs}. "
                    "Ensure prior steps finish successfully before continuing."
                ),
            }
            errors.append(err)
            logger.error(err["traceback"])
            if not continue_on_error:
                break
            continue

        command = meta["build_command"](config, overrides)
        start = datetime.now(timezone.utc)
        logger.info("Running step %s (%s): %s", step, meta["name"], " ".join(command))

        result_code = 0
        if not dry_run:
            proc = subprocess.run(command, capture_output=True, text=True)
            logger.info(proc.stdout)
            if proc.stderr:
                logger.error(proc.stderr)
            result_code = proc.returncode

        end = datetime.now(timezone.utc)
        record = {
            "step_number": step,
            "name": meta["name"],
            "cmd": command,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "return_code": result_code,
            "status": "ok" if result_code == 0 else "failed",
            "output_paths": [meta.get("default_output_path")],
        }
        step_records.append(record)

        if result_code != 0:
            err = {
                "step_number": step,
                "name": meta["name"],
                "command": command,
                "traceback": f"Command exited with return code {result_code}: {' '.join(command)}",
            }
            errors.append(err)
            if not continue_on_error:
                break
        else:
            completed_steps.add(step)

    dump_json(run_dir / "pipeline_steps_executed.json", step_records)
    if errors:
        dump_json(run_dir / "pipeline_errors.json", errors)

    prov = collect_provenance(
        config_sha=config_sha256(config),
        config=config,
        executed_commands=step_records,
    )
    dump_json(run_dir / "provenance.json", prov)
    write_environment_txt(run_dir / "environment.txt")

    return RunnerResult(run_dir=run_dir, steps=step_records, errors=errors)
