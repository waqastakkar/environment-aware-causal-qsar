from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from ptp1b_causal_qsar.config import (
    deep_update,
    load_yaml_config,
    parse_overrides,
    resolve_paths,
    validate_minimum_schema,
)
from ptp1b_causal_qsar.runner import create_pipeline_run_dir, execute_steps
from ptp1b_causal_qsar.steps_registry import STEPS_REGISTRY, get_nested, parse_step_range
from ptp1b_causal_qsar.utils.logging import configure_pipeline_logger


def _load_resolved_config(config_path: str, override_args: list[str]) -> dict[str, Any]:
    cfg = load_yaml_config(config_path)
    overrides = parse_overrides(override_args)
    cfg = deep_update(cfg, overrides)
    return resolve_paths(cfg, config_path)


def run_checks(config: dict[str, Any], steps: list[int]) -> list[str]:
    issues = validate_minimum_schema(config)

    outputs_root = Path(config.get("paths", {}).get("outputs_root", "outputs"))
    try:
        outputs_root.mkdir(parents=True, exist_ok=True)
        test_file = outputs_root / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
    except Exception as exc:
        issues.append(f"Outputs directory not writable: {exc}")

    try:
        importlib.import_module("rdkit")
    except Exception as exc:
        issues.append(f"RDKit import failed: {exc}")

    for step in steps:
        if step == 0:
            continue
        meta = STEPS_REGISTRY[step]
        script = meta.get("script")
        if script:
            script_path = Path(script)
            if not script_path.exists():
                issues.append(f"Missing script for step {step}: {script_path}")
        for req in meta.get("required_inputs", []):
            val = get_nested(config, req)
            if val is None:
                issues.append(f"Missing required config key for step {step}: {req}")
            elif req.startswith("paths."):
                p = Path(str(val))
                if req.endswith("outputs_root"):
                    continue
                if not p.exists():
                    issues.append(f"WARNING: Required path does not exist for step {step}: {req}={p}")
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptp1bqsar", description="Unified PTP1B QSAR pipeline CLI")
    parser.add_argument("--override", action="append", default=[], help="Override config values with dotted KEY=VALUE")

    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Validate configuration and environment")
    p_check.add_argument("--config", required=True)
    p_check.add_argument("--steps", default="0-15")

    p_step = sub.add_parser("step", help="Run a single step")
    p_step.add_argument("step", type=int)
    p_step.add_argument("--config", required=True)
    p_step.add_argument("--dry_run", action="store_true")
    p_step.add_argument("--continue_on_error", action="store_true")

    p_run = sub.add_parser("run", help="Run multiple steps")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--steps", required=True)
    p_run.add_argument("--dry_run", action="store_true")
    p_run.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue attempting later steps after a failure (default: True)",
    )

    p_man = sub.add_parser("manuscript", help="Run manuscript build (step 15)")
    p_man.add_argument("--config", required=True)
    p_man.add_argument("--paper_id")
    p_man.add_argument("--dry_run", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    unknown_overrides = []
    key = None
    for token in unknown:
        if token.startswith("--"):
            key = token[2:]
        elif key:
            unknown_overrides.append(f"{key}={token}")
            key = None

    all_overrides = list(args.override) + unknown_overrides
    config = _load_resolved_config(args.config, all_overrides)

    if args.command == "check":
        steps = parse_step_range(args.steps, set(STEPS_REGISTRY))
        issues = run_checks(config, steps)
        if issues:
            hard_errors = [i for i in issues if not i.startswith("WARNING:")]
            print("Validation report:")
            for issue in issues:
                print(f" - {issue}")
            if hard_errors:
                return 1
        print("Validation passed.")
        return 0

    if args.command == "step":
        steps = parse_step_range(str(args.step), set(STEPS_REGISTRY))
    elif args.command == "run":
        steps = parse_step_range(args.steps, set(STEPS_REGISTRY))
    else:
        if args.paper_id:
            config["paper_id"] = args.paper_id
        steps = [15]

    run_dir = create_pipeline_run_dir(config["paths"]["outputs_root"])
    logger = configure_pipeline_logger(run_dir / "pipeline_log.txt")

    result = execute_steps(
        config=config,
        steps=steps,
        run_dir=run_dir,
        logger=logger,
        continue_on_error=getattr(args, "continue_on_error", False),
        dry_run=getattr(args, "dry_run", False),
        overrides=parse_overrides(unknown_overrides),
    )
    print(f"Pipeline run artifacts: {result.run_dir}")
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
