from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

StepBuilder = Callable[[dict[str, Any], dict[str, Any]], list[str]]

STYLE_KEYS = [
    "svg_only",
    "font",
    "bold_text",
    "palette",
    "font_title",
    "font_label",
    "font_tick",
    "font_legend",
]

STYLE_ARG_ALIASES = {
    # Most scripts expose `--svg`; config keeps `svg_only` for manuscript pack compatibility.
    "svg_only": "svg",
}


def _python_bin(config: dict[str, Any]) -> str:
    """Resolve the interpreter used to execute pipeline steps.

    Priority:
    1) `config.runtime.python` (explicit pipeline config override)
    2) `PIPELINE_PYTHON` environment variable
    3) current interpreter (`sys.executable`)
    """

    runtime_python = config.get("runtime", {}).get("python") if isinstance(config.get("runtime"), dict) else None
    if runtime_python:
        return str(runtime_python)

    return os.environ.get("PIPELINE_PYTHON") or sys.executable


def _style_flags(config: dict[str, Any]) -> list[str]:
    style = config.get("style", {})
    flags: list[str] = []
    for key in STYLE_KEYS:
        if key not in style or style[key] is None:
            continue

        cli_key = STYLE_ARG_ALIASES.get(key, key)
        value = style[key]
        if isinstance(value, bool):
            if value:
                flags.append(f"--{cli_key}")
            continue

        flags.extend([f"--{cli_key}", str(value)])
    return flags


def _step5_run_benchmark_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    training = config.get("training", {})

    cmd = [_python_bin(config), str(Path("scripts") / "run_benchmark.py")]
    cmd.extend(["--target", str(config["target"])])
    cmd.extend(["--dataset_parquet", str(out_root / "step3" / "multienv_compound_level.parquet")])
    cmd.extend(["--splits_dir", str(out_root / "step4")])
    cmd.extend(["--split_names", str(training.get("split_default", "scaffold_bm"))])
    cmd.extend(["--outdir", str(out_root / "step5")])
    cmd.extend(["--task", str(training["task"])])
    cmd.extend(["--label_col", str(training["label_col"])])
    cmd.extend(["--env_col", str(training.get("env_col", "env_id"))])

    seeds = training.get("seeds")
    if isinstance(seeds, list) and seeds:
        cmd.extend(["--seeds", ",".join(str(seed) for seed in seeds)])

    cmd.extend(_style_flags(config))

    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _resolve_first_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    run_candidates = sorted(runs_root.glob("**/checkpoints/best.pt"))
    if not run_candidates:
        return None
    return run_candidates[0].parent.parent


def _resolve_step3_bbb_parquet(out_root: Path) -> Path | None:
    """Return step3 BBB annotations path, supporting both legacy and current layouts."""

    candidates = [
        out_root / "step3" / "data" / "bbb_annotations.parquet",
        out_root / "step3" / "bbb_annotations.parquet",
    ]
    return next((path for path in candidates if path.exists()), None)


def _default_builder(script_name: str, include_style: bool = False) -> StepBuilder:
    def _build(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
        cmd = [_python_bin(config), str(Path("scripts") / script_name)]
        cmd.extend(["--config", config["_config_path"]])
        if include_style:
            cmd.extend(_style_flags(config))
        for key, value in overrides.items():
            cmd.extend([f"--{key}", str(value)])
        return cmd

    return _build


def _step1_extract_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    cmd = [_python_bin(config), str(Path("scripts") / "extract_chembl36_sqlite.py")]
    cmd.extend(["--config", config["_config_path"]])
    cmd.extend(["--db", str(config["paths"]["chembl_sqlite"])])
    cmd.extend(["--target", str(config["target"])])
    cmd.extend(["--outdir", str(Path(config["paths"]["outputs_root"]) / "step1")])
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _step2_postprocess_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    step1_out = out_root / "step1" / f"{config['target']}_qsar_ready.csv"
    cmd = [_python_bin(config), str(Path("scripts") / "qsar_postprocess.py")]
    cmd.extend(["--config", config["_config_path"]])
    cmd.extend(["--input", str(step1_out)])
    cmd.extend(["--outdir", str(out_root / "step2")])
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _step3_assemble_environments_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    step2_out = out_root / "step2"
    row_level_csv = step2_out / "row_level_with_pIC50.csv"
    compound_level_csv = step2_out / "compound_level_with_properties.csv"
    raw_extract_csv = out_root / "step1" / f"{config['target']}_qsar_ready.csv"

    env_cfg = config.get("environments", {})
    bbb_rules = env_cfg.get("bbb_rules", str(Path("configs") / "bbb_rules.yaml"))
    series_rules = env_cfg.get("series_rules")
    env_keys = env_cfg.get("env_keys")

    cmd = [_python_bin(config), str(Path("scripts") / "assemble_environments.py")]
    cmd.extend(["--target", str(config["target"])])
    cmd.extend(["--row_level_csv", str(row_level_csv)])
    cmd.extend(["--compound_level_csv", str(compound_level_csv)])
    cmd.extend(["--raw_extract_csv", str(raw_extract_csv)])
    cmd.extend(["--outdir", str(out_root / "step3")])
    cmd.extend(["--bbb_rules", str(bbb_rules)])
    if series_rules:
        cmd.extend(["--series_rules", str(series_rules)])
    if env_keys:
        cmd.extend(["--env_keys", *[str(k) for k in env_keys]])
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _step7_generate_counterfactuals_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    step3_dataset = out_root / "step3" / "multienv_compound_level.parquet"
    step6_runs_root = out_root / "step6" / str(config["target"])
    step5_runs_root = out_root / "step5" / str(config["target"])
    step7_out = out_root / "step7"
    rules_parquet = step7_out / "rules" / "mmp_rules.parquet"
    run_dir = _resolve_first_run_dir(step6_runs_root) or _resolve_first_run_dir(step5_runs_root) or step6_runs_root

    py = _python_bin(config)
    build_rules_cmd = [
        py,
        str(Path("scripts") / "build_mmp_rules.py"),
        "--target",
        str(config["target"]),
        "--input_parquet",
        str(step3_dataset),
        "--outdir",
        str(step7_out),
    ]
    build_rules_cmd.extend(_style_flags(config))

    generate_cmd = [
        py,
        str(Path("scripts") / "generate_counterfactuals.py"),
        "--target",
        str(config["target"]),
        "--run_dir",
        str(run_dir),
        "--dataset_parquet",
        str(step3_dataset),
        "--mmp_rules_parquet",
        str(rules_parquet),
        "--outdir",
        str(step7_out),
    ]

    bbb_parquet = _resolve_step3_bbb_parquet(out_root)
    if bbb_parquet is not None:
        generate_cmd.extend(["--bbb_parquet", str(bbb_parquet)])

    screening_cfg = config.get("screening", {})
    if screening_cfg.get("cns_mpo_threshold") is not None:
        generate_cmd.extend(["--cns_mpo_threshold", str(screening_cfg["cns_mpo_threshold"])])

    generate_cmd.extend(_style_flags(config))
    for key, value in overrides.items():
        generate_cmd.extend([f"--{key}", str(value)])

    candidates_dir = step7_out / "candidates"
    expected_outputs = [
        candidates_dir / "generated_counterfactuals.parquet",
        candidates_dir / "filtered_counterfactuals.parquet",
        candidates_dir / "ranked_topk.parquet",
    ]

    shell_cmd = (
        "set -euo pipefail; "
        f"if [ ! -f {shlex.quote(str(rules_parquet))} ]; then "
        f"{' '.join(shlex.quote(x) for x in build_rules_cmd)}; "
        "fi; "
        "echo '[step7] Starting generate_counterfactuals.py'; "
        f"{' '.join(shlex.quote(x) for x in generate_cmd)}; "
        "echo '[step7] Finished generate_counterfactuals.py'; "
        + " ".join(
            f"if [ ! -f {shlex.quote(str(path))} ]; then "
            f"echo '[step7] ERROR: missing expected output file: {shlex.quote(str(path))}' >&2; "
            "exit 1; "
            "fi;"
            for path in expected_outputs
        )
    )
    return ["bash", "-c", shell_cmd]


def _step8_evaluate_model_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    step3_dataset = out_root / "step3" / "multienv_compound_level.parquet"
    step4_splits = out_root / "step4"
    step6_runs_root = out_root / "step6" / str(config["target"])
    step5_runs_root = out_root / "step5" / str(config["target"])
    step8_out = out_root / "step8"
    bbb_parquet = _resolve_step3_bbb_parquet(out_root)
    training = config.get("training", {})
    runs_root = step6_runs_root if step6_runs_root.exists() else step5_runs_root

    cmd = [
        _python_bin(config),
        str(Path("scripts") / "evaluate_runs.py"),
        "--target",
        str(config["target"]),
        "--runs_root",
        str(runs_root),
        "--splits_dir",
        str(step4_splits),
        "--dataset_parquet",
        str(step3_dataset),
        "--outdir",
        str(step8_out),
        "--task",
        str(training.get("task", "regression")),
        "--label_col",
        str(training.get("label_col", "pIC50")),
        "--env_col",
        str(training.get("env_col", "env_id_manual")),
    ]
    if bbb_parquet is not None:
        cmd.extend(["--bbb_parquet", str(bbb_parquet)])

    cmd.extend(_style_flags(config))
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _step6_train_exact_objective_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    training = config.get("training", {})

    cmd = [
        _python_bin(config),
        str(Path("scripts") / "train_causal_qsar.py"),
        "--target",
        str(config["target"]),
        "--dataset_parquet",
        str(out_root / "step3" / "multienv_compound_level.parquet"),
        "--splits_dir",
        str(out_root / "step4"),
        "--split_name",
        str(training.get("split_default", "scaffold_bm")),
        "--outdir",
        str(out_root / "step6"),
        "--task",
        str(training.get("task", "regression")),
        "--label_col",
        str(training.get("label_col", "pIC50")),
        "--env_col",
        str(training.get("env_col", "env_id_manual")),
        "--epochs",
        str(training.get("epochs", 300)),
        "--early_stopping_patience",
        str(training.get("early_stopping_patience", 30)),
    ]

    seeds = training.get("seeds")
    if isinstance(seeds, list) and seeds:
        cmd.extend(["--seed", str(seeds[0])])

    bbb_parquet = _resolve_step3_bbb_parquet(out_root)
    if bbb_parquet is not None:
        cmd.extend(["--bbb_parquet", str(bbb_parquet)])

    cmd.extend(_style_flags(config))
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _step9_evaluate_cross_endpoint_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    step6_runs_root = out_root / "step6" / str(config["target"])
    step5_runs_root = out_root / "step5" / str(config["target"])
    step9_out = out_root / "step9"
    bbb_parquet = _resolve_step3_bbb_parquet(out_root)

    external_candidates = [
        Path("data/external/processed") / f"{str(config['target']).lower()}_inhibition" / "data" / "inhibition_external_final.parquet",
        Path("data/external/processed") / "ptp1b_inhibition_chembl335" / "data" / "inhibition_external_final.parquet",
    ]
    external_parquet = next((p for p in external_candidates if p.exists()), None)

    run_dir = _resolve_first_run_dir(step6_runs_root) or _resolve_first_run_dir(step5_runs_root)

    if run_dir is None or external_parquet is None:
        reason = []
        if run_dir is None:
            reason.append("missing trained run checkpoint under outputs/step5")
        if external_parquet is None:
            reason.append("missing external inhibition parquet")
        message = ", ".join(reason)
        script = (
            "from pathlib import Path; "
            f"d=Path({step9_out.as_posix()!r}); "
            "d.mkdir(parents=True, exist_ok=True); "
            "(d/'step9_noop.txt').write_text(" + repr(f"skipped step 9: {message}\n") + ", encoding='utf-8'); "
            "print('Step 9 skipped:', " + repr(message) + ")"
        )
        return [_python_bin(config), "-c", script]

    cmd = [
        _python_bin(config),
        str(Path("scripts") / "evaluate_cross_endpoint.py"),
        "--target",
        str(config["target"]),
        "--run_dir",
        str(run_dir),
        "--external_parquet",
        str(external_parquet),
        "--outdir",
        str(step9_out),
    ]
    if bbb_parquet is not None:
        cmd.extend(["--bbb_parquet", str(bbb_parquet)])

    cmd.extend(_style_flags(config))
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _step10_interpret_model_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    step3_dataset = out_root / "step3" / "multienv_compound_level.parquet"
    step6_runs_root = out_root / "step6" / str(config["target"])
    step5_runs_root = out_root / "step5" / str(config["target"])
    step7_counterfactuals = out_root / "step7" / "candidates" / "ranked_topk.parquet"
    step10_out = out_root / "step10"
    bbb_parquet = _resolve_step3_bbb_parquet(out_root)

    run_dir = _resolve_first_run_dir(step6_runs_root) or _resolve_first_run_dir(step5_runs_root)

    if run_dir is None or not step3_dataset.exists():
        reason = []
        if run_dir is None:
            reason.append("missing trained run checkpoint under outputs/step5")
        if not step3_dataset.exists():
            reason.append("missing multienv dataset parquet at outputs/step3")
        message = ", ".join(reason)
        script = (
            "from pathlib import Path; "
            f"d=Path({step10_out.as_posix()!r}); "
            "d.mkdir(parents=True, exist_ok=True); "
            "(d/'step10_noop.txt').write_text(" + repr(f"skipped step 10: {message}\n") + ", encoding='utf-8'); "
            "print('Step 10 skipped:', " + repr(message) + ")"
        )
        return [_python_bin(config), "-c", script]

    cmd = [
        _python_bin(config),
        str(Path("scripts") / "interpret_model.py"),
        "--target",
        str(config["target"]),
        "--run_dir",
        str(run_dir),
        "--dataset_parquet",
        str(step3_dataset),
        "--outdir",
        str(step10_out),
    ]
    if bbb_parquet is not None:
        cmd.extend(["--bbb_parquet", str(bbb_parquet)])
    if step7_counterfactuals.exists():
        cmd.extend(["--counterfactuals_parquet", str(step7_counterfactuals)])

    cmd.extend(_style_flags(config))
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


STEPS_REGISTRY: dict[int, dict[str, Any]] = {
    1: {
        "name": "extract_chembl36_sqlite",
        "script": "scripts/extract_chembl36_sqlite.py",
        "required_inputs": ["paths.chembl_sqlite"],
        "default_output_path": "{paths.outputs_root}/step1",
        "build_command": _step1_extract_builder,
        "depends_on": [],
    },
    2: {
        "name": "qsar_postprocess",
        "script": "scripts/qsar_postprocess.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step2",
        "build_command": _step2_postprocess_builder,
        "depends_on": [1],
    },
    3: {
        "name": "assemble_environments",
        "script": "scripts/assemble_environments.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step3",
        "build_command": _step3_assemble_environments_builder,
        "depends_on": [2],
    },
    4: {
        "name": "generate_splits",
        "script": "scripts/generate_splits.py",
        "required_inputs": ["paths.outputs_root", "training.split_default"],
        "default_output_path": "{paths.outputs_root}/step4",
        "build_command": _default_builder("generate_splits.py"),
        "depends_on": [3],
    },
    5: {
        "name": "run_benchmark",
        "script": "scripts/run_benchmark.py",
        "required_inputs": ["training.task", "training.label_col"],
        "default_output_path": "{paths.outputs_root}/step5",
        "build_command": _step5_run_benchmark_builder,
        "depends_on": [4],
    },
    6: {
        "name": "train_exact_objective",
        "script": "scripts/train_causal_qsar.py",
        "required_inputs": ["paths.outputs_root", "training.task", "training.label_col"],
        "default_output_path": "{paths.outputs_root}/step6",
        "build_command": _step6_train_exact_objective_builder,
        "depends_on": [5],
    },
    7: {
        "name": "generate_counterfactuals",
        "script": "scripts/generate_counterfactuals.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step7",
        "build_command": _step7_generate_counterfactuals_builder,
        "depends_on": [6],
    },
    8: {
        "name": "evaluate_model",
        "script": "scripts/evaluate_model.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step8",
        "build_command": _step8_evaluate_model_builder,
        "depends_on": [7],
    },
    9: {
        "name": "evaluate_cross_endpoint",
        "script": "scripts/evaluate_cross_endpoint.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step9",
        "build_command": _step9_evaluate_cross_endpoint_builder,
        "depends_on": [8],
    },
    10: {
        "name": "interpret_model",
        "script": "scripts/interpret_model.py",
        "required_inputs": ["paths.outputs_root", "style.font"],
        "default_output_path": "{paths.outputs_root}/step10",
        "build_command": _step10_interpret_model_builder,
        "depends_on": [9],
    },
    11: {
        "name": "evaluate_robustness",
        "script": "scripts/evaluate_robustness.py",
        "required_inputs": ["robustness.ensemble_size"],
        "default_output_path": "{paths.outputs_root}/step11",
        "build_command": _default_builder("evaluate_robustness.py", include_style=True),
        "depends_on": [10],
    },
    12: {
        "name": "screen_library",
        "script": "scripts/screen_library.py",
        "required_inputs": ["screening.input_format", "screening.smiles_col_name"],
        "default_output_path": "{paths.outputs_root}/step12",
        "build_command": _default_builder("screen_library.py", include_style=True),
        "depends_on": [11],
    },
    13: {
        "name": "analyze_screening",
        "script": "scripts/analyze_screening.py",
        "required_inputs": ["paths.outputs_root", "screening.topk"],
        "default_output_path": "{paths.outputs_root}/step13",
        "build_command": _default_builder("analyze_screening.py", include_style=True),
        "depends_on": [12],
    },
    14: {
        "name": "match_screening_features",
        "script": "scripts/match_screening_features.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step14",
        "build_command": _default_builder("match_screening_features.py", include_style=True),
        "depends_on": [13],
    },
    15: {
        "name": "build_manuscript_pack",
        "script": "scripts/build_manuscript_pack.py",
        "required_inputs": ["paper_id", "paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step15",
        "build_command": _default_builder("build_manuscript_pack.py", include_style=True),
        "depends_on": [14],
    },
}


def parse_step_range(spec: str, valid_steps: set[int] | None = None) -> list[int]:
    parsed: set[int] = set()
    for chunk in [part.strip() for part in spec.split(",") if part.strip()]:
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Invalid range {chunk}: start>end")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(chunk))
    steps = sorted(parsed)
    if valid_steps is not None:
        bad = [step for step in steps if step != 0 and step not in valid_steps]
        if bad:
            raise ValueError(f"Unregistered steps requested: {bad}")
    return steps


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor
