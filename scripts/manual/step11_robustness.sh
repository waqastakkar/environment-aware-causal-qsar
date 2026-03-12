#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"
manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"
manual_style_flags "$CONFIG"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out=Path(cfg.get('paths',{}).get('outputs_root','outputs')).resolve()
training = cfg.get('training', {}) if isinstance(cfg.get('training'), dict) else {}
rob = cfg.get('robustness', {}) if isinstance(cfg.get('robustness'), dict) else {}
print(out); print(cfg['target']); print(training.get('task','regression')); print(training.get('label_col','pIC50')); print(training.get('env_col','env_id_manual'))
print(rob.get('ensemble_size',5)); print(rob.get('conformal_coverage',0.90))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; TASK="${CFG[2]}"; LABEL_COL="${CFG[3]}"; ENV_COL="${CFG[4]}"; ENS_SIZE="${CFG[5]}"; CONF_COV="${CFG[6]}"
STEP_OUT="$OUTPUTS_ROOT/step11"; LOG_FILE="$STEP_OUT/step11_robustness.log"; mkdir -p "$STEP_OUT"

EXPLICIT_RUN_DIR="$(manual_get_override run_dir "${EXTRA_ARGS[@]}")"
EXPLICIT_RUNS_ROOT="$(manual_get_override runs_root "${EXTRA_ARGS[@]}")"
if [[ -n "$EXPLICIT_RUNS_ROOT" ]]; then
  RUNS_ROOT="$EXPLICIT_RUNS_ROOT"
elif [[ -n "$EXPLICIT_RUN_DIR" ]]; then
  RUNS_ROOT="$EXPLICIT_RUN_DIR"
else
  RUNS_ROOT="$OUTPUTS_ROOT/step6/$TARGET"
  [[ -d "$RUNS_ROOT" ]] || RUNS_ROOT="$OUTPUTS_ROOT/step5/$TARGET"
fi
manual_require_dir "$RUNS_ROOT" "run step06 or step05 first"
manual_require_file "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "run step03 first"
CMD=("$PYTHON_BIN" "scripts/evaluate_robustness.py" "--target" "$TARGET" "--runs_root" "$RUNS_ROOT" "--dataset_parquet" "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "--outdir" "$STEP_OUT" "--task" "$TASK" "--label_col" "$LABEL_COL" "--env_col" "$ENV_COL" "--ensemble_size" "$ENS_SIZE" "--conformal_coverage" "$CONF_COV" "${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
