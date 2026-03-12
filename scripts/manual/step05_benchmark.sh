#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"
manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"
SMOKE_ENABLED="$(manual_smoke_enabled "$PYTHON_BIN" "$CONFIG" "${EXTRA_ARGS[@]}")"
if [[ "$SMOKE_ENABLED" == "1" ]]; then
  manual_apply_smoke_overrides EXTRA_ARGS
fi
manual_style_flags "$CONFIG"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out_root = Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve()
training = cfg.get('training', {}) if isinstance(cfg.get('training'), dict) else {}
seeds = training.get('seeds') or []
print(str(out_root)); print(str(cfg['target']))
print(str(training['task'])); print(str(training['label_col'])); print(str(training.get('env_col', 'env_id')))
print(','.join(str(s) for s in seeds))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; TASK="${CFG[2]}"; LABEL_COL="${CFG[3]}"; ENV_COL="${CFG[4]}"; SEEDS="${CFG[5]}"
OVERRIDE_SEEDS="$(manual_get_override training.seeds "${EXTRA_ARGS[@]}")"
if [[ -n "$OVERRIDE_SEEDS" ]]; then
  SEEDS="${OVERRIDE_SEEDS//[[]/}"; SEEDS="${SEEDS//[]]/}"; SEEDS="${SEEDS// /}"
fi
STEP_OUT="$OUTPUTS_ROOT/step5"
LOG_FILE="$STEP_OUT/step05_benchmark.log"
mkdir -p "$STEP_OUT"
manual_require_file "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "run step03 first"
manual_require_dir "$OUTPUTS_ROOT/step4" "run step04 first"
mapfile -t SPLITS < <(manual_resolve_splits_to_run "$PYTHON_BIN" "$CONFIG" "$OUTPUTS_ROOT/step4" "${EXTRA_ARGS[@]}")
[[ ${#SPLITS[@]} -gt 0 ]] || manual_fail_preflight "no splits resolved from training.splits_to_run"

LAST_RUN=""
for SPLIT_NAME in "${SPLITS[@]}"; do
  manual_require_dir "$OUTPUTS_ROOT/step4/$SPLIT_NAME" "split '$SPLIT_NAME' missing; run step04"
  CMD=("$PYTHON_BIN" "scripts/run_benchmark.py" "--target" "$TARGET" "--dataset_parquet" "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "--splits_dir" "$OUTPUTS_ROOT/step4" "--split_names" "$SPLIT_NAME" "--outdir" "$STEP_OUT" "--task" "$TASK" "--label_col" "$LABEL_COL" "--env_col" "$ENV_COL")
  if [[ -n "$SEEDS" ]]; then CMD+=("--seeds" "$SEEDS"); fi
  CMD+=("${STYLE_FLAGS[@]}")
  manual_append_overrides EXTRA_ARGS CMD
  manual_run_with_log "$LOG_FILE" "${CMD[@]}"

  BEST_RUN="$("$PYTHON_BIN" - "$OUTPUTS_ROOT" "$TARGET" "$SPLIT_NAME" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]) / 'step5' / sys.argv[2] / sys.argv[3]
cands = sorted(root.glob('*/checkpoints/best.pt'), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
print(str(cands[0].parent.parent.resolve()) if cands else '')
PY
)"
  [[ -n "$BEST_RUN" ]] || manual_fail_preflight "no benchmark run with checkpoints/best.pt for split $SPLIT_NAME"
  manual_require_file "$BEST_RUN/artifacts/feature_schema.json" "benchmark run missing feature schema"
  manual_require_file "$BEST_RUN/predictions/test_predictions.parquet" "benchmark run missing test predictions"

  manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/run_pointer.json" "$BEST_RUN" "step05_benchmark"
  manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/$TARGET/latest_run.json" "$BEST_RUN" "step05_benchmark"
  manual_write_run_pointer "$PYTHON_BIN" "$OUTPUTS_ROOT/step6/$TARGET/latest_run.json" "$BEST_RUN" "step05_benchmark"
  manual_write_run_pointer "$PYTHON_BIN" "$OUTPUTS_ROOT/step6/$TARGET/$SPLIT_NAME/latest_run.json" "$BEST_RUN" "step05_benchmark"
  LAST_RUN="$BEST_RUN"
done

[[ -n "$LAST_RUN" ]] || manual_fail_preflight "step05 completed without producing runs"
