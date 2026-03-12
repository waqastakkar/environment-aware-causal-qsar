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
print(str(out_root))
print(str(cfg['target']))
print(str(training.get('task', 'regression')))
print(str(training.get('label_col', 'pIC50')))
print(str(training.get('env_col', 'env_id_manual')))
print(str(training.get('epochs', 300)))
print(str(training.get('early_stopping_patience', 30)))
print(",".join(str(s) for s in seeds))
PY
)

OUTPUTS_ROOT="${CFG[0]}"
TARGET="${CFG[1]}"
TASK="${CFG[2]}"
LABEL_COL="${CFG[3]}"
ENV_COL="${CFG[4]}"
EPOCHS="${CFG[5]}"
PATIENCE="${CFG[6]}"
SEEDS_CSV="${CFG[7]}"

OVERRIDE_SEEDS="$(manual_get_override training.seeds "${EXTRA_ARGS[@]}")"
if [[ -n "$OVERRIDE_SEEDS" ]]; then
  OVERRIDE_SEEDS="${OVERRIDE_SEEDS//[[]/}"
  OVERRIDE_SEEDS="${OVERRIDE_SEEDS//[]]/}"
  OVERRIDE_SEEDS="${OVERRIDE_SEEDS// /}"
  SEEDS_CSV="$OVERRIDE_SEEDS"
fi

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

STEP_OUT="$OUTPUTS_ROOT/step6"
LOG_FILE="$STEP_OUT/step06_train_causal.log"
mkdir -p "$STEP_OUT"

manual_require_file "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "run step03 first"
manual_require_dir "$OUTPUTS_ROOT/step4" "run step04 first"

mapfile -t SPLITS < <(manual_resolve_splits_to_run "$PYTHON_BIN" "$CONFIG" "$OUTPUTS_ROOT/step4" "${EXTRA_ARGS[@]}")
[[ ${#SPLITS[@]} -gt 0 ]] || manual_fail_preflight "no splits resolved from training.splits_to_run"

BBB_PARQUET="$OUTPUTS_ROOT/step3/data/bbb_annotations.parquet"
[[ -f "$BBB_PARQUET" ]] || BBB_PARQUET="$OUTPUTS_ROOT/step3/bbb_annotations.parquet"

LAST_RUN=""

for SPLIT_NAME in "${SPLITS[@]}"; do
  manual_require_dir "$OUTPUTS_ROOT/step4/$SPLIT_NAME" "split '$SPLIT_NAME' missing; run step04"

  for SEED in "${SEEDS[@]}"; do
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${SPLIT_NAME}_seed${SEED}"
    RUN_DIR="$STEP_OUT/$TARGET/$SPLIT_NAME/$RUN_ID"

    CMD=(
      "$PYTHON_BIN" "scripts/train_causal_qsar.py"
      "--target" "$TARGET"
      "--dataset_parquet" "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
      "--splits_dir" "$OUTPUTS_ROOT/step4"
      "--split_name" "$SPLIT_NAME"
      "--outdir" "$STEP_OUT"
      "--task" "$TASK"
      "--label_col" "$LABEL_COL"
      "--env_col" "$ENV_COL"
      "--epochs" "$EPOCHS"
      "--early_stopping_patience" "$PATIENCE"
      "--run_id" "$RUN_ID"
      "--seed" "$SEED"
    )

    if [[ -f "$BBB_PARQUET" ]]; then
      CMD+=("--bbb_parquet" "$BBB_PARQUET")
    fi

    CMD+=("${STYLE_FLAGS[@]}")
    manual_append_overrides EXTRA_ARGS CMD
    manual_run_with_log "$LOG_FILE" "${CMD[@]}"

    manual_require_file "$RUN_DIR/checkpoints/best.pt" "training failed to produce best checkpoint; rerun step06_train_causal.sh"
    manual_require_file "$RUN_DIR/artifacts/feature_schema.json" "training failed to write feature schema; rerun step06_train_causal.sh"
    manual_require_file "$RUN_DIR/predictions/test_predictions.parquet" "training failed to write test predictions; rerun step06_train_causal.sh"

    manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/run_pointer.json" "$RUN_DIR" "step06_train_causal"
    manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/$TARGET/latest_run.json" "$RUN_DIR" "step06_train_causal"
    manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/$TARGET/$SPLIT_NAME/latest_run.json" "$RUN_DIR" "step06_train_causal"

    LAST_RUN="$RUN_DIR"
  done
done

[[ -n "$LAST_RUN" ]] || manual_fail_preflight "step06 completed without producing runs"

