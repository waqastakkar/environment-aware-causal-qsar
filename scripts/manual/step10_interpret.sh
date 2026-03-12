#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"
manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"
manual_style_flags "$CONFIG"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out_root = Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve(); print(str(out_root)); print(str(cfg['target']))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"
STEP_OUT="$OUTPUTS_ROOT/step10"; LOG_FILE="$STEP_OUT/step10_interpret.log"; mkdir -p "$STEP_OUT"
DATASET_PARQUET="$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
manual_require_file "$DATASET_PARQUET" "run step03 first"

EXPLICIT_RUN_DIR="$(manual_get_override run_dir "${EXTRA_ARGS[@]}")"
EXPLICIT_RUNS_ROOT="$(manual_get_override runs_root "${EXTRA_ARGS[@]}")"
RUN_DIRS=()
if [[ -n "$EXPLICIT_RUN_DIR" ]]; then
  RUN_DIRS+=("$EXPLICIT_RUN_DIR")
elif [[ -n "$EXPLICIT_RUNS_ROOT" ]]; then
  while IFS= read -r p; do RUN_DIRS+=("$p"); done < <(find "$EXPLICIT_RUNS_ROOT" -type f -path '*/checkpoints/best.pt' -print | sed 's|/checkpoints/best.pt$||' | sort)
else
  mapfile -t SPLITS < <(manual_resolve_splits_to_run "$PYTHON_BIN" "$CONFIG" "$OUTPUTS_ROOT/step4" "${EXTRA_ARGS[@]}")
  for SPLIT_NAME in "${SPLITS[@]}"; do
    PTR="$OUTPUTS_ROOT/step6/$TARGET/$SPLIT_NAME/latest_run.json"
    [[ -f "$PTR" ]] || PTR="$OUTPUTS_ROOT/step5/$TARGET/$SPLIT_NAME/latest_run.json"
    [[ -f "$PTR" ]] || continue
    RD="$(manual_read_run_pointer "$PYTHON_BIN" "$PTR")"
    [[ -n "$RD" ]] && RUN_DIRS+=("$RD")
  done
  if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    PTR="$OUTPUTS_ROOT/step6/$TARGET/latest_run.json"
    [[ -f "$PTR" ]] || PTR="$OUTPUTS_ROOT/step5/$TARGET/latest_run.json"
    [[ -f "$PTR" ]] && RUN_DIRS+=("$(manual_read_run_pointer "$PYTHON_BIN" "$PTR")")
  fi
fi
[[ ${#RUN_DIRS[@]} -gt 0 ]] || manual_fail_preflight "no run_dir resolved for interpret"

for RUN_DIR in "${RUN_DIRS[@]}"; do
  manual_require_file "$RUN_DIR/checkpoints/best.pt"
  SPLIT_NAME="$(basename "$(dirname "$RUN_DIR")")"
  RUN_ID="$(basename "$RUN_DIR")"
  OUTDIR="$STEP_OUT/$SPLIT_NAME/$RUN_ID"
  echo "[step10] Starting interpret run split=$SPLIT_NAME run_id=$RUN_ID run_dir=$RUN_DIR" | tee -a "$LOG_FILE"
  CMD=("$PYTHON_BIN" "scripts/interpret_model.py" "--target" "$TARGET" "--run_dir" "$RUN_DIR" "--dataset_parquet" "$DATASET_PARQUET" "--outdir" "$OUTDIR")
  BBB_PARQUET="$OUTPUTS_ROOT/step3/data/bbb_annotations.parquet"; [[ -f "$BBB_PARQUET" ]] || BBB_PARQUET="$OUTPUTS_ROOT/step3/bbb_annotations.parquet"
  if [[ -f "$BBB_PARQUET" ]]; then CMD+=("--bbb_parquet" "$BBB_PARQUET"); fi
  if [[ -f "$OUTPUTS_ROOT/step7/candidates/ranked_topk.parquet" ]]; then CMD+=("--counterfactuals_parquet" "$OUTPUTS_ROOT/step7/candidates/ranked_topk.parquet"); fi

  # Safe defaults for memory-constrained interpretation runs.
  # Can be overridden via EXTRA_ARGS, e.g. --max_compounds 2000 --skip_shape
  CMD+=("--max_compounds" "5000")

  CMD+=("${STYLE_FLAGS[@]}")
  manual_append_overrides EXTRA_ARGS CMD
  manual_run_with_log "$LOG_FILE" "${CMD[@]}"

  # Persist pointers so downstream steps can resolve the exact interpret artifacts
  # without guessing by checkpoint IDs.
  manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/latest_run.json" "$OUTDIR" "step10_interpret"
  manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/$SPLIT_NAME/latest_run.json" "$OUTDIR" "step10_interpret"
done
