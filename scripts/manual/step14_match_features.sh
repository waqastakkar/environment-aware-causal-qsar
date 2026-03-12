#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"; manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"; manual_style_flags "$CONFIG"

readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out=Path(cfg.get('paths',{}).get('outputs_root','outputs')).resolve()
print(out)
print(str(cfg['target']))
print(str(cfg.get('screening',{}).get('topk',100)))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; HITS_TOPK="${CFG[2]}"
STEP_OUT="$OUTPUTS_ROOT/step14"; LOG_FILE="$STEP_OUT/step14_match_features.log"; mkdir -p "$STEP_OUT"

manual_require_dir "$OUTPUTS_ROOT/step10" "run step10 first"
manual_require_dir "$OUTPUTS_ROOT/step12" "run step12 first"
manual_require_file "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "run step03 first"

EXPLICIT_SCREEN_DIR="$(manual_get_override screen_dir "${EXTRA_ARGS[@]}")"
SCREEN_DIR="$(manual_resolve_step12_screen_dir "$PYTHON_BIN" "$OUTPUTS_ROOT/step12" "$EXPLICIT_SCREEN_DIR")"
manual_require_dir "$SCREEN_DIR" "resolved screening run directory"
manual_require_file "$SCREEN_DIR/artifacts/feature_schema.json" "screening requires artifacts/feature_schema.json"

OUTDIR_OVERRIDE="$(manual_get_override outdir "${EXTRA_ARGS[@]}")"
INTERPRET_OVERRIDE="$(manual_get_override interpret_dir "${EXTRA_ARGS[@]}")"
SCREEN_ANALYSIS_OVERRIDE="$(manual_get_override screen_analysis_dir "${EXTRA_ARGS[@]}")"

TRAIN_PARQUET="$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
SCREEN_TARGET="$(basename "$(dirname "$SCREEN_DIR")")"
SCREEN_ID="$(basename "$SCREEN_DIR")"

run_for_interpret() {
  local interpret_dir="$1"
  local split_name="$2"
  local run_id="$3"

  manual_require_dir "$interpret_dir" "interpret outputs missing (run step10 first or pass --interpret_dir)"
  manual_require_file "$interpret_dir/shape/shape_descriptors.parquet" "interpret shape descriptors required"
  manual_require_file "$interpret_dir/attribution/fragment_attributions.csv" "interpret fragment attributions required"

  local outdir
  if [[ -n "$OUTDIR_OVERRIDE" ]]; then
    outdir="$OUTDIR_OVERRIDE"
  else
    outdir="$STEP_OUT/$SCREEN_TARGET/$SCREEN_ID/$split_name/$run_id"
  fi

  local screen_analysis_dir="$SCREEN_ANALYSIS_OVERRIDE"
  if [[ -z "$screen_analysis_dir" ]]; then
    screen_analysis_dir="$OUTPUTS_ROOT/step13/$SCREEN_TARGET/$SCREEN_ID"
  fi

  local cmd=("$PYTHON_BIN" "scripts/match_screening_features.py"
    "--target" "$TARGET"
    "--screen_dir" "$SCREEN_DIR"
    "--train_parquet" "$TRAIN_PARQUET"
    "--interpret_dir" "$interpret_dir"
    "--outdir" "$outdir"
    "--hits_source" "top100_diverse"
    "--hits_topk" "$HITS_TOPK"
    "${STYLE_FLAGS[@]}")

  if [[ -d "$screen_analysis_dir" ]]; then
    cmd+=("--screen_analysis_dir" "$screen_analysis_dir")
  fi

  manual_append_overrides EXTRA_ARGS cmd
  manual_run_with_log "$LOG_FILE" "${cmd[@]}"
}

resolve_interpret_dir_for_split() {
  local split_name="$1"
  local run_id="$2"

  local ptr="$OUTPUTS_ROOT/step10/$split_name/latest_run.json"
  if [[ -f "$ptr" ]]; then
    local resolved
    resolved="$(manual_read_run_pointer "$PYTHON_BIN" "$ptr")"
    if [[ -n "$resolved" ]]; then
      echo "$resolved"
      return 0
    fi
  fi

  local candidate="$OUTPUTS_ROOT/step10/$split_name/$run_id"
  echo "$candidate"
}

if [[ -n "$INTERPRET_OVERRIDE" ]]; then
  split_name="$(basename "$(dirname "$INTERPRET_OVERRIDE")")"
  run_id="$(basename "$INTERPRET_OVERRIDE")"
  run_for_interpret "$INTERPRET_OVERRIDE" "$split_name" "$run_id"
  exit 0
fi

mapfile -t SPLITS < <(manual_resolve_splits_to_run "$PYTHON_BIN" "$CONFIG" "$OUTPUTS_ROOT/step4" "${EXTRA_ARGS[@]}")
[[ ${#SPLITS[@]} -gt 0 ]] || SPLITS=("default")

resolved_any=0
for SPLIT_NAME in "${SPLITS[@]}"; do
  PTR="$OUTPUTS_ROOT/step6/$TARGET/$SPLIT_NAME/latest_run.json"
  [[ -f "$PTR" ]] || PTR="$OUTPUTS_ROOT/step5/$TARGET/$SPLIT_NAME/latest_run.json"
  [[ -f "$PTR" ]] || continue
  RUN_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$PTR")"
  [[ -n "$RUN_DIR" ]] || continue
  RUN_ID="$(basename "$RUN_DIR")"
  INTERPRET_DIR="$(resolve_interpret_dir_for_split "$SPLIT_NAME" "$RUN_ID")"
  run_for_interpret "$INTERPRET_DIR" "$SPLIT_NAME" "$RUN_ID"
  resolved_any=1
done

if [[ "$resolved_any" -eq 0 ]]; then
  PTR="$OUTPUTS_ROOT/step6/$TARGET/latest_run.json"
  [[ -f "$PTR" ]] || PTR="$OUTPUTS_ROOT/step5/$TARGET/latest_run.json"
  [[ -f "$PTR" ]] || manual_fail_preflight "missing run pointer for feature matching (expected step6/step5 latest_run.json)"
  RUN_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$PTR")"
  [[ -n "$RUN_DIR" ]] || manual_fail_preflight "run pointer did not contain run_dir: $PTR"
  SPLIT_NAME="$(basename "$(dirname "$RUN_DIR")")"
  RUN_ID="$(basename "$RUN_DIR")"
  INTERPRET_DIR="$(resolve_interpret_dir_for_split "$SPLIT_NAME" "$RUN_ID")"
  run_for_interpret "$INTERPRET_DIR" "$SPLIT_NAME" "$RUN_ID"
fi
