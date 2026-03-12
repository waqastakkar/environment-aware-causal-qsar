#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"; manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"; manual_style_flags "$CONFIG"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out=Path(cfg.get('paths',{}).get('outputs_root','outputs')).resolve()
s=cfg.get('screening',{}) if isinstance(cfg.get('screening'),dict) else {}
print(out)
print(cfg['target'])
print(s.get('cns_mpo_threshold',4.0))
print(s.get('topk',500))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; CNS_MPO="${CFG[2]}"; TOPK="${CFG[3]}"
STEP_OUT="$OUTPUTS_ROOT/step12"; SCREEN_ROOT="$STEP_OUT/screening"; LOG_FILE="$STEP_OUT/step12b_screen_library.log"; mkdir -p "$STEP_OUT"
RUN_DIR="$(manual_get_override run_dir "${EXTRA_ARGS[@]}")"
PREP_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$STEP_OUT/latest_prepare.json")"
[[ -n "$PREP_DIR" ]] || manual_fail_preflight "missing prepared-library pointer: $STEP_OUT/latest_prepare.json (run step12a first)"
PREP_DEDUP="$PREP_DIR/processed/library_dedup.parquet"
manual_require_file "$PREP_DEDUP" "run step12a first"

RUN_ARGS=()
if [[ -n "$RUN_DIR" ]]; then
  manual_require_dir "$RUN_DIR" "explicit run_dir must exist"
  manual_require_file "$RUN_DIR/artifacts/feature_schema.json" "required for featurization"
  RUN_ARGS=("--run_dir" "$RUN_DIR")
else
  mapfile -t SPLITS < <(manual_resolve_splits_to_run "$PYTHON_BIN" "$CONFIG" "$OUTPUTS_ROOT/step4" "${EXTRA_ARGS[@]}")
  [[ ${#SPLITS[@]} -gt 0 ]] || manual_fail_preflight "no splits resolved from training.splits_to_run"
  [[ ${#SPLITS[@]} -eq 1 ]] || manual_fail_preflight "step12b requires exactly one split in multi-seed mode; got: ${SPLITS[*]}"

  SPLIT_NAME="${SPLITS[0]}"
  RUNS_ROOT="$OUTPUTS_ROOT/step6/$TARGET/$SPLIT_NAME"
  manual_require_dir "$RUNS_ROOT" "missing step6 split directory; run step06 for target=$TARGET split=$SPLIT_NAME"

  mapfile -t RUN_DIRS < <("$PYTHON_BIN" - "$RUNS_ROOT" <<'PY'
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
for candidate in sorted(runs_root.iterdir()):
    if not candidate.is_dir():
        continue
    fs = candidate / "artifacts" / "feature_schema.json"
    ckpt = candidate / "checkpoints" / "best.pt"
    if fs.exists() and ckpt.exists():
        print(str(candidate.resolve()))
PY
)

  [[ ${#RUN_DIRS[@]} -gt 0 ]] || manual_fail_preflight "no valid Step 6 runs found under $RUNS_ROOT (expected run dirs with artifacts/feature_schema.json and checkpoints/best.pt)"
  RUN_ARGS=("--run_dirs" "${RUN_DIRS[@]}")
fi

SCREEN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
CMD=("$PYTHON_BIN" "scripts/screen_library.py" "--target" "$TARGET" "--screen_id" "$SCREEN_ID" "${RUN_ARGS[@]}" "--prepared_library_path" "$PREP_DEDUP" "--outdir" "$SCREEN_ROOT" "--cns_mpo_threshold" "$CNS_MPO" "--topk" "$TOPK" "${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD; manual_run_with_log "$LOG_FILE" "${CMD[@]}"
SCREEN_DIR="$SCREEN_ROOT/$TARGET/$SCREEN_ID"
[[ -d "$SCREEN_DIR" ]] || manual_fail_preflight "missing screening output directory: $SCREEN_DIR"
manual_require_file "$SCREEN_DIR/predictions/scored_with_uncertainty.parquet" "step12b output missing"
manual_require_file "$SCREEN_DIR/predictions/predictions_ensemble.parquet" "step12b ensemble output missing"
manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/latest_screen.json" "$SCREEN_DIR" "step12b_screen_library"
# back-compat pointer for downstream steps expecting step12 root pointer
manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/run_pointer.json" "$SCREEN_DIR" "step12b_screen_library"
