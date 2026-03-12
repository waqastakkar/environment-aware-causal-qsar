#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"; manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"; manual_style_flags "$CONFIG"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out=Path(cfg.get('paths',{}).get('outputs_root','outputs')).resolve()
target=str(cfg['target'])
s=cfg.get('screening',{}) if isinstance(cfg.get('screening'),dict) else {}
print(out)
print(target)
print(s.get('topk',500))
print(s.get('cns_mpo_threshold',4.0))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; TOPK="${CFG[2]}"; CNS_MPO="${CFG[3]}"
STEP_OUT="$OUTPUTS_ROOT/step13"; LOG_FILE="$STEP_OUT/step13_analyze_screening.log"; mkdir -p "$STEP_OUT"
manual_require_dir "$OUTPUTS_ROOT/step12" "run step12 first"
manual_require_file "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "run step03 first"
EXPLICIT_SCREEN_DIR="$(manual_get_override screen_dir "${EXTRA_ARGS[@]}")"
SCREEN_DIR="$(manual_resolve_step12_screen_dir "$PYTHON_BIN" "$OUTPUTS_ROOT/step12" "$EXPLICIT_SCREEN_DIR")"
manual_require_dir "$SCREEN_DIR" "resolved screening run directory"

OUTDIR_OVERRIDE="$(manual_get_override outdir "${EXTRA_ARGS[@]}")"
if [[ -n "$OUTDIR_OVERRIDE" ]]; then
  OUTDIR="$OUTDIR_OVERRIDE"
else
  SCREEN_TARGET="$(basename "$(dirname "$SCREEN_DIR")")"
  SCREEN_ID="$(basename "$SCREEN_DIR")"
  OUTDIR="$STEP_OUT/$SCREEN_TARGET/$SCREEN_ID"
fi

CMD=("$PYTHON_BIN" "scripts/analyze_screening.py"
  "--target" "$TARGET"
  "--screen_dir" "$SCREEN_DIR"
  "--train_parquet" "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
  "--outdir" "$OUTDIR"
  "--topk" "$TOPK"
  "--cns_mpo_threshold" "$CNS_MPO"
  "${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD; manual_run_with_log "$LOG_FILE" "${CMD[@]}"
