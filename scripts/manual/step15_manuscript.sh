#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"; manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"; manual_style_flags "$CONFIG"
OUTPUTS_ROOT="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
print(Path(cfg.get('paths',{}).get('outputs_root','outputs')).resolve())
PY
)"
STEP_OUT="$OUTPUTS_ROOT/step15"; LOG_FILE="$STEP_OUT/step15_manuscript.log"; mkdir -p "$STEP_OUT"
manual_require_dir "$OUTPUTS_ROOT/step6" "run step06 first"
SCREEN_DIR=""
if [[ -d "$OUTPUTS_ROOT/step12" ]]; then
  EXPLICIT_SCREEN_DIR="$(manual_get_override screen_dir "${EXTRA_ARGS[@]}")"
  SCREEN_DIR="$(manual_resolve_step12_screen_dir "$PYTHON_BIN" "$OUTPUTS_ROOT/step12" "$EXPLICIT_SCREEN_DIR" || true)"
fi
CMD=("$PYTHON_BIN" "scripts/build_manuscript_pack.py" "--config" "$CONFIG" "${STYLE_FLAGS[@]}")
[[ -n "$SCREEN_DIR" ]] && CMD+=("--screen_dir" "$SCREEN_DIR")
manual_append_overrides EXTRA_ARGS CMD; manual_run_with_log "$LOG_FILE" "${CMD[@]}"
