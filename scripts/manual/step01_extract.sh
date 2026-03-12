#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_helpers.sh"
manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"

readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out_root = Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve()
print(str(out_root))
print(str(cfg['paths']['chembl_sqlite']))
print(str(cfg['target']))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; CHEMBL_DB="${CFG[1]}"; TARGET="${CFG[2]}"
manual_require_file "$CHEMBL_DB" "set paths.chembl_sqlite in config"
STEP_OUT="$OUTPUTS_ROOT/step1"
LOG_FILE="$STEP_OUT/step01_extract.log"
mkdir -p "$STEP_OUT"

CMD=("$PYTHON_BIN" "scripts/extract_chembl36_sqlite.py" "--config" "$CONFIG" "--db" "$CHEMBL_DB" "--target" "$TARGET" "--outdir" "$STEP_OUT")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
manual_require_file "$STEP_OUT/${TARGET}_qsar_ready.csv" "step01 extract output"
