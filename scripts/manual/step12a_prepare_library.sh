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
print(s.get('input_path',''))
print(s.get('input_format','csv'))
print(s.get('sep',','))
print(s.get('header','auto'))
print(s.get('smiles_col_name','smiles'))
print(s.get('id_col_name','compound_id'))
print(s.get('name_col_name',''))
print(s.get('smi_layout','smiles_id'))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; INPUT_PATH="${CFG[2]}"; INPUT_FORMAT="${CFG[3]}"; SEP="${CFG[4]}"; HEADER="${CFG[5]}"; SMILES_COL="${CFG[6]}"; ID_COL="${CFG[7]}"; NAME_COL="${CFG[8]}"; SMI_LAYOUT="${CFG[9]}"
STEP_OUT="$OUTPUTS_ROOT/step12"; PREP_ROOT="$STEP_OUT/prepared"; LOG_FILE="$STEP_OUT/step12a_prepare_library.log"; mkdir -p "$STEP_OUT"
SCREEN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
[[ -n "$INPUT_PATH" ]] || manual_fail_preflight "missing screening.input_path in config (CSV file required for step12a)"
manual_require_file "$INPUT_PATH" "set screening.input_path to a CSV file"
if [[ "$INPUT_FORMAT" != "csv" ]]; then
  manual_fail_preflight "step12a expects screening.input_format=csv (got: $INPUT_FORMAT)"
fi
[[ "$SEP" == "," ]] || manual_fail_preflight "step12a expects screening.sep=',' (got: $SEP)"
CMD=("$PYTHON_BIN" "scripts/prepare_library.py" "--target" "$TARGET" "--screen_id" "$SCREEN_ID" "--input_path" "$INPUT_PATH" "--input_format" "$INPUT_FORMAT" "--outdir" "$PREP_ROOT" "--header" "$HEADER" "--sep" "$SEP" "--smiles_col" "$SMILES_COL" "--id_col" "$ID_COL" "--smi_layout" "$SMI_LAYOUT")
[[ -n "$NAME_COL" ]] && CMD+=("--name_col" "$NAME_COL")
manual_append_overrides EXTRA_ARGS CMD; manual_run_with_log "$LOG_FILE" "${CMD[@]}"
PREP_DIR="$PREP_ROOT/$TARGET/$SCREEN_ID"
[[ -d "$PREP_DIR" ]] || manual_fail_preflight "missing prepared output directory: $PREP_DIR"
manual_require_file "$PREP_DIR/processed/library_dedup.parquet" "step12a output missing"
manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/latest_prepare.json" "$PREP_DIR" "step12a_prepare_library"
