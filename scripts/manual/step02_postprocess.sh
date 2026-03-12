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
target = str(cfg['target'])
post = cfg.get('postprocess', {}) if isinstance(cfg.get('postprocess'), dict) else {}
print(str(post.get('primary_endpoint', 'IC50')))
print(str(post.get('threshold', 6.0)))
print(str(post.get('aggregate', 'median')))
print(str(post.get('max_value_nM', 1e9)))
rels = post.get('allowed_relations_primary', ['='])
for rel in rels:
    print(f"REL::{rel}")
print(f"OUT_ROOT::{out_root}")
print(f"TARGET::{target}")
PY
)
PRIMARY_ENDPOINT="${CFG[0]}"; THRESHOLD="${CFG[1]}"; AGGREGATE="${CFG[2]}"; MAX_VALUE_NM="${CFG[3]}"
OUTPUTS_ROOT=""
TARGET=""
RELATIONS=()
for line in "${CFG[@]:4}"; do
  if [[ "$line" == REL::* ]]; then
    RELATIONS+=("${line#REL::}")
  elif [[ "$line" == OUT_ROOT::* ]]; then
    OUTPUTS_ROOT="${line#OUT_ROOT::}"
  elif [[ "$line" == TARGET::* ]]; then
    TARGET="${line#TARGET::}"
  fi
done

[[ -n "$OUTPUTS_ROOT" ]] || manual_fail_preflight "unable to resolve outputs_root from config"
[[ -n "$TARGET" ]] || manual_fail_preflight "unable to resolve target from config"

STEP_OUT="$OUTPUTS_ROOT/step2"
INPUT_CSV_OVERRIDE="$(manual_get_override input "${EXTRA_ARGS[@]}")"
if [[ -z "$INPUT_CSV_OVERRIDE" ]]; then
  INPUT_CSV_OVERRIDE="$(manual_get_override postprocess.input "${EXTRA_ARGS[@]}")"
fi
if [[ -n "$INPUT_CSV_OVERRIDE" && "$INPUT_CSV_OVERRIDE" = /* ]]; then
  INPUT_CSV="$INPUT_CSV_OVERRIDE"
else
  INPUT_CSV="$OUTPUTS_ROOT/step1/${TARGET}_qsar_ready.csv"
fi
LOG_FILE="$STEP_OUT/step02_postprocess.log"
mkdir -p "$STEP_OUT"
echo "[$(basename "$0")] Resolved Step1 input: $INPUT_CSV"
manual_require_file "$INPUT_CSV" "run step01_extract first"
manual_require_columns "$PYTHON_BIN" "$INPUT_CSV" "canonical_smiles,molecule_chembl_id,standard_type,standard_units,standard_relation,standard_value"
CMD=("$PYTHON_BIN" "scripts/qsar_postprocess.py" "--config" "$CONFIG" "--input" "$INPUT_CSV" "--outdir" "$STEP_OUT" "--endpoint" "$PRIMARY_ENDPOINT" "--threshold" "$THRESHOLD" "--aggregate" "$AGGREGATE" "--max_value_nM" "$MAX_VALUE_NM")
if [[ ${#RELATIONS[@]} -gt 0 ]]; then
  CMD+=("--relation_keep" "${RELATIONS[@]}")
fi
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
manual_require_file "$STEP_OUT/row_level_primary.csv"
manual_require_file "$STEP_OUT/compound_level_with_properties.csv"
