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
env = cfg.get('environments', {}) if isinstance(cfg.get('environments'), dict) else {}
bbb_rules = env.get('bbb_rules', str(Path('configs') / 'bbb_rules.yaml'))
series_rules = env.get('series_rules')
env_keys = env.get('env_keys') or []
print(str(out_root)); print(target); print(str(bbb_rules)); print(str(series_rules or ''))
for k in env_keys: print(f"ENVKEY::{k}")
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; BBB_RULES="${CFG[2]}"; SERIES_RULES="${CFG[3]}"
STEP_OUT="$OUTPUTS_ROOT/step3"
LOG_FILE="$STEP_OUT/step03_assemble_environments.log"
mkdir -p "$STEP_OUT"
manual_require_file "$OUTPUTS_ROOT/step2/row_level_with_pIC50.csv" "run step02_postprocess first"
manual_require_file "$OUTPUTS_ROOT/step2/compound_level_with_properties.csv" "run step02_postprocess first"
manual_require_file "$OUTPUTS_ROOT/step1/${TARGET}_qsar_ready.csv" "run step01_extract first"
manual_require_file "$BBB_RULES" "bbb rules config"
CMD=("$PYTHON_BIN" "scripts/assemble_environments.py"
  "--target" "$TARGET"
  "--row_level_csv" "$OUTPUTS_ROOT/step2/row_level_with_pIC50.csv"
  "--compound_level_csv" "$OUTPUTS_ROOT/step2/compound_level_with_properties.csv"
  "--raw_extract_csv" "$OUTPUTS_ROOT/step1/${TARGET}_qsar_ready.csv"
  "--outdir" "$STEP_OUT"
  "--bbb_rules" "$BBB_RULES")
if [[ -n "$SERIES_RULES" ]]; then CMD+=("--series_rules" "$SERIES_RULES"); fi
ENV_KEYS=()
for line in "${CFG[@]:4}"; do
  if [[ "$line" == ENVKEY::* ]]; then
    ENV_KEYS+=("${line#ENVKEY::}")
  fi
done
if [[ ${#ENV_KEYS[@]} -gt 0 ]]; then
  CMD+=("--env_keys" "${ENV_KEYS[@]}")
fi
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
manual_require_file "$STEP_OUT/multienv_compound_level.parquet"
manual_require_columns "$PYTHON_BIN" "$STEP_OUT/multienv_compound_level.parquet" "molecule_id,smiles,pIC50,env_id_manual"
