#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"
manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"
OUTPUTS_ROOT="$($PYTHON_BIN - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
print(Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve())
PY
)"
STEP_OUT="$OUTPUTS_ROOT/step4"
LOG_FILE="$STEP_OUT/step04_generate_splits.log"
mkdir -p "$STEP_OUT"
manual_require_file "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "run step03_assemble_environments first"
CMD=("$PYTHON_BIN" "scripts/generate_splits.py" "--config" "$CONFIG")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"

"$PYTHON_BIN" - "$STEP_OUT" <<'PY'
import json
from pathlib import Path
step4 = Path(__import__('sys').argv[1])
splits = sorted([p.name for p in step4.iterdir() if p.is_dir() and (p / 'train_ids.csv').exists()])
manifest = {
    'splits_root': str(step4.resolve()),
    'split_names': splits,
    'required_files_per_split': ['train_ids.csv', 'val_ids.csv', 'test_ids.csv', 'split_config.json'],
}
(step4 / 'splits_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f"Wrote split manifest: {step4 / 'splits_manifest.json'}")
if not splits:
    raise SystemExit('No split directories found with train_ids.csv after step04')
PY
