#!/usr/bin/env bash

manual_parse_common() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [overrides...]" >&2
    return 1
  fi
  CONFIG="$1"
  shift
  EXTRA_ARGS=("$@")
  if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    return 1
  fi
}

manual_fail_preflight() {
  local msg="$1"
  echo "[$(basename "$0")] ERROR: $msg" >&2
  exit 2
}

manual_require_file() {
  local path="$1"
  local hint="${2:-}"
  [[ -f "$path" ]] || manual_fail_preflight "missing required file: $path${hint:+ ($hint)}"
}

manual_require_dir() {
  local path="$1"
  local hint="${2:-}"
  [[ -d "$path" ]] || manual_fail_preflight "missing required directory: $path${hint:+ ($hint)}"
}

manual_require_columns() {
  local python_bin="$1"
  local file_path="$2"
  local columns_csv="$3"
  "$python_bin" - "$file_path" "$columns_csv" <<'PY' || exit 2
import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1])
required = [c for c in sys.argv[2].split(',') if c]
if not path.exists():
    raise SystemExit(f"missing required tabular file: {path}")
if path.suffix.lower() == '.parquet':
    df = pd.read_parquet(path)
else:
    df = pd.read_csv(path)
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"missing required columns in {path}: {missing}; available={list(df.columns)}")
PY
}

manual_python_for_config() {
  local config_path="$1"
  local bootstrap_py="${PIPELINE_PYTHON:-python}"
  "$bootstrap_py" - "$config_path" <<'PY'
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required to parse config: {exc}")

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}

runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
runtime_python = runtime.get("python")
if runtime_python:
    print(runtime_python)
else:
    print(os.environ.get("PIPELINE_PYTHON") or "python")
PY
}

manual_style_flags() {
  local config_path="$1"
  local bootstrap_py="${PIPELINE_PYTHON:-python}"
  mapfile -t STYLE_FLAGS < <("$bootstrap_py" - "$config_path" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required to parse config: {exc}")

STYLE_KEYS = [
    "svg_only",
    "font",
    "bold_text",
    "palette",
    "font_title",
    "font_label",
    "font_tick",
    "font_legend",
]
STYLE_ARG_ALIASES = {"svg_only": "svg"}

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}

style = cfg.get("style", {}) if isinstance(cfg.get("style"), dict) else {}
for key in STYLE_KEYS:
    if key not in style or style[key] is None:
        continue
    cli_key = STYLE_ARG_ALIASES.get(key, key)
    value = style[key]
    if isinstance(value, bool):
        if value:
            print(f"--{cli_key}")
        continue
    print(f"--{cli_key}")
    print(str(value))
PY
)
}

manual_append_overrides() {
  local -n _extra_ref=$1
  local -n _cmd_ref=$2
  local arg
  for arg in "${_extra_ref[@]}"; do
    if [[ "$arg" == *=* ]]; then
      local key="${arg%%=*}"
      local value="${arg#*=}"
      case "$key" in
        training.splits_to_run|run_dir|runs_root|smoke)
          ;;
        training.epochs)
          _cmd_ref+=("--epochs" "$value")
          ;;
        training.max_rows)
          _cmd_ref+=("--max_rows" "$value")
          ;;
        training.seeds)
          ;;
        training.early_stopping_patience)
          _cmd_ref+=("--early_stopping_patience" "$value")
          ;;
        *)
          _cmd_ref+=("--$key" "$value")
          ;;
      esac
    else
      _cmd_ref+=("$arg")
    fi
  done
}

manual_smoke_enabled() {
  local python_bin="$1"
  local config_path="$2"
  shift 2
  "$python_bin" - "$config_path" "$@" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
extra = sys.argv[2:]
cfg = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}

enabled = bool(cfg.get('smoke', False))
for token in extra:
    if token.startswith('smoke='):
        enabled = token.split('=', 1)[1].strip().lower() in {'1', 'true', 'yes', 'on'}

print('1' if enabled else '0')
PY
}

manual_apply_smoke_overrides() {
  local -n _extra_ref=$1
  local found_epochs=""
  local found_max_rows=""
  local found_seeds=""
  local arg
  for arg in "${_extra_ref[@]}"; do
    [[ "$arg" == training.epochs=* ]] && found_epochs=1
    [[ "$arg" == training.max_rows=* ]] && found_max_rows=1
    [[ "$arg" == training.seeds=* ]] && found_seeds=1
  done
  [[ -n "$found_epochs" ]] || _extra_ref+=("training.epochs=1")
  [[ -n "$found_max_rows" ]] || _extra_ref+=("training.max_rows=200")
  [[ -n "$found_seeds" ]] || _extra_ref+=("training.seeds=1")
}

manual_run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "[$(basename "$0")] Running command:"
  printf '  %q' "$@"
  printf '\n'
  "$@" 2>&1 | tee -a "$log_file"
}

manual_write_run_pointer() {
  local python_bin="$1"
  local pointer_path="$2"
  local run_dir="$3"
  local producer_step="$4"
  "$python_bin" - "$pointer_path" "$run_dir" "$producer_step" <<'PY'
import json, sys
from pathlib import Path
ptr = Path(sys.argv[1]); run_dir = str(Path(sys.argv[2]).resolve()); step = sys.argv[3]
ptr.parent.mkdir(parents=True, exist_ok=True)
ptr.write_text(json.dumps({"run_dir": run_dir, "producer_step": step}, indent=2), encoding="utf-8")
print(f"Wrote run pointer: {ptr} -> {run_dir}")
PY
}

manual_read_run_pointer() {
  local python_bin="$1"
  local pointer_path="$2"
  "$python_bin" - "$pointer_path" <<'PY'
import json, sys
from pathlib import Path
ptr = Path(sys.argv[1])
if not ptr.exists():
    print("")
    raise SystemExit(0)
data = json.loads(ptr.read_text(encoding='utf-8'))
print(data.get('run_dir', ''))
PY
}

manual_get_override() {
  local key="$1"
  shift
  local -a args=("$@")
  local value=""
  local i=0
  while [[ $i -lt ${#args[@]} ]]; do
    local token="${args[$i]}"
    if [[ "$token" == "$key="* ]]; then
      value="${token#*=}"
    elif [[ "$token" == "--$key" ]]; then
      if [[ $((i+1)) -lt ${#args[@]} ]]; then
        value="${args[$((i+1))]}"
        i=$((i+1))
      fi
    fi
    i=$((i+1))
  done
  echo "$value"
}

manual_resolve_splits_to_run() {
  local python_bin="$1"
  local config_path="$2"
  local splits_root="$3"
  shift 3
  "$python_bin" - "$config_path" "$splits_root" "$@" <<'PY'
import json
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
splits_root = Path(sys.argv[2])
extra = sys.argv[3:]
cfg = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
training = cfg.get('training', {}) if isinstance(cfg.get('training'), dict) else {}
default_split = str(training.get('split_default', 'scaffold_bm'))

raw = None
for token in extra:
    if token.startswith('training.splits_to_run='):
        raw = token.split('=', 1)[1]

if raw is None or str(raw).strip() == '' or str(raw).strip().lower() in {'null', 'none', 'unset'}:
    selected = [default_split]
else:
    val = str(raw).strip()
    if val.lower() == 'all':
        selected = sorted([p.name for p in splits_root.iterdir() if p.is_dir()]) if splits_root.exists() else []
    else:
        if val.startswith('[') and val.endswith(']'):
            try:
                parsed = json.loads(val)
                selected = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                selected = [s.strip() for s in val.strip('[]').split(',') if s.strip()]
        else:
            selected = [s.strip() for s in val.split(',') if s.strip()]

for name in selected:
    print(name)
PY
}

manual_resolve_step12_screen_dir() {
  local python_bin="$1"
  local step12_root="$2"
  local explicit_screen_dir="${3:-}"
  "$python_bin" - "$step12_root" "$explicit_screen_dir" <<'PY'
import sys
from pathlib import Path

repo_root = Path.cwd()
scripts_dir = repo_root / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from screening_compat import resolve_step12_screen_outputs

step12_root = Path(sys.argv[1])
explicit = sys.argv[2] or None
resolved = resolve_step12_screen_outputs(step12_root, explicit_screen_dir=explicit)
print(str(resolved["screen_dir"]))
PY
}
