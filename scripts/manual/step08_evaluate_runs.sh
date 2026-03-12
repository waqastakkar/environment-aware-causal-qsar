#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"
manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"
manual_style_flags "$CONFIG"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out_root = Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve(); training = cfg.get('training', {}) if isinstance(cfg.get('training'), dict) else {}
print(str(out_root)); print(str(cfg['target'])); print(str(training.get('task', 'regression'))); print(str(training.get('label_col', 'pIC50'))); print(str(training.get('env_col', 'env_id_manual')))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; TASK="${CFG[2]}"; LABEL_COL="${CFG[3]}"; ENV_COL="${CFG[4]}"
STEP_OUT="$OUTPUTS_ROOT/step8"; LOG_FILE="$STEP_OUT/step08_evaluate_runs.log"; mkdir -p "$STEP_OUT"

resolve_counterfactual_dir() {
  local outputs_root="$1"
  local target="$2"
  local direct="$outputs_root/step7/candidates"
  local target_dir="$outputs_root/step7/$target/candidates"
  if [[ -d "$direct" ]]; then
    echo "$direct"
    return 0
  fi
  if [[ -d "$target_dir" ]]; then
    echo "$target_dir"
    return 0
  fi

  local -a ranked_matches=()
  while IFS= read -r m; do
    ranked_matches+=("$m")
  done < <(find "$outputs_root/step7" -type f -name 'ranked_topk.parquet' 2>/dev/null | sort)

  if [[ ${#ranked_matches[@]} -eq 1 ]]; then
    dirname "${ranked_matches[0]}"
    return 0
  fi
  if [[ ${#ranked_matches[@]} -gt 1 ]]; then
    local -a target_matches=()
    local path
    for path in "${ranked_matches[@]}"; do
      [[ "$path" == *"/$target/"* ]] && target_matches+=("$path")
    done
    if [[ ${#target_matches[@]} -eq 1 ]]; then
      dirname "${target_matches[0]}"
      return 0
    fi
  fi
  echo ""
}

EXPLICIT_RUN_DIR="$(manual_get_override run_dir "${EXTRA_ARGS[@]}")"
EXPLICIT_RUNS_ROOT="$(manual_get_override runs_root "${EXTRA_ARGS[@]}")"
if [[ -n "$EXPLICIT_RUNS_ROOT" ]]; then
  RUNS_ROOT="$EXPLICIT_RUNS_ROOT"
elif [[ -n "$EXPLICIT_RUN_DIR" ]]; then
  RUNS_ROOT="$EXPLICIT_RUN_DIR"
else
  RUNS_ROOT=""
  PTR="$OUTPUTS_ROOT/step6/$TARGET/latest_run.json"
  [[ -f "$PTR" ]] || PTR="$OUTPUTS_ROOT/step5/$TARGET/latest_run.json"
  if [[ -f "$PTR" ]]; then
    RUNS_ROOT="$(manual_read_run_pointer "$PYTHON_BIN" "$PTR")"
  fi
  [[ -n "$RUNS_ROOT" ]] || RUNS_ROOT="$OUTPUTS_ROOT/step6/$TARGET"
  [[ -d "$RUNS_ROOT" ]] || RUNS_ROOT="$OUTPUTS_ROOT/step5/$TARGET"
fi

manual_require_dir "$RUNS_ROOT" "missing runs_root (step06 or step05 outputs)"
manual_require_file "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
manual_require_dir "$OUTPUTS_ROOT/step4"

CF_DIR_OVERRIDE="$(manual_get_override counterfactual_dir "${EXTRA_ARGS[@]}")"
if [[ -z "$CF_DIR_OVERRIDE" ]]; then
  CF_DIR_OVERRIDE="$(manual_get_override cf_dir "${EXTRA_ARGS[@]}")"
fi
if [[ -n "$CF_DIR_OVERRIDE" ]]; then
  CF_DIR="$CF_DIR_OVERRIDE"
else
  CF_DIR="$(resolve_counterfactual_dir "$OUTPUTS_ROOT" "$TARGET")"
fi
echo "[$(basename "$0")] Counterfactual directory resolved: ${CF_DIR:-<none>}"

CMD=("$PYTHON_BIN" "scripts/evaluate_runs.py" "--target" "$TARGET" "--runs_root" "$RUNS_ROOT" "--splits_dir" "$OUTPUTS_ROOT/step4" "--dataset_parquet" "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "--outdir" "$STEP_OUT" "--task" "$TASK" "--label_col" "$LABEL_COL" "--env_col" "$ENV_COL" "--compute_cf_consistency")
if [[ -n "$CF_DIR" ]]; then CMD+=("--counterfactual_dir" "$CF_DIR"); fi
BBB_PARQUET="$OUTPUTS_ROOT/step3/data/bbb_annotations.parquet"; [[ -f "$BBB_PARQUET" ]] || BBB_PARQUET="$OUTPUTS_ROOT/step3/bbb_annotations.parquet"
if [[ -f "$BBB_PARQUET" ]]; then CMD+=("--bbb_parquet" "$BBB_PARQUET"); fi
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
