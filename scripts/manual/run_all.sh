#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [step-range] [overrides...]" >&2
  exit 1
fi
CONFIG="$1"; shift
STEP_RANGE="${STEPS:-${1:-1-15}}"
if [[ $# -gt 0 && "$1" != *=* && "$1" != --* ]]; then
  shift
fi
EXTRA_ARGS=("$@")

in_range() {
  local step="$1" token a b
  IFS=',' read -ra tokens <<< "$STEP_RANGE"
  for token in "${tokens[@]}"; do
    token="${token// /}"
    [[ -z "$token" ]] && continue
    if [[ "$token" == *-* ]]; then
      a="${token%-*}"; b="${token#*-}"
      if (( step >= a && step <= b )); then return 0; fi
    else
      if (( step == token )); then return 0; fi
    fi
  done
  return 1
}

ordered_steps=(
  "1:scripts/manual/step01_extract.sh"
  "2:scripts/manual/step02_postprocess.sh"
  "3:scripts/manual/step03_assemble_environments.sh"
  "4:scripts/manual/step04_generate_splits.sh"
  "5:scripts/manual/step05_benchmark.sh"
  "6:scripts/manual/step06_train_causal.sh"
  "7:scripts/manual/step07_counterfactuals.sh"
  "8:scripts/manual/step08_evaluate_runs.sh"
  "8:scripts/manual/step08a_prepare_external_inhibition.sh"
  "9:scripts/manual/step09_cross_endpoint.sh"
  "10:scripts/manual/step10_interpret.sh"
  "11:scripts/manual/step11_robustness.sh"
  "12:scripts/manual/step12a_prepare_library.sh"
  "12:scripts/manual/step12b_screen_library.sh"
  "13:scripts/manual/step13_analyze_screening.sh"
  "14:scripts/manual/step14_match_features.sh"
  "15:scripts/manual/step15_manuscript.sh"
)

for item in "${ordered_steps[@]}"; do
  n="${item%%:*}"; script="${item#*:}"
  if in_range "$n"; then
    bash "$script" "$CONFIG" "${EXTRA_ARGS[@]}"
  fi
done
