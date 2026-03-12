#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"; manual_parse_common "$@"
bash "$(dirname "$0")/step12a_prepare_library.sh" "$CONFIG" "${EXTRA_ARGS[@]}"
bash "$(dirname "$0")/step12b_screen_library.sh" "$CONFIG" "${EXTRA_ARGS[@]}"
