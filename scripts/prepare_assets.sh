#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Existing chunks and fully verified files are reused on every attempt.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$@"; then return 0; fi
    if [[ "$attempt" == 5 ]]; then return 1; fi
    sleep 10
  done
}

component="${1:-all}"
case "$component" in all|models|data) ;; *) exit 2 ;; esac
if [[ "$component" != data ]]; then
  retry conda run --no-capture-output -n image-edit image-edit download \
    --asset all --transport ranges --payload-mirror modelscope --workers 2
fi
if [[ "$component" != models ]]; then
  retry conda run --no-capture-output -n image-edit image-edit prepare --transport shards
  conda run --no-capture-output -n image-edit image-edit validate
  conda run --no-capture-output -n image-edit image-edit gallery
fi
