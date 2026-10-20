#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep this project isolated from base and all other project environments.
if ! conda env list --json | python3 -c 'import json,sys; from pathlib import Path; sys.exit(not any(Path(p).name == "image-edit" for p in json.load(sys.stdin)["envs"]))'; then
  if [[ -f configs/conda-linux-64.lock ]]; then
    conda create -n image-edit --file configs/conda-linux-64.lock -y
  else
    conda env create -f environment.yml
  fi
fi
if ! conda run -n image-edit sh -c 'test -n "${CC:-}" && test -x "$CC"'; then
  conda install -n image-edit gcc_linux-64 gxx_linux-64 -y
fi
PROJECT_PYTHON=$(conda run -n image-edit python -c 'import sys; print(sys.executable)')
"$PROJECT_PYTHON" -m pip install uv
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"

python3 -m src.infra.repositories

if [[ -f requirements.lock ]]; then
  "$PROJECT_PYTHON" -m uv pip install --python "$PROJECT_PYTHON" -r requirements.lock
else
  "$PROJECT_PYTHON" -m uv pip install --python "$PROJECT_PYTHON" \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118
  "$PROJECT_PYTHON" -m uv pip install --python "$PROJECT_PYTHON" -r requirements-train.in
  "$PROJECT_PYTHON" -m uv pip install --python "$PROJECT_PYTHON" ./external/diffusers -e '.[dev]'
fi
"$PROJECT_PYTHON" -m uv pip install --python "$PROJECT_PYTHON" --no-deps -e .
"$PROJECT_PYTHON" -m pip check
git config core.hooksPath .githooks
