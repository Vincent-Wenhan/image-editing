#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Reuse the project environment; never alter other projects or the NVIDIA driver.
if [[ ! -x .venv/bin/python ]]; then
  conda create --prefix "$PWD/.venv" python=3.11 pip -y
fi
.venv/bin/python -m pip install uv

python3 - <<'PY'
import json
import subprocess
from pathlib import Path

lock = json.loads(Path('configs/assets.lock.json').read_text())
repositories = dict(lock['repositories'])
repositories['diffusers'] = dict(url='https://github.com/huggingface/diffusers.git',
                                 revision=lock['diffusers_revision'])
for name, repo in repositories.items():
    destination = Path('external') / name
    if destination.exists():
        current = subprocess.check_output(
            ['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
        if current != repo['revision']:
            raise SystemExit(f'Revision mismatch: {destination}; inspect before changing it')
        continue
    subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                    repo['url'], str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', repo['revision']], check=True)
PY

if [[ -f requirements.lock ]]; then
  .venv/bin/uv pip install --python .venv/bin/python -r requirements.lock
else
  .venv/bin/uv pip install --python .venv/bin/python -r requirements-train.in
  .venv/bin/uv pip install --python .venv/bin/python ./external/diffusers -e '.[dev]'
fi
.venv/bin/uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pip check
git config core.hooksPath .githooks
