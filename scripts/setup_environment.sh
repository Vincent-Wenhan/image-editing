#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep this project isolated from base and all other project environments.
if ! conda env list --json | python3 -c 'import json,sys; from pathlib import Path; sys.exit(not any(Path(p).name == "image-edit" for p in json.load(sys.stdin)["envs"]))'; then
  conda create -n image-edit python=3.11 pip gcc_linux-64 gxx_linux-64 -y
fi
conda install -n image-edit gcc_linux-64 gxx_linux-64 -y
PROJECT_PYTHON=$(conda run -n image-edit python -c 'import sys; print(sys.executable)')
"$PROJECT_PYTHON" -m pip install uv
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"

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
