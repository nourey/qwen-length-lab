#!/usr/bin/env bash
set -euo pipefail

# Run from any directory; retain Kaggle's installed CUDA-enabled PyTorch.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_root="${QWEN_LAB_ENV:-/kaggle/temp/qwen-length-env}"
base_python="${QWEN_LAB_BASE_PYTHON:-python3}"
requirements="requirements-kaggle.txt"
case "${1:-}" in
  "") ;;
  --notebook) requirements="requirements-notebook.txt" ;;
  *) echo "Usage: bash scripts/setup_kaggle.sh [--notebook]" >&2; exit 2 ;;
esac

"$base_python" -c 'import sys; from pathlib import Path; assert sys.version_info >= (3, 10); assert Path("/kaggle").is_dir(), "Run this setup on Kaggle"'
mkdir -p "$env_root" /kaggle/working/qwen-length-runs
"$base_python" -m venv --system-site-packages "$env_root"
"$base_python" - "$env_root/torch-constraint.txt" <<'PY'
import importlib.metadata
from pathlib import Path
import sys
Path(sys.argv[1]).write_text('torch==' + importlib.metadata.version('torch') + '\n')
PY
"$env_root/bin/python" -m pip install --disable-pip-version-check \
  -c "$env_root/torch-constraint.txt" -r "$repo_root/$requirements"
if [[ "${1:-}" == "--notebook" ]]; then
  "$env_root/bin/python" -m ipykernel install --prefix "$env_root" \
    --name qwen-length-lab --display-name "Qwen Length Lab"
fi
echo "Environment ready: $env_root/bin/python"
