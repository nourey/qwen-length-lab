#!/usr/bin/env bash
set -euo pipefail

# Run from the qwen-length-lab checkout in a Kaggle notebook with Internet enabled.
# The server and controller use separate Python environments.
VLLM_ENV=/kaggle/temp/qwen30-vllm-env
CONTROL_ENV=/kaggle/temp/system-one-env
ADAPTER_COMMIT=0bb819b85d67a98c736d7c3004eae95f49f3daa3
VLLM_WHEEL='https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
TORCH_INDEX='https://download.pytorch.org/whl/cu129'

mkdir -p /kaggle/temp

if ! command -v uv >/dev/null 2>&1; then
  # Kaggle's sitecustomize imports wrapt, which breaks ensurepip in a clean
  # stdlib venv. Install uv with the working base interpreter, then let uv
  # create and populate the isolated environments without invoking ensurepip.
  python3 -m pip install --disable-pip-version-check --no-cache-dir --no-deps uv
fi
UV_BIN=$(command -v uv)

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  if python3 -c 'import importlib.metadata as m, torch; assert m.version("vllm").split("+", 1)[0] == "0.29.0"; assert torch.__version__.startswith("2.13.0+cu129"); assert torch.version.cuda == "12.9"' >/dev/null 2>&1 && command -v vllm >/dev/null 2>&1; then
    "$UV_BIN" venv --system-site-packages --python "$(command -v python3)" "$VLLM_ENV"
    ln -s "$(command -v vllm)" "$VLLM_ENV/bin/vllm"
  else
    "$UV_BIN" venv --python "$(command -v python3)" "$VLLM_ENV"
  fi
fi
if ! "$VLLM_ENV/bin/python" -c 'import importlib.metadata as m, torch; assert m.version("vllm").split("+", 1)[0] == "0.29.0"; assert torch.__version__.startswith("2.13.0+cu129"); assert torch.version.cuda == "12.9"' >/dev/null 2>&1; then
  if grep -qi '^include-system-site-packages = true' "$VLLM_ENV/pyvenv.cfg"; then
    echo "Existing vLLM environment inherits incompatible Kaggle packages. Remove $VLLM_ENV and rerun this script." >&2
    exit 1
  fi
  echo "Installing the official vLLM 0.29.0 CUDA 12.9 wheel in $VLLM_ENV"
  UV_CACHE_DIR=/kaggle/temp/uv-cache UV_HTTP_TIMEOUT=600 "$UV_BIN" pip install \
    --python "$VLLM_ENV/bin/python" wrapt "$VLLM_WHEEL" \
    --extra-index-url "$TORCH_INDEX"
fi
test -x "$VLLM_ENV/bin/vllm"

if [[ ! -x "$CONTROL_ENV/bin/python" ]]; then
  "$UV_BIN" venv --python "$(command -v python3)" "$CONTROL_ENV"
fi

UV_CACHE_DIR=/kaggle/temp/uv-cache UV_HTTP_TIMEOUT=600 "$UV_BIN" pip install \
  --python "$CONTROL_ENV/bin/python" wrapt \
  "system-one-adapter[openai] @ git+https://github.com/typesafe-ai/system-one-adapter-python.git@${ADAPTER_COMMIT}" \
  'openai==3.14.1' 'httpx2==2.13.0'

"$VLLM_ENV/bin/python" - <<'PY'
import importlib.metadata as m
import torch
assert m.version("vllm").split("+", 1)[0] == "0.29.0", m.version("vllm")
assert torch.__version__.startswith("2.13.0+cu129"), torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert all("T4" in torch.cuda.get_device_name(i) for i in range(2))
print("vLLM environment:", m.version("vllm"), torch.__version__, torch.version.cuda)
PY

"$CONTROL_ENV/bin/python" - <<'PY'
import importlib.metadata as m
import system_one_adapter, openai, httpx2
assert m.version("system-one-adapter") == "0.1.4", m.version("system-one-adapter")
assert openai.__version__ == "3.14.1", openai.__version__
assert httpx2.__version__ == "2.13.0", httpx2.__version__
print("Control environment:", system_one_adapter.__version__, openai.__version__, httpx2.__version__)
PY

echo "Kaggle environments ready: $VLLM_ENV and $CONTROL_ENV"
