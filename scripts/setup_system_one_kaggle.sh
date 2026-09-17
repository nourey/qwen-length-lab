#!/usr/bin/env bash
set -euo pipefail

# Run from the qwen-length-lab checkout in a Kaggle notebook with Internet enabled.
# The server and controller use separate Python environments.
VLLM_ENV=/kaggle/temp/qwen30-vllm-env
CONTROL_ENV=/kaggle/temp/system-one-env
ADAPTER_COMMIT=0bb819b85d67a98c736d7c3004eae95f49f3daa3

mkdir -p /kaggle/temp

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VLLM_ENV"
fi

if ! "$VLLM_ENV/bin/python" -c 'import importlib.metadata as m; assert m.version("vllm") == "0.29.0"' >/dev/null 2>&1; then
  # Only this environment changes. Kaggle's working Python is untouched.
  "$VLLM_ENV/bin/python" -m pip install 'vllm==0.29.0'
fi
if [[ ! -x "$VLLM_ENV/bin/vllm" ]] && command -v vllm >/dev/null 2>&1; then
  # A venv inheriting Kaggle's already working vLLM sees its package but not
  # its console entry point. Reuse that entry point without reinstalling it.
  ln -s "$(command -v vllm)" "$VLLM_ENV/bin/vllm"
fi
test -x "$VLLM_ENV/bin/vllm"

if [[ ! -x "$CONTROL_ENV/bin/python" ]]; then
  python3 -m venv "$CONTROL_ENV"
fi

"$CONTROL_ENV/bin/python" -m pip install \
  "system-one-adapter[openai] @ git+https://github.com/typesafe-ai/system-one-adapter-python.git@${ADAPTER_COMMIT}" \
  'openai==3.14.1' 'httpx2==2.13.0'

"$VLLM_ENV/bin/python" - <<'PY'
import importlib.metadata as m
import torch
assert m.version("vllm") == "0.29.0", m.version("vllm")
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
