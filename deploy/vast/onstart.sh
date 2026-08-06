#!/usr/bin/env bash
# Vast.ai on-start: keep env visible under SSH/Jupyter and land in the workspace.
set -euo pipefail
env >> /etc/environment || true
cd /workspace 2>/dev/null || cd /
nvidia-smi || true
python - <<'PY' || true
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
# Keep the instance alive for interactive use (SSH / Jupyter attach).
exec sleep infinity
