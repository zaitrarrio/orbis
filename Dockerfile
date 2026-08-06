# Linux + NVIDIA GPU image for Orbis (dev, RunPod, Vast).
# CUDA 12.8 + PyTorch cu128 — RTX 4090 (Ada), RTX 5090 (Blackwell sm_120), H100 (Hopper).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=automatic \
    WORKSPACE=/workspace

WORKDIR ${WORKSPACE}

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md LICENSE .python-version ./
COPY orbis ./orbis
COPY scripts ./scripts
COPY tests ./tests

# Resolve into a project venv; torch comes from the cu128 index in pyproject/uv.lock
RUN uv sync --frozen --group dev

ENV PATH="${WORKSPACE}/.venv/bin:${PATH}" \
    VIRTUAL_ENV="${WORKSPACE}/.venv" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Interactive GPU pods (override CMD/entrypoint on RunPod or Vast as needed)
CMD ["bash", "-lc", "nvidia-smi && python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')\" && exec bash"]
