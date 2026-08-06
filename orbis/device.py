"""CUDA-first device selection for Linux GPU runtimes (Docker / RunPod / Vast)."""

from __future__ import annotations

from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device]


def get_device(prefer: Optional[DeviceLike] = None) -> torch.device:
    """Return a compute device.

    Prefers CUDA when available (RTX 4090 / 5090 / H100 and other NVIDIA GPUs).
    Pass ``prefer`` to force a device (e.g. ``\"cpu\"`` in unit tests).
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device: Optional[torch.device] = None) -> str:
    """Human-readable device label for logs."""
    device = device or get_device()
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return str(device)
