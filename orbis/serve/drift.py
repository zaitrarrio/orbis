"""Content-adaptive drift stabilization across chunk boundaries.

Detects abrupt color / identity statistic shifts between the last delivered
frame and a newly generated chunk; when the shift exceeds a threshold, blends
a conservative correction into the new chunk's first frames so long-horizon
rollouts do not visibly snap.  Never revises already-delivered frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DriftState:
    last_mean: Optional[np.ndarray] = None   # (3,)
    last_std: Optional[np.ndarray] = None


def frame_stats(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over spatial dims of the last frame ``(H,W,3)``."""
    f = frames[-1] if frames.ndim == 4 else frames
    mean = f.reshape(-1, 3).mean(axis=0).astype(np.float32)
    std = f.reshape(-1, 3).std(axis=0).astype(np.float32) + 1e-6
    return mean, std


def drift_score(prev_mean: np.ndarray, prev_std: np.ndarray,
                cur_mean: np.ndarray, cur_std: np.ndarray) -> float:
    """Normalized mean+std shift in [0, ~inf); ~0 means stable."""
    dm = np.abs(cur_mean - prev_mean).mean()
    ds = np.abs(cur_std - prev_std).mean()
    return float(dm + 0.5 * ds)


def stabilize_chunk(
    frames: np.ndarray,
    state: DriftState,
    threshold: float = 0.35,
    blend: float = 0.15,
    enabled: bool = True,
) -> np.ndarray:
    """Optionally blend the new chunk toward previous color stats.

    ``frames`` is ``(T,H,W,3)`` in [0,1]. Returns a copy when correction applies.
    """
    mean, std = frame_stats(frames)
    if (not enabled) or state.last_mean is None:
        state.last_mean, state.last_std = mean, std
        return frames

    score = drift_score(state.last_mean, state.last_std, mean, std)
    if score < threshold:
        state.last_mean, state.last_std = mean, std
        return frames

    # Conservative gain/bias match of first half of the chunk toward prev stats
    out = frames.copy()
    n = max(1, out.shape[0] // 2)
    # match: (x - cur_mean) * (prev_std/cur_std) + prev_mean, then blend
    scale = (state.last_std / (std + 1e-6)).reshape(1, 1, 1, 3)
    shift = state.last_mean.reshape(1, 1, 1, 3)
    matched = (out[:n] - mean.reshape(1, 1, 1, 3)) * scale + shift
    out[:n] = (1.0 - blend) * out[:n] + blend * np.clip(matched, 0.0, 1.0)
    state.last_mean, state.last_std = frame_stats(out)
    return out
