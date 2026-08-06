"""Evaluation on the toy benchmark.

Because the world is known, we can measure prompt following and stability
directly rather than via a learned reward.  Following the paper's principle, a
whole *event* (a full rollout) is the unit of analysis, not a single frame:

* ``prompt alignment`` -- do the rendered color / shape / motion match the
  active instruction?
* ``temporal stability`` -- is the object's appearance consistent frame to
  frame (no flicker / identity drift)?
* ``long-horizon drift`` -- does alignment hold over a long rollout?
* ``switch compliance`` -- after a mid-rollout prompt switch, do the frames
  before the boundary stay on the old attribute and those after adopt the new?
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .vocab import COLORS, BACKGROUND, DIRECTIONS
from .world import render_frame, SceneSpec


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------
def foreground_mask(frame: np.ndarray, thr: float = 0.12) -> np.ndarray:
    bg = np.array(BACKGROUND, dtype=np.float32)
    return np.abs(frame - bg).max(axis=-1) > thr


def detect_color(frame: np.ndarray) -> Optional[str]:
    m = foreground_mask(frame)
    if m.sum() < 2:
        return None
    mean = frame[m].mean(axis=0)
    best, bestd = None, 1e9
    for name, rgb in COLORS.items():
        d = float(((mean - np.array(rgb)) ** 2).sum())
        if d < bestd:
            best, bestd = name, d
    return best


def detect_centroid(frame: np.ndarray):
    m = foreground_mask(frame)
    if m.sum() < 2:
        return None
    ys, xs = np.nonzero(m)
    h, w = frame.shape[:2]
    return (float(xs.mean()) / w, float(ys.mean()) / h)


def detect_shape(frame: np.ndarray) -> Optional[str]:
    m = foreground_mask(frame)
    area = float(m.sum())
    if area < 3:
        return None
    h, w = frame.shape[:2]
    c = detect_centroid(frame)
    radius = (area / (h * w)) ** 0.5 * 0.6  # rough normalized radius
    best, best_iou = None, -1.0
    for shape in ("circle", "square", "triangle"):
        spec = SceneSpec(shape=shape, color="white", x=c[0], y=c[1])
        # scale radius toward the detected area
        proto = foreground_mask(_render_radius(spec, radius, h, w))
        inter = np.logical_and(m, proto).sum()
        union = np.logical_or(m, proto).sum()
        iou = inter / max(union, 1)
        if iou > best_iou:
            best, best_iou = shape, iou
    return best


def _render_radius(spec: SceneSpec, radius: float, h: int, w: int):
    from .world import BASE_RADIUS
    scale = radius / BASE_RADIUS
    from dataclasses import replace
    # emulate size by picking the nearest size multiplier is coarse; instead
    # temporarily monkey-free render by scaling through the size field proxy.
    from .vocab import SIZES
    # choose the size key closest to the requested scale
    key = min(SIZES, key=lambda k: abs(SIZES[k] - scale))
    return render_frame(replace(spec, size=key), h, w)


# ---------------------------------------------------------------------------
# Rollout-level metrics
# ---------------------------------------------------------------------------
def analyze_rollout(frames: np.ndarray, controls: Dict[str, str]) -> Dict:
    colors = [detect_color(f) for f in frames]
    shapes = [detect_shape(f) for f in frames]
    cents = [detect_centroid(f) for f in frames]
    valid = [c for c in colors if c is not None]

    color_acc = np.mean([c == controls.get("color") for c in colors if c]) \
        if valid else 0.0
    shape_acc = np.mean([s == controls.get("shape") for s in shapes if s]) \
        if any(shapes) else 0.0
    # temporal stability: fraction of frames on the modal color (no flicker)
    if valid:
        modal = max(set(valid), key=valid.count)
        stability = valid.count(modal) / len(valid)
    else:
        stability = 0.0
    # motion: mean centroid displacement per frame
    disp = []
    for a, b in zip(cents, cents[1:]):
        if a and b:
            disp.append(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
    motion = float(np.mean(disp)) if disp else 0.0
    # direction agreement with the requested motion
    dir_ok = _direction_agreement(cents, controls.get("direction"))
    return {"color_acc": float(color_acc), "shape_acc": float(shape_acc),
            "stability": float(stability), "motion": motion,
            "direction_ok": dir_ok, "visible": len(valid) / max(len(frames), 1)}


def _direction_agreement(cents, direction) -> float:
    if direction is None:
        return 0.0
    want = np.array(DIRECTIONS[direction], dtype=np.float32)
    dots = []
    for a, b in zip(cents, cents[1:]):
        if a and b:
            mv = np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)
            if np.linalg.norm(mv) > 1e-4:
                dots.append(float(np.dot(mv, want) / (np.linalg.norm(mv) + 1e-6)))
    if not dots:
        return 0.0
    return float(np.mean([d > 0.3 for d in dots]))


def switch_compliance(frames: np.ndarray, boundary: int,
                      old_color: str, new_color: str) -> Dict:
    """Fraction of pre-boundary frames on ``old_color`` and post on ``new_color``."""
    before = [detect_color(f) for f in frames[:boundary]]
    after = [detect_color(f) for f in frames[boundary:]]
    before = [c for c in before if c]
    after = [c for c in after if c]
    pre = np.mean([c == old_color for c in before]) if before else 0.0
    post = np.mean([c == new_color for c in after]) if after else 0.0
    return {"pre_old": float(pre), "post_new": float(post)}


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------
def default_cases() -> List[str]:
    return [
        "a red circle moving right",
        "a blue square moving down",
        "a green triangle moving up",
        "a yellow circle moving left",
        "a cyan square moving upright",
        "a magenta triangle moving downleft",
        "a white circle moving up fast",
        "a big red square moving right",
    ]


def evaluate(engine, cases: Optional[List[str]] = None, n_chunks: int = 4,
             seed: int = 0) -> Dict:
    """Run each case and aggregate rollout metrics."""
    from .text import parse_controls
    cases = cases or default_cases()
    rows = []
    for prompt in cases:
        res = engine.generate_video(prompt, n_chunks=n_chunks, seed=seed)
        controls = parse_controls(prompt)
        m = analyze_rollout(res["frames"], controls)
        m["prompt"] = prompt
        rows.append(m)
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("color_acc", "shape_acc", "stability", "motion",
                     "direction_ok", "visible")}
    return {"per_case": rows, "aggregate": agg}
