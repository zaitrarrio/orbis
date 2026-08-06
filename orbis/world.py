"""The toy video world -- Orbis's ground-truth data-generating process.

A *scene* is a single moving shape (subject) with a color (attribute) travelling
in a direction (motion) over a dark background.  A *rollout* advances the shape
frame by frame, bouncing off the frame edges, producing a video whose content is
fully determined by the prompt plus an initial position.  This gives us:

* a learnable text->video mapping (prompt controls shape/color/motion);
* genuine temporal state that must be carried across chunks (position/velocity),
  which is what the bounded memory is there to preserve;
* a visible response to *mid-rollout prompt switching* -- changing the color or
  direction of an in-progress rollout changes future frames while the object
  keeps moving from where it was.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .vocab import COLORS, DIRECTIONS, SHAPES, SPEEDS, SIZES, BACKGROUND


BASE_SPEED = 0.16   # normalized units per frame at "medium" speed
BASE_RADIUS = 0.16  # normalized radius at "medium" size


@dataclass
class SceneSpec:
    """Fully-specified state of a toy scene at one instant."""

    shape: str = "circle"
    color: str = "red"
    direction: str = "right"
    speed: str = "medium"
    size: str = "medium"
    x: float = 0.5
    y: float = 0.5

    def control_prompt(self) -> str:
        """Canonical English prompt describing this scene."""
        mods = []
        if self.size != "medium":
            mods.append(self.size)
        mods.append(self.color)
        mods.append(self.shape)
        s = "a " + " ".join(mods) + f" moving {self.direction}"
        if self.speed != "medium":
            s += f" {self.speed}"
        return s

    @property
    def velocity(self) -> np.ndarray:
        vx, vy = DIRECTIONS[self.direction]
        return np.array([vx, vy], dtype=np.float32) * BASE_SPEED * SPEEDS[self.speed]

    @property
    def radius(self) -> float:
        return BASE_RADIUS * SIZES[self.size]


def sample_scene(rng: np.random.Generator) -> SceneSpec:
    """Sample a random scene with a random starting position."""
    return SceneSpec(
        shape=rng.choice(SHAPES),
        color=rng.choice(list(COLORS.keys())),
        direction=rng.choice(list(DIRECTIONS.keys())),
        speed=rng.choice(list(SPEEDS.keys()), p=[0.25, 0.5, 0.25]),
        size=rng.choice(list(SIZES.keys()), p=[0.25, 0.5, 0.25]),
        x=float(rng.uniform(0.25, 0.75)),
        y=float(rng.uniform(0.25, 0.75)),
    )


def _step(spec: SceneSpec) -> SceneSpec:
    """Advance the scene one frame with wall bouncing."""
    v = spec.velocity
    x, y = spec.x + v[0], spec.y + v[1]
    direction = spec.direction
    vx, vy = DIRECTIONS[direction]
    # Reflect off the walls, keeping a margin so the shape stays visible.
    m = spec.radius
    if x < m:
        x, vx = m, abs(vx)
    elif x > 1 - m:
        x, vx = 1 - m, -abs(vx)
    if y < m:
        y, vy = m, abs(vy)
    elif y > 1 - m:
        y, vy = 1 - m, -abs(vy)
    # Update direction only if a component actually flipped, snapping back onto
    # the nearest canonical direction so the control stays interpretable.
    new_dir = _nearest_direction(vx, vy, fallback=direction)
    return replace(spec, x=float(x), y=float(y), direction=new_dir)


def _nearest_direction(vx: float, vy: float, fallback: str) -> str:
    best, best_dot = fallback, -1e9
    n = (vx * vx + vy * vy) ** 0.5 or 1.0
    vx, vy = vx / n, vy / n
    for name, (dx, dy) in DIRECTIONS.items():
        dn = (dx * dx + dy * dy) ** 0.5
        dot = (vx * dx + vy * dy) / dn
        if dot > best_dot:
            best, best_dot = name, dot
    return best


def render_frame(spec: SceneSpec, height: int, width: int) -> np.ndarray:
    """Render a single ``(H, W, 3)`` float32 frame in ``[0, 1]``.

    Shapes have softened (anti-aliased) edges so the target is smooth and easy
    for a small model to fit.
    """
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    dx, dy = gx - spec.x, gy - spec.y
    r = spec.radius
    aa = 1.5 / max(height, width)  # anti-alias band width

    if spec.shape == "circle":
        dist = np.sqrt(dx * dx + dy * dy)
        mask = _smoothstep(r + aa, r - aa, dist)
    elif spec.shape == "square":
        cheb = np.maximum(np.abs(dx), np.abs(dy))
        mask = _smoothstep(r + aa, r - aa, cheb)
    else:  # triangle pointing up
        # Signed distance-ish: inside if below the two slanted top edges and
        # above the base.  Uses a normalized barycentric test.
        h = 1.7 * r
        top = spec.y - h * 0.6
        base = spec.y + h * 0.4
        half = r * 1.15
        # horizontal half-width grows linearly from apex (0) to base (half)
        frac = np.clip((gy - top) / (base - top + 1e-6), 0.0, 1.0)
        width_at = frac * half
        inside = (gy >= top) & (gy <= base) & (np.abs(dx) <= width_at)
        mask = inside.astype(np.float32)
        # soften edges
        edge = np.abs(np.abs(dx) - width_at)
        soft = _smoothstep(aa, 0.0, edge) * ((gy >= top) & (gy <= base))
        mask = np.maximum(mask, soft)

    mask = mask[..., None]
    bg = np.array(BACKGROUND, dtype=np.float32)
    fg = np.array(COLORS[spec.color], dtype=np.float32)
    frame = bg[None, None, :] * (1 - mask) + fg[None, None, :] * mask
    return frame.astype(np.float32)


def _smoothstep(hi: float, lo: float, x: np.ndarray) -> np.ndarray:
    """Smooth 1->0 transition as x goes from ``lo`` to ``hi`` (hi>lo)."""
    t = np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return (1 - (t * t * (3 - 2 * t))).astype(np.float32)


def rollout(spec: SceneSpec, n_frames: int, height: int, width: int,
            control_schedule: Optional[dict] = None):
    """Render ``n_frames`` frames, returning ``(frames, specs)``.

    ``control_schedule`` optionally maps a frame index -> partial control dict
    (e.g. ``{20: {"color": "blue"}}``) to model a prompt switch: the attribute
    changes at that frame while position/velocity continue.
    """
    control_schedule = control_schedule or {}
    frames = np.empty((n_frames, height, width, 3), dtype=np.float32)
    specs = []
    cur = spec
    for i in range(n_frames):
        if i in control_schedule:
            cur = replace(cur, **control_schedule[i])
        frames[i] = render_frame(cur, height, width)
        specs.append(cur)
        cur = _step(cur)
    return frames, specs
