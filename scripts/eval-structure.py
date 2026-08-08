#!/usr/bin/env python3
"""Gate spatial structure for curriculum Wan demos.

Pass criteria (ideal compact object):
  n_cc <= 3, top_frac >= 0.6, local_mass >= 0.4, bright% in [0.5, 18]
  (upper bright% 18 allows soft blob halos; GT toy is ~8%)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")


def load_gif(path: str) -> np.ndarray:
    im = Image.open(path)
    frames = []
    for i in range(im.n_frames):
        im.seek(i)
        im.load()
        frames.append(np.array(im.convert("RGB"), dtype=np.float32) / 255.0)
    return np.stack(frames)


def n_cc(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    sizes = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or vis[y, x]:
                continue
            stack = [(y, x)]
            vis[y, x] = True
            sz = 0
            while stack:
                cy, cx = stack.pop()
                sz += 1
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = True
                        stack.append((ny, nx))
            sizes.append(sz)
    sizes.sort(reverse=True)
    total = sum(sizes) or 1
    return len(sizes), (sizes[0] / total if sizes else 0.0)


def stats(frames: np.ndarray) -> dict:
    bright = frames.max(-1) > 0.35
    mid = bright[len(bright) // 2]
    ys, xs = np.where(mid)
    out = {
        "bright_pct": float(100.0 * bright.mean()),
        "n_cc": 0,
        "top_frac": 0.0,
        "local_mass": 0.0,
        "compact": 0.0,
        "bbox": None,
    }
    if len(ys) == 0:
        return out
    bh, bw = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    out["compact"] = float(mid.sum()) / (bh * bw)
    out["bbox"] = (int(xs.min()), int(ys.min()), bw, bh)
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = mid.shape
    rh, rw = max(4, int(0.15 * h)), max(4, int(0.15 * w))
    y0, y1 = int(max(0, cy - rh)), int(min(h, cy + rh))
    x0, x1 = int(max(0, cx - rw)), int(min(w, cx + rw))
    out["local_mass"] = float(mid[y0:y1, x0:x1].sum()) / max(1.0, float(mid.sum()))
    out["n_cc"], out["top_frac"] = n_cc(mid)
    return out


def passes(s: dict) -> bool:
    return (
        0.5 <= s["bright_pct"] <= 18.0
        and s["n_cc"] <= 3
        and s["top_frac"] >= 0.6
        and s["local_mass"] >= 0.4
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gifs", nargs="+")
    args = ap.parse_args()
    ok_all = True
    for path in args.gifs:
        s = stats(load_gif(path))
        ok = passes(s)
        ok_all = ok_all and ok
        print(f"{path}: {'PASS' if ok else 'FAIL'} {s}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
