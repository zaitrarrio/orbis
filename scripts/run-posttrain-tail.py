#!/usr/bin/env python3
"""GRPO + distill + structure re-gate (for ≥80GB, e.g. H100).

Intended after guidance / EMA / DMD already ran (or to re-run those tails).

Usage:
  python scripts/run-posttrain-tail.py \\
    --ckpt /workspace/orbis-wan-real-v2.pt \\
    --out /workspace/orbis-wan-real-v3.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from orbis.data.video_dataset import MidTrainBatcher, SyntheticVideoDataset
from orbis.dataset import RolloutSampler
from orbis.device import device_name, get_device
from orbis.distill import distill
from orbis.engine import LiveEngine
from orbis.posttrain import grpo_align
from orbis.system import OrbisSystem
from orbis import media


def _structure_stats(frames: np.ndarray) -> dict:
    bright = frames.max(-1) > 0.35
    mid = bright[len(bright) // 2]
    ys, xs = np.where(mid)
    out = {"bright_pct": float(100 * bright.mean()), "n_cc": 0,
           "top_frac": 0.0, "local_mass": 0.0}
    if len(ys) == 0:
        return out
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = mid.shape
    rh, rw = max(4, int(0.15 * h)), max(4, int(0.15 * w))
    y0, y1 = int(max(0, cy - rh)), int(min(h, cy + rh))
    x0, x1 = int(max(0, cx - rw)), int(min(w, cx + rw))
    out["local_mass"] = float(mid[y0:y1, x0:x1].sum()) / max(1.0, float(mid.sum()))
    vis = np.zeros_like(mid, dtype=bool)
    sizes = []
    for y in range(h):
        for x in range(w):
            if not mid[y, x] or vis[y, x]:
                continue
            stack = [(y, x)]
            vis[y, x] = True
            sz = 0
            while stack:
                cy_, cx_ = stack.pop()
                sz += 1
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = cy_ + dy, cx_ + dx
                    if 0 <= ny < h and 0 <= nx < w and mid[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = True
                        stack.append((ny, nx))
            sizes.append(sz)
    sizes.sort(reverse=True)
    total = sum(sizes) or 1
    out["n_cc"] = len(sizes)
    out["top_frac"] = sizes[0] / total if sizes else 0.0
    return out


def _gate_pass(s: dict) -> bool:
    return (0.5 <= s["bright_pct"] <= 18.0 and s["n_cc"] <= 3
            and s["top_frac"] >= 0.6 and s["local_mass"] >= 0.4)


def eval_pair(system: OrbisSystem, out_gif: str, live_gif: str,
              chunks: int = 4) -> tuple[bool, dict, dict]:
    eng = LiveEngine(system)
    frames = list(eng.generate_video(
        "a red circle moving right", n_chunks=chunks, mode="t2v")["frames"])
    media.save_gif(out_gif, frames, fps=eng.cfg.world.fps, scale=1)
    s = eng.start("a red circle moving right", mode="t2v")
    all_frames = []
    for k in range(chunks):
        if k == 2:
            s.set_prompt("a blue square moving up")
        all_frames.append(eng.generate_chunk(s).frames)
    media.save_gif(live_gif, list(np.concatenate(all_frames, 0)),
                   fps=eng.cfg.world.fps, scale=1)
    so = _structure_stats(np.asarray(frames, np.float32))
    sl = _structure_stats(np.asarray(np.concatenate(all_frames, 0), np.float32))
    ok = _gate_pass(so) and _gate_pass(sl)
    print(f"  out:  {'PASS' if _gate_pass(so) else 'FAIL'} {so}")
    print(f"  live: {'PASS' if _gate_pass(sl) else 'FAIL'} {sl}")
    return ok, so, sl


def _batch_fn(system: OrbisSystem):
    cfg = system.cfg
    ds = SyntheticVideoDataset(cfg, seed=cfg.seed + 10)
    batcher = MidTrainBatcher(cfg, system.vae, ds, seed=cfg.seed + 11)
    sampler = RolloutSampler(cfg, system.vae, seed=cfg.seed + 12)

    def batch_fn(mode: str = "history"):
        if mode == "event":
            return batcher.training_batch(1, mode="event")
        return sampler.training_batch(
            1, mode=mode if mode in ("history", "reference", "text_only")
            else "history",
            memory_context_chunks=1 if mode == "history" else 0)

    return batch_fn


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/workspace")
    ap.add_argument("--grpo-steps", type=int, default=50)
    ap.add_argument("--distill-steps", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--student-steps", type=int, default=4)
    ap.add_argument("--teacher-steps", type=int, default=16)
    ap.add_argument("--skip-grpo", action="store_true")
    ap.add_argument("--skip-distill", action="store_true")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    device = get_device()
    t0 = time.time()
    print(f"[tail] device {device_name(device)}")
    system = OrbisSystem.load(args.ckpt).to(device)
    system.cfg.flow.student_steps = args.student_steps
    system.cfg.flow.teacher_steps = args.teacher_steps
    print(f"[tail] loaded {args.ckpt} student_steps={args.student_steps}")

    print("\n=== baseline gate ===")
    ok0, _, _ = eval_pair(
        system, str(work / "tail-baseline-out.gif"),
        str(work / "tail-baseline-live.gif"))
    print(f"[tail] baseline {'PASS' if ok0 else 'FAIL'}")

    batch_fn = _batch_fn(system)

    if not args.skip_grpo:
        print(f"\n=== GRPO steps={args.grpo_steps} G={args.group_size} ===")
        grpo_align(system, lambda: batch_fn("history"),
                   steps=args.grpo_steps, group_size=args.group_size)
        _gc()
        system.save(args.out)
        print(f"[tail] saved after GRPO → {args.out}")

    if not args.skip_distill:
        print(f"\n=== distill steps={args.distill_steps} ===")
        distill(system, steps=args.distill_steps, batch=1, lr=8e-5)
        _gc()
        system.save(args.out)

    print("\n=== gate after tail ===")
    ok, _, _ = eval_pair(
        system, str(work / "tail-out.gif"), str(work / "tail-live.gif"))
    system.save(args.out)
    if not ok:
        raise RuntimeError(f"structure gate failed after tail — saved {args.out}")
    print(f"[tail] ALL PASSED in {time.time()-t0:.1f}s → {args.out}")


if __name__ == "__main__":
    main()
