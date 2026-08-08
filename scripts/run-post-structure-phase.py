#!/usr/bin/env python3
"""Phase after structure gates: multi-step student → post-train → re-gate.

Starts from a structure-gated S2 checkpoint (1-step student). Sequence:
  1. Raise ``student_steps`` to 4 (methodology default) and finetune
  2. Structure-gate out + live (must stay green)
  3. Guidance → EMA consistency → self-forcing DMD → GRPO
  4. Structure-gate again; save if still green

Usage (GPU host):
  python scripts/run-post-structure-phase.py \\
    --ckpt /workspace/orbis-wan-real.pt \\
    --out /workspace/orbis-wan-real-v2.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from orbis.device import device_name, get_device
from orbis.distill import distill
from orbis.engine import LiveEngine
from orbis.posttrain import (
    ema_consistency_distill,
    grpo_align,
    guidance_distill,
    self_forcing_dmd,
)
from orbis.system import OrbisSystem
from orbis.train import train_generator, _log
from orbis import media
from orbis.data.video_dataset import MidTrainBatcher, SyntheticVideoDataset
from orbis.dataset import RolloutSampler


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
    bs = 1

    def batch_fn(mode: str = "history"):
        if mode == "event":
            return batcher.training_batch(bs, mode="event")
        return sampler.training_batch(
            bs, mode=mode if mode in ("history", "reference", "text_only")
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
    ap.add_argument("--ckpt", default="/workspace/orbis-wan-real.pt")
    ap.add_argument("--out", default="/workspace/orbis-wan-real-v2.pt")
    ap.add_argument("--work", default="/workspace")
    ap.add_argument("--student-steps", type=int, default=4)
    ap.add_argument("--teacher-steps", type=int, default=16)
    ap.add_argument("--multistep-boot", type=int, default=1200)
    ap.add_argument("--multistep-stream", type=int, default=400)
    ap.add_argument("--skip-posttrain", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Scale post-train step counts")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    s = lambda n: max(4, int(n * args.scale))

    device = get_device()
    t0 = time.time()
    print(f"[phase2] device {device_name(device)}")
    system = OrbisSystem.load(args.ckpt).to(device)
    system.cfg.flow.student_steps = args.student_steps
    system.cfg.flow.teacher_steps = args.teacher_steps
    print(f"[phase2] loaded {args.ckpt} student_steps={args.student_steps} "
          f"teacher_steps={args.teacher_steps}")

    print("\n=== baseline gate (1-step weights @ N-step decode) ===")
    ok0, _, _ = eval_pair(
        system, str(work / "phase2-baseline-out.gif"),
        str(work / "phase2-baseline-live.gif"))
    print(f"[phase2] baseline {'PASS' if ok0 else 'FAIL'} "
          f"(expected often FAIL until multistep finetune)")

    print("\n=== multi-step structure finetune ===")
    # Full unrolled few-step grads on bootstrap. Skip RF stream here: combining
    # RF + 4-step endpoint decode OOMs even on 32GB 5090 (~28GB reserved).
    train_generator(
        system, pretrain_steps=0, stream_steps=0,
        batch=1, lr=1.5e-4, pixel_aux=0.75, endpoint_aux=1.0,
        sigma_power=2.0, t2v_first=True, geometry_w=2.5,
        bootstrap_steps=args.multistep_boot, shrink=False)
    _gc()
    system.save(args.out)

    print("\n=== gate after multi-step ===")
    ok1, _, _ = eval_pair(
        system, str(work / "phase2-ms-out.gif"),
        str(work / "phase2-ms-live.gif"))
    if not ok1:
        # Compact recovery before abandoning post-train.
        print("[phase2] multi-step gate failed — shrink recovery")
        train_generator(
            system, pretrain_steps=0, stream_steps=0, batch=1, lr=1.2e-4,
            pixel_aux=0.75, endpoint_aux=1.0, sigma_power=2.0,
            t2v_first=True, geometry_w=3.0, bootstrap_steps=800,
            shrink=True)
        _gc()
        ok1, _, _ = eval_pair(
            system, str(work / "phase2-ms-out.gif"),
            str(work / "phase2-ms-live.gif"))
        system.save(args.out)
    if not ok1:
        raise RuntimeError("structure gate failed after multi-step finetune")
    print("[phase2] multi-step gate PASS")

    if args.skip_posttrain:
        print(f"[phase2] done (no post-train) in {time.time()-t0:.1f}s → {args.out}")
        return

    batch_fn = _batch_fn(system)
    print("\n=== post-train: guidance ===")
    try:
        guidance_distill(system, lambda: batch_fn("history"), steps=s(200))
    except torch.cuda.OutOfMemoryError as e:
        print(f"[phase2] guidance OOM ({e}); skip")
    _gc()

    print("\n=== post-train: EMA consistency ===")
    try:
        ema_consistency_distill(system, lambda: batch_fn("history"), steps=s(200))
    except torch.cuda.OutOfMemoryError as e:
        print(f"[phase2] ema OOM ({e}); skip")
    _gc()

    print("\n=== post-train: self-forcing DMD ===")
    try:
        self_forcing_dmd(system, lambda: batch_fn("history"),
                         steps=s(200), force_chunks=1)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[phase2] dmd OOM ({e}); skip")
    _gc()

    print("\n=== post-train: GRPO ===")
    try:
        grpo_align(system, lambda: batch_fn("history"),
                   steps=s(50), group_size=2)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[phase2] grpo OOM ({e}); skip")
    _gc()

    try:
        distill(system, steps=s(200), batch=1, lr=8e-5)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[phase2] distill OOM ({e}); skip")
    _gc()

    print("\n=== gate after post-train ===")
    ok2, _, _ = eval_pair(
        system, str(work / "phase2-pt-out.gif"),
        str(work / "phase2-pt-live.gif"))
    system.save(args.out)
    if not ok2:
        raise RuntimeError(
            "structure gate failed after post-train — weights saved for debug "
            f"at {args.out}")
    print(f"[phase2] ALL PASSED in {time.time()-t0:.1f}s → {args.out}")


if __name__ == "__main__":
    main()
