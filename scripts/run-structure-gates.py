#!/usr/bin/env python3
"""Train / eval loop until structure gates pass (S0 live → S1 → optional S2).

Usage (on GPU host):
  python scripts/run-structure-gates.py [--through s0|s1|s2]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from orbis.config import (
    wan_real_scale_config,
    wan_structure_curriculum_config,
    wan_structure_micro_config,
)
from orbis.data.video_dataset import MidTrainBatcher, SyntheticVideoDataset
from orbis.dataset import RolloutSampler
from orbis.device import device_name, get_device
from orbis.distill import distill
from orbis.engine import LiveEngine
from orbis.system import OrbisSystem
from orbis.train import (
    _endpoint_bootstrap_step,
    _log,
    train_generator,
    train_sr,
    train_vae,
)
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


def live_event_bootstrap(system: OrbisSystem, steps: int, batch: int,
                         lr: float = 3e-4, geometry_w: float = 2.5):
    """Endpoint bootstrap on event-aligned (post-switch) batches + text_only mix."""
    cfg = system.cfg
    device = get_device()
    system.to(device)
    gen = system.generator
    ds = SyntheticVideoDataset(cfg, seed=cfg.seed + 77)
    batcher = MidTrainBatcher(cfg, system.vae, ds, seed=cfg.seed + 78)
    sampler = RolloutSampler(cfg, system.vae, seed=cfg.seed + 79)
    params = list(
        gen.trainable_parameters() if hasattr(gen, "trainable_parameters")
        else gen.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    gen.train()
    _log(f"[live-boot] device {device_name(device)} steps={steps}", None)
    ema = None
    for step in range(steps):
        if step % 3 == 0:
            data = sampler.training_batch(batch, mode="text_only")
        else:
            data = batcher.training_batch(batch, mode="event")
        loss = _endpoint_bootstrap_step(system, data, device, geometry_w=geometry_w)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 100 == 0 or step == steps - 1:
            _log(f"[live-boot] step {step:4d}/{steps} loss {loss.item():.4f} "
                 f"ema {ema:.4f}", None)
    gen.eval()


def ensure_s0(ckpt: str, work: Path, max_rounds: int = 8) -> OrbisSystem:
    device = get_device()
    if Path(ckpt).is_file():
        system = OrbisSystem.load(ckpt).to(device)
        print(f"[s0] resumed {ckpt}")
    else:
        print("[s0] cold start micro")
        system = OrbisSystem.build(wan_structure_micro_config()).to(device)
        train_vae(system, steps=600, batch=32, lr=3e-4)
        train_generator(
            system, pretrain_steps=0, stream_steps=200, batch=16, lr=8e-4,
            pixel_aux=0.75, endpoint_aux=1.0, sigma_power=2.0, t2v_first=True,
            geometry_w=2.5, bootstrap_steps=4000)
        system.save(ckpt)

    for rnd in range(max_rounds):
        print(f"\n=== S0 eval round {rnd} ===")
        ok, so, sl = eval_pair(
            system, str(work / "micro-out.gif"), str(work / "micro-live.gif"))
        if ok:
            print("[s0] ALL GATES PASSED")
            system.save(ckpt)
            return system
        print(f"[s0] live-boot round {rnd} (800 steps)")
        live_event_bootstrap(system, steps=800, batch=16, lr=2.5e-4, geometry_w=3.0)
        try:
            distill(system, steps=150, batch=8, lr=1e-4)
        except torch.cuda.OutOfMemoryError:
            print("[s0] distill OOM skipped")
        system.save(ckpt)
    raise RuntimeError("S0 gates failed after max_rounds")


def ensure_s1(ckpt: str, work: Path, max_rounds: int = 10) -> OrbisSystem:
    device = get_device()
    print("\n=== S1 cold start 128² ===")
    cfg = wan_structure_curriculum_config()
    # Match S0 sampling: 1-step student for structure
    cfg.flow.student_steps = 1
    cfg.flow.teacher_steps = 4
    # Always cold-start S1 (do not resume speckled / black lineage).
    if Path(ckpt).is_file():
        Path(ckpt).unlink()
        print(f"[s1] removed stale {ckpt}")
    system = OrbisSystem.build(cfg).to(device)
    n_params = sum(p.numel() for p in system.generator.parameters()) / 1e6
    print(f"[s1] generator params={n_params:.1f}M dim={cfg.model.dim} "
          f"depth={cfg.model.depth}")
    train_vae(system, steps=1000, batch=24, lr=3e-4)
    # Wider stub + longer bootstrap; stream RF kept short (hurts structure).
    train_generator(
        system, pretrain_steps=0, stream_steps=100, batch=6, lr=6e-4,
        pixel_aux=0.85, endpoint_aux=1.25, sigma_power=2.0, t2v_first=True,
        geometry_w=3.0, bootstrap_steps=6000)
    system.save(ckpt)

    for rnd in range(max_rounds):
        print(f"\n=== S1 eval round {rnd} ===")
        ok, so, sl = eval_pair(
            system, str(work / "s1-out.gif"), str(work / "s1-live.gif"))
        if ok:
            print("[s1] ALL GATES PASSED")
            try:
                train_sr(system, steps=100, batch=8)
            except Exception as e:
                print(f"[s1] sr skip {e}")
            system.save(ckpt)
            return system
        out_ok = _gate_pass(so)
        # Live-boot collapses structure when out is already black/speckled —
        # recover with endpoint bootstrap first; only live-boot once out passes.
        if not out_ok:
            print(f"[s1] structure-boot round {rnd} (out failed)")
            train_generator(
                system, pretrain_steps=0, stream_steps=0, batch=8, lr=5e-4,
                pixel_aux=0.85, endpoint_aux=1.25, sigma_power=2.0,
                t2v_first=True, geometry_w=3.5, bootstrap_steps=1500)
        else:
            print(f"[s1] live-boot round {rnd} (out ok, live failed)")
            live_event_bootstrap(
                system, steps=500, batch=8, lr=1.5e-4, geometry_w=3.0)
            try:
                distill(system, steps=100, batch=4, lr=8e-5)
            except torch.cuda.OutOfMemoryError:
                print("[s1] distill OOM skipped")
        system.save(ckpt)
    raise RuntimeError("S1 gates failed after max_rounds")


def ensure_s2(ckpt: str, work: Path, max_rounds: int = 10,
              resume: bool = False) -> OrbisSystem:
    device = get_device()
    vram = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if torch.cuda.is_available() else 0)
    print(f"\n=== S2 480p (vram≈{vram:.0f}GB) resume={resume} ===")
    if vram < 20:
        raise RuntimeError("S2 needs >=20GB VRAM")
    gen_bs = 2 if vram >= 60 else 1

    if resume and Path(ckpt).is_file():
        print(f"[s2] resumed {ckpt}")
        system = OrbisSystem.load(ckpt).to(device)
        # Keep 1-step student for structure gates.
        system.cfg.flow.student_steps = 1
        system.cfg.flow.teacher_steps = 4
    else:
        if Path(ckpt).is_file():
            Path(ckpt).unlink()
            print(f"[s2] removed stale {ckpt}")
        cfg = wan_real_scale_config(stub=True)
        cfg.flow.student_steps = 1
        cfg.flow.teacher_steps = 4
        system = OrbisSystem.build(cfg).to(device)
        train_vae(system, steps=1200, batch=4 if vram < 60 else 8, lr=3e-4)
        train_generator(
            system, pretrain_steps=0, stream_steps=200, batch=gen_bs, lr=3e-4,
            pixel_aux=0.75, endpoint_aux=1.0, sigma_power=2.0, t2v_first=True,
            geometry_w=2.5, bootstrap_steps=3000)
        system.save(ckpt)

    for rnd in range(max_rounds):
        print(f"\n=== S2 eval round {rnd} ===")
        ok, so, sl = eval_pair(
            system, str(work / "s2-out.gif"), str(work / "s2-live.gif"),
            chunks=4)
        if ok:
            print("[s2] ALL GATES PASSED")
            system.save(ckpt)
            return system
        out_ok = _gate_pass(so)
        oversized = so["bright_pct"] > 18.0 or so["local_mass"] < 0.4
        speckled = so["n_cc"] > 3 or so["top_frac"] < 0.6
        if not out_ok and (oversized or speckled):
            # Shrink / compact: strong geometry + asymmetric size prior in boot.
            print(f"[s2] shrink-boot round {rnd} bright={so['bright_pct']:.1f} "
                  f"local={so['local_mass']:.2f} n_cc={so['n_cc']}")
            train_generator(
                system, pretrain_steps=0, stream_steps=0, batch=gen_bs, lr=2e-4,
                pixel_aux=0.9, endpoint_aux=1.25, sigma_power=2.0,
                t2v_first=True, geometry_w=5.0, bootstrap_steps=1200,
                shrink=True)
        elif not out_ok:
            print(f"[s2] structure-boot round {rnd}")
            train_generator(
                system, pretrain_steps=0, stream_steps=0, batch=gen_bs, lr=2e-4,
                pixel_aux=0.75, endpoint_aux=1.0, sigma_power=2.0,
                t2v_first=True, geometry_w=3.0, bootstrap_steps=800)
        else:
            print(f"[s2] live-boot round {rnd}")
            live_event_bootstrap(system, steps=300, batch=gen_bs, lr=8e-5,
                                 geometry_w=2.5)
        system.save(ckpt)
    raise RuntimeError("S2 gates failed after max_rounds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", choices=("s0", "s1", "s2"), default="s2")
    ap.add_argument("--from-stage", choices=("s0", "s1", "s2"), default="s0",
                    help="Skip stages before this one (assumes prior ckpts pass).")
    ap.add_argument("--resume-s2", action="store_true",
                    help="Continue S2 from existing orbis-wan-real.pt")
    ap.add_argument("--work", default="/workspace")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[gates] device {device_name(get_device())} through={args.through} "
          f"from={args.from_stage}")

    stage_order = ("s0", "s1", "s2")
    start = stage_order.index(args.from_stage)
    end = stage_order.index(args.through)
    if end < start:
        raise SystemExit("--through must be >= --from-stage")

    if start <= 0 <= end:
        ensure_s0(str(work / "orbis-wan-micro.pt"), work)
    if start <= 1 <= end:
        ensure_s1(str(work / "orbis-wan-curr.pt"), work)
    if start <= 2 <= end:
        ensure_s2(str(work / "orbis-wan-real.pt"), work, resume=args.resume_s2)
        print(f"[gates] ALL STAGES PASSED in {time.time()-t0:.1f}s")
        return
    print(f"[gates] done through={args.through} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
