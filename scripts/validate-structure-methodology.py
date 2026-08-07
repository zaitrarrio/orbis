#!/usr/bin/env python3
"""Validate structure-training methodology with known parameters.

Exit 0 only if all methodology checks pass (see
``doc/structure-training-methodology.md`` §5). Does **not** start training.

Usage:
  python scripts/validate-structure-methodology.py [--ckpt PATH]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from orbis.distill import _few_step
from orbis.flow import RectifiedFlow
from orbis.geometry_loss import geometry_aux, soft_dice_loss, soft_fg_energy
from orbis.world import SceneSpec, rollout


def _structure_stats(frames: np.ndarray) -> dict:
    """Same metrics as ``eval-structure.py`` (mid-frame)."""
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
    # connected components
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
    return (0.5 <= s["bright_pct"] <= 12.0 and s["n_cc"] <= 3
            and s["top_frac"] >= 0.6 and s["local_mass"] >= 0.4)


def check_gt_structure() -> tuple[bool, str]:
    msgs = []
    ok = True
    for h, w in ((64, 64), (128, 128)):
        spec = SceneSpec(shape="circle", color="red", direction="right",
                         speed="medium", size="medium")
        frames, _ = rollout(spec, 16, h, w)
        s = _structure_stats(frames.astype(np.float32))
        p = _gate_pass(s)
        ok = ok and p
        msgs.append(f"  {h}x{w}: {'PASS' if p else 'FAIL'} {s}")
    return ok, "gt_structure\n" + "\n".join(msgs)


def check_rf_identity() -> tuple[bool, str]:
    torch.manual_seed(0)
    z = torch.randn(2, 4, 8, 16, 16)
    eps = torch.randn_like(z)
    sigma = torch.tensor([0.2, 0.7])
    flow = RectifiedFlow()
    zt = flow.interpolate(z, eps, sigma)
    v = flow.target(z, eps)  # eps - z
    s = sigma.view(-1, 1, 1, 1, 1)
    z_hat = zt - s * v
    err = (z_hat - z).abs().max().item()
    ok = err < 1e-5
    return ok, f"rf_identity max|z_hat-z|={err:.2e} ({'PASS' if ok else 'FAIL'})"


def check_euler_perfect() -> tuple[bool, str]:
    torch.manual_seed(1)
    z = torch.randn(1, 4, 8, 8, 8)
    eps = torch.randn_like(z)
    # Perfect constant velocity along the RF path.
    v_star = eps - z

    def vel_fn(zz, ss):
        return v_star.expand_as(zz)

    z_end = _few_step(vel_fn, eps, steps=4)
    rel = (z_end - z).norm() / z.norm().clamp_min(1e-8)
    ok = float(rel) < 1e-4
    return ok, f"euler_perfect rel_err={float(rel):.2e} ({'PASS' if ok else 'FAIL'})"


def check_mask_loss_zero() -> tuple[bool, str]:
    torch.manual_seed(2)
    # Synthetic compact blob (known geometry).
    gt = torch.zeros(2, 3, 64, 64)
    gt[:, 0, 20:40, 20:40] = 1.0  # red square
    loss = float(geometry_aux(gt, gt))
    ok = loss < 1e-4
    return ok, f"mask_loss_zero loss={loss:.2e} ({'PASS' if ok else 'FAIL'})"


def check_mask_loss_uniform() -> tuple[bool, str]:
    torch.manual_seed(3)
    gt = torch.zeros(2, 3, 64, 64)
    gt[:, 0, 20:40, 20:40] = 1.0
    pred = torch.rand_like(gt)
    loss = float(geometry_aux(pred, gt))
    # Soft Dice on noise vs compact GT must stay clearly nonzero.
    ok = loss > 0.3
    return ok, f"mask_loss_uniform loss={loss:.3f} ({'PASS' if ok else 'FAIL'})"


def check_vae_recon_structure(ckpt: str) -> tuple[bool, str]:
    from orbis.system import OrbisSystem
    from orbis.device import get_device
    from orbis.vae import frames_to_tensor

    system = OrbisSystem.load(ckpt)
    device = get_device()
    system.to(device)
    h, w = system.cfg.world.height, system.cfg.world.width
    spec = SceneSpec(shape="circle", color="red", direction="right",
                     speed="medium", size="medium")
    frames, _ = rollout(spec, system.cfg.model.chunk_frames, h, w)
    t = frames_to_tensor(frames).to(device)  # (F,3,H,W)
    with torch.no_grad():
        z = system.vae.encode(t)
        recon = system.vae.decode(z)
    # to numpy NHWC
    arr = recon.detach().float().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    s = _structure_stats(arr)
    p = _gate_pass(s)
    return p, f"vae_recon_structure {h}x{w}: {'PASS' if p else 'FAIL'} {s}"


def check_curr_gen_baseline(ckpt: str) -> tuple[bool, str]:
    """Known-bad generator must FAIL the structure gate (gate sanity)."""
    from orbis.system import OrbisSystem
    from orbis.engine import LiveEngine

    system = OrbisSystem.load(ckpt)
    eng = LiveEngine(system)
    frames = eng.generate_video(
        "a red circle moving right", n_chunks=2, mode="t2v")["frames"]
    s = _structure_stats(np.asarray(frames, dtype=np.float32))
    fails_gate = not _gate_pass(s)
    # Methodology check passes when the gate correctly rejects the model.
    return fails_gate, (
        f"curr_gen_baseline: gate_rejects_model="
        f"{'yes' if fails_gate else 'no'} {s}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/orbis-wan-curr.pt",
                    help="Curriculum (or other) ckpt for VAE/gen checks")
    ap.add_argument("--skip-gen", action="store_true",
                    help="Skip generator baseline (CPU-only hosts)")
    args = ap.parse_args()

    checks = [
        ("gt_structure", check_gt_structure),
        ("rf_identity", check_rf_identity),
        ("euler_perfect", check_euler_perfect),
        ("mask_loss_zero", check_mask_loss_zero),
        ("mask_loss_uniform", check_mask_loss_uniform),
    ]

    results = []
    print("=== structure methodology validation ===")
    for name, fn in checks:
        ok, msg = fn()
        results.append((name, ok))
        print(("OK  " if ok else "FAIL") + " " + msg)

    # GPU / ckpt-dependent
    import os
    if os.path.isfile(args.ckpt):
        ok, msg = check_vae_recon_structure(args.ckpt)
        results.append(("vae_recon_structure", ok))
        print(("OK  " if ok else "FAIL") + " " + msg)
        if not args.skip_gen and torch.cuda.is_available():
            ok, msg = check_curr_gen_baseline(args.ckpt)
            results.append(("curr_gen_baseline", ok))
            print(("OK  " if ok else "FAIL") + " " + msg)
    else:
        print(f"SKIP vae_recon_structure / curr_gen_baseline (no ckpt {args.ckpt})")
        results.append(("vae_recon_structure", False))
        results.append(("curr_gen_baseline", False))

    failed = [n for n, ok in results if not ok]
    print("---")
    if failed:
        print(f"METHODOLOGY VALIDATION FAILED: {failed}")
        print("Do not start the next training round.")
        sys.exit(1)
    print("METHODOLOGY VALIDATION PASSED — safe to implement S0 micro train.")
    sys.exit(0)


if __name__ == "__main__":
    main()
