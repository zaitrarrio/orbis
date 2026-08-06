"""Train the full Wan live methodology pipeline.

Usage:
  python scripts/train-live-wan.py [out_path] [scale] [--smoke] [--data DIR]

Stages:
  0 VAE
  1 Streaming adaptation (history / reference mix)
  2 Single/multi-event mid-training
  3 Guidance distillation
  4 EMA consistency distillation
  5 Self-forcing DMD
  6 GRPO + world-model consistency
  7 Streaming SR

Default uses ``wan_real_scale_config`` (480x832). Pass ``--smoke`` for the
small Wan stub config suitable for CI. Official Wan2.1 weights are optional
(``backbone.wan_stub=True`` trains the structural twin + LoRA methodology).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from orbis.config import wan_real_scale_config, wan_smoke_config
from orbis.data.video_dataset import MidTrainBatcher, SyntheticVideoDataset, VideoClipDataset
from orbis.dataset import RolloutSampler
from orbis.device import device_name, get_device
from orbis.distill import distill
from orbis.posttrain import (
    ema_consistency_distill,
    grpo_align,
    guidance_distill,
    self_forcing_dmd,
)
from orbis.system import OrbisSystem
from orbis.train import train_generator, train_sr, train_vae, _log


def _make_batch_fn(system, data_root: str | None):
    cfg = system.cfg
    if data_root:
        ds = VideoClipDataset(data_root, cfg, seed=cfg.seed + 10)
    else:
        ds = SyntheticVideoDataset(cfg, seed=cfg.seed + 10)
    batcher = MidTrainBatcher(cfg, system.vae, ds, seed=cfg.seed + 11)
    sampler = RolloutSampler(cfg, system.vae, seed=cfg.seed + 12)

    def batch_fn(mode: str = "history"):
        if mode == "event":
            return batcher.training_batch(4, mode="event")
        # Prefer toy RolloutSampler when geometry matches toy world speeds
        try:
            return sampler.training_batch(
                4, mode=mode if mode in ("history", "reference", "text_only")
                else "history",
                memory_context_chunks=1 if mode == "history" else 0)
        except Exception:
            return batcher.training_batch(4, mode="history")

    return batch_fn


def mid_train_events(system: OrbisSystem, steps: int, batch_fn, log_cb=None):
    """Fine-tune on mid-rollout condition changes (event-aligned)."""
    from orbis.flow import RectifiedFlow
    from orbis.train import _flow_step

    cfg = system.cfg
    device = get_device()
    system.to(device)
    gen = system.generator
    flow = RectifiedFlow(cfg.flow.train_sigma_eps)
    params = list(
        gen.trainable_parameters() if hasattr(gen, "trainable_parameters")
        else gen.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    gen.train()
    _log(f"[mid-train] device {device_name(device)} steps={steps}", log_cb)
    for step in range(steps):
        data = batch_fn("event")
        loss = _flow_step(system, flow, data, device)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            _log(f"[mid-train] step {step:4d}/{steps} loss {loss.item():.5f}",
                 log_cb)
    gen.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="orbis-wan.pt")
    ap.add_argument("scale", nargs="?", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true",
                    help="Use tiny Wan stub config for CI")
    ap.add_argument("--data", default=None,
                    help="Directory with manifest.jsonl / video clips")
    ap.add_argument("--load-hf", action="store_true",
                    help="Attempt to load official Wan2.1 Diffusers weights")
    args = ap.parse_args()

    device = get_device()
    print(f"[wan-live] device {device_name(device)}")
    if device.type == "cpu":
        torch.set_num_threads(4)

    cfg = wan_smoke_config() if args.smoke else wan_real_scale_config(
        stub=not args.load_hf)
    if args.load_hf:
        cfg.backbone.wan_stub = False

    s = lambda n: max(4, int(n * args.scale))
    t0 = time.time()
    system = OrbisSystem.build(cfg).to(device)
    if args.load_hf and hasattr(system.generator, "try_load_hf"):
        ok = system.generator.try_load_hf(cfg.backbone.checkpoint_path)
        print(f"[wan-live] HF weights loaded={ok}")

    batch_fn = _make_batch_fn(system, args.data)

    train_vae(system, steps=s(200 if args.smoke else 900))
    # Streaming adaptation
    train_generator(
        system,
        pretrain_steps=s(100 if args.smoke else 700),
        stream_steps=s(200 if args.smoke else 1800),
    )
    mid_train_events(system, steps=s(50 if args.smoke else 400),
                     batch_fn=batch_fn)
    guidance_distill(system, lambda: batch_fn("history"),
                     steps=s(40 if args.smoke else 200))
    ema_consistency_distill(system, lambda: batch_fn("history"),
                            steps=s(40 if args.smoke else 200))
    self_forcing_dmd(system, lambda: batch_fn("history"),
                     steps=s(40 if args.smoke else 200),
                     force_chunks=1 if args.smoke else 2)
    grpo_align(system, lambda: batch_fn("history"),
               steps=s(20 if args.smoke else 100),
               group_size=2 if args.smoke else 4)
    train_sr(system, steps=s(50 if args.smoke else 300))

    system.save(args.out)
    print(f"[wan-live] saved {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
