"""Train the full Wan live methodology pipeline.

Usage:
  python scripts/train-live-wan.py [out_path] [scale] [--smoke|--curriculum] [--data DIR]

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
tiny CI stub, or ``--curriculum`` for the 128² structure-first path.
Official Wan2.1 weights are optional (``backbone.wan_stub=True`` trains the
structural twin + LoRA methodology).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from orbis.config import (
    wan21_real_config,
    wan_real_scale_config,
    wan_smoke_config,
    wan_structure_curriculum_config,
    wan_structure_micro_config,
)
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

    # Micro-batch: smoke can afford 4; real-scale latents need 1 on 24GB.
    bs = 4 if cfg.world.height <= 64 else 1

    def batch_fn(mode: str = "history"):
        if mode == "event":
            return batcher.training_batch(bs, mode="event")
        # Prefer toy RolloutSampler when geometry matches toy world speeds
        try:
            return sampler.training_batch(
                bs, mode=mode if mode in ("history", "reference", "text_only")
                else "history",
                memory_context_chunks=1 if mode == "history" else 0)
        except Exception:
            return batcher.training_batch(bs, mode="history")

    return batch_fn


def mid_train_events(system: OrbisSystem, steps: int, batch_fn, log_cb=None,
                     endpoint_aux: float = 0.0):
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
        loss = _flow_step(system, flow, data, device,
                          endpoint_aux=endpoint_aux, sigma_power=2.0,
                          pixel_aux=0.35)
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
    ap.add_argument("--curriculum", action="store_true",
                    help="128² structure-first Wan stub (endpoint loss from step 0)")
    ap.add_argument("--micro", action="store_true",
                    help="64² S0 structure proof (mask/centroid + endpoint; see methodology)")
    ap.add_argument("--data", default=None,
                    help="Directory with manifest.jsonl / video clips")
    ap.add_argument("--load-hf", action="store_true",
                    help="Attempt to load official Wan2.1 Diffusers weights")
    ap.add_argument("--real-wan", action="store_true",
                    help="Drive the REAL pretrained Wan2.1-1.3B transformer "
                    "frozen + LoRA (orbis.adapters.wan21_real.RealWanBackbone) "
                    "instead of the structural Wan-scale stub. Requires "
                    "`uv sync --extra wan` and, beyond CPU shape tests, a real "
                    "GPU -- see deploy/README.md. Independent of --load-hf, "
                    "which only affects the old stub's (no-op) weight-mapping "
                    "attempt.")
    ap.add_argument("--wan-checkpoint", default=None,
                    help="HF repo/path for --real-wan "
                    "(default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)")
    ap.add_argument("--wan-text-encoder", default=None,
                    help="HF repo/path for the UMT5 tokenizer/text encoder "
                    "used by --real-wan (default: same repo as --wan-checkpoint)")
    ap.add_argument("--no-posttrain", action="store_true",
                    help="Skip guidance/EMA/DMD/GRPO (faster; use until shapes work)")
    ap.add_argument("--resume", default=None,
                    help="Resume/finetune from checkpoint (skips VAE train)")
    ap.add_argument("--finetune-steps", type=int, default=0,
                    help="If --resume, run this many stream steps (0=scale defaults)")
    args = ap.parse_args()

    device = get_device()
    print(f"[wan-live] device {device_name(device)}")
    if device.type == "cpu":
        torch.set_num_threads(4)

    if args.smoke:
        cfg = wan_smoke_config()
    elif args.micro:
        cfg = wan_structure_micro_config()
    elif args.curriculum:
        cfg = wan_structure_curriculum_config()
    elif args.real_wan:
        cfg = wan21_real_config(checkpoint_path=args.wan_checkpoint,
                                 text_encoder_path=args.wan_text_encoder)
    else:
        cfg = wan_real_scale_config(stub=not args.load_hf)
    if args.load_hf:
        cfg.backbone.wan_stub = False

    curriculum = bool(args.curriculum)
    micro = bool(args.micro)
    structure_stage = curriculum or micro
    s = lambda n: max(4, int(n * args.scale))
    t0 = time.time()
    if args.resume:
        system = OrbisSystem.load(args.resume).to(device)
        print(f"[wan-live] resumed {args.resume}")
    else:
        system = OrbisSystem.build(cfg).to(device)
        if args.load_hf and hasattr(system.generator, "try_load_hf"):
            ok = system.generator.try_load_hf(cfg.backbone.checkpoint_path)
            print(f"[wan-live] HF weights loaded={ok}")

    if structure_stage:
        print(f"[wan-live] {'micro' if micro else 'curriculum'} "
              f"{cfg.world.height}x{cfg.world.width} "
              f"patch={cfg.model.patch_size} ds={cfg.vae.downsample}")

    batch_fn = _make_batch_fn(system, args.data)
    import torch as _torch
    vram_gb = (_torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
               if _torch.cuda.is_available() else 0)
    h = cfg.world.height
    if args.smoke:
        vae_bs, gen_bs, sr_bs, force_chunks = 16, 16, 8, 1
    elif micro:
        vae_bs, gen_bs, sr_bs, force_chunks = 32, 16, 8, 2
    elif curriculum or h <= 128:
        vae_bs = 32 if vram_gb >= 20 else 16
        gen_bs = 8 if vram_gb >= 20 else 4
        sr_bs = 8
        force_chunks = 2
    else:
        big = vram_gb >= 60
        vae_bs = 8 if big else 4
        gen_bs = 2 if big else 1
        sr_bs = 2 if big else 1
        force_chunks = 2 if big else 1
    print(f"[wan-live] vram≈{vram_gb:.0f}GB batches vae={vae_bs} gen={gen_bs} "
          f"sr={sr_bs} force_chunks={force_chunks}")

    def _gc():
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.resume:
        if args.smoke:
            vae_steps = 200
        elif micro:
            vae_steps = 600
        elif curriculum:
            vae_steps = 800
        else:
            vae_steps = 2500
        train_vae(system, steps=s(vae_steps), batch=vae_bs,
                  lr=3e-4 if not args.smoke else 1e-3)
        _gc()

    if args.resume and args.finetune_steps > 0:
        pre_steps, stream_steps = 0, args.finetune_steps
        gen_lr = 2e-4
    elif micro:
        pre_steps, stream_steps = s(500), s(1200)
        gen_lr = 5e-4
    elif curriculum:
        pre_steps, stream_steps = s(400), s(1600)
        gen_lr = 4e-4
    else:
        pre_steps = s(100 if args.smoke else 700)
        stream_steps = s(200 if args.smoke else 1800)
        gen_lr = 6e-4 if args.smoke else 3e-4

    use_structure_obj = structure_stage or bool(args.resume) or not args.smoke
    if args.smoke:
        endpoint_aux = 0.0
    elif micro:
        # S0.1: stronger endpoint to clear satellites / over-bright BG.
        endpoint_aux = 1.0
    elif curriculum:
        endpoint_aux = 0.55
    elif args.resume:
        endpoint_aux = 0.5
    else:
        endpoint_aux = 0.35
    geometry_w = 2.5 if micro else (0.5 if curriculum else 0.0)
    pixel_aux = 0.75 if micro else (0.5 if not args.smoke else 0.2)
    if args.resume and args.finetune_steps > 0 and micro:
        gen_lr = 1.5e-4  # stable cleanup finetune from S0 blob prior
    # S0: pure few-step→GT bootstrap (primary structure teacher).
    bootstrap_steps = s(4000) if micro else 0
    if args.resume and micro and args.finetune_steps > 0:
        bootstrap_steps = args.finetune_steps
        pre_steps, stream_steps = 0, max(200, args.finetune_steps // 4)
    elif micro and not args.resume:
        # Cold start: bootstrap only + tiny RF polish (wash came from RF mix).
        pre_steps, stream_steps = 0, s(200)
        gen_lr = 8e-4
    train_generator(
        system,
        pretrain_steps=pre_steps,
        stream_steps=stream_steps,
        batch=gen_bs,
        lr=gen_lr,
        pixel_aux=pixel_aux,
        endpoint_aux=endpoint_aux,
        sigma_power=2.0 if use_structure_obj else 1.0,
        t2v_first=use_structure_obj,
        geometry_w=geometry_w,
        bootstrap_steps=bootstrap_steps,
    )
    _gc()
    mid_steps = (100 if args.smoke else
                 (200 if micro else
                  (300 if curriculum else (200 if args.resume else 400))))
    mid_train_events(system, steps=s(mid_steps), batch_fn=batch_fn,
                     endpoint_aux=endpoint_aux if structure_stage else 0.0)
    _gc()
    system.save(args.out)
    print(f"[wan-live] mid checkpoint {args.out}")

    if structure_stage or args.resume:
        if not args.no_posttrain:
            try:
                distill(system, steps=s(80 if args.smoke else (300 if micro else 400)),
                        batch=max(1, gen_bs), lr=1e-4)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[wan-live] distill OOM ({e}); keeping prior weights")
                _gc()
        else:
            print("[wan-live] skipping post-train")
    elif not args.no_posttrain:
        guidance_distill(system, lambda: batch_fn("history"),
                         steps=s(40 if args.smoke else 200))
        _gc()
        ema_consistency_distill(system, lambda: batch_fn("history"),
                                steps=s(40 if args.smoke else 200))
        _gc()
        try:
            self_forcing_dmd(system, lambda: batch_fn("history"),
                             steps=s(40 if args.smoke else 200),
                             force_chunks=force_chunks)
            _gc()
            grpo_align(system, lambda: batch_fn("history"),
                       steps=s(20 if args.smoke else 50),
                       group_size=2)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[wan-live] post-train OOM ({e}); keeping prior weights")
            _gc()
    else:
        print("[wan-live] skipping post-train")

    if not args.resume:
        sr_steps = (50 if args.smoke else
                    (100 if micro else (150 if curriculum else 300)))
        train_sr(system, steps=s(sr_steps), batch=sr_bs)

    system.save(args.out)
    print(f"[wan-live] saved {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
