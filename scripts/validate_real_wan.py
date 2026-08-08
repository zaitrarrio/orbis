"""Standalone real-GPU fidelity smoke test for RealWanBackbone (Phase 1a).

Loads the actual Wan2.1-1.3B transformer + UMT5 text encoder from Hugging
Face, wraps them with orbis's LoRA/memory/history integration
(``orbis.adapters.wan21_real.RealWanBackbone``), and runs a few real
forward + backward passes with synthetic latents at the real config's
shapes to confirm:

  * the real ``diffusers.WanTransformer3DModel`` loads and its block
    layout matches what ``wan21_real.py``'s structural discovery expects
    (no ``ImportError``/``AttributeError`` -- this is the main risk this
    script exists to catch, since the sandbox that wrote this integration
    has no GPU/diffusers and could only validate against a hand-built
    mock transformer; see ``tests/test_wan21_real_adapter.py``)
  * a full ``velocity()`` call (``encode_context`` + ``forward``) produces
    the expected output shape
  * gradients flow into LoRA + memory/projection parameters and NOT into
    the frozen base transformer
  * ``state_dict()``/checkpoint save round-trips a SMALL file (prints
    size) rather than the multi-GB frozen base

This intentionally does NOT run the full 7-stage pipeline in
``scripts/train-live-wan.py`` (VAE pretraining, guidance distillation,
DMD, GRPO, streaming SR) -- that is a much longer, more expensive run
appropriate for a dedicated training session once this integration itself
is confirmed sound. This script is a fast (~1-3 min after weights are
cached), cheap architecture-fidelity check for this PR specifically.

Usage (see deploy/README.md):
    uv sync --extra wan
    export HF_HOME=/workspace/hf-cache
    python scripts/validate_real_wan.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, ".")

from orbis.adapters.factory import build_backbone
from orbis.config import wan21_real_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="HF repo/path (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)")
    ap.add_argument("--text-encoder", default=None,
                    help="HF repo/path for UMT5 tokenizer/encoder (default: same as --checkpoint)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ckpt-out", default="/tmp/orbis-real-wan-smoke.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[validate-real-wan] device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        print("[validate-real-wan] WARNING: no GPU detected; this will be "
              "very slow and is not representative of real usage.")

    cfg = wan21_real_config(checkpoint_path=args.checkpoint,
                             text_encoder_path=args.text_encoder)
    print(f"[validate-real-wan] checkpoint={cfg.backbone.checkpoint_path} "
          f"text_encoder={cfg.backbone.text_encoder_path or '(same repo)'}")

    t0 = time.time()
    print("[validate-real-wan] loading real Wan2.1-1.3B transformer + UMT5 "
          "text encoder from Hugging Face (first run downloads weights) ...")
    gen = build_backbone(cfg).to(device)
    print(f"[validate-real-wan] loaded in {time.time() - t0:.1f}s")

    trainable = sum(p.numel() for p in gen.trainable_parameters())
    total = sum(p.numel() for p in gen.parameters())
    frac = trainable / total if total else 0.0
    print(f"[validate-real-wan] trainable params: {trainable:,} / "
          f"total params: {total:,} ({frac:.4%})")
    assert trainable > 0, "no trainable parameters found (LoRA injection failed?)"
    assert frac < 0.05, (
        "expected LoRA + memory/projection trainable params to be a small "
        f"fraction of the frozen base transformer, got {frac:.2%}")

    b = args.batch
    lh, lw = cfg.latent_hw
    lc = cfg.vae.latent_channels
    cf = cfg.model.chunk_frames
    hist_f = cfg.model.history_frames
    z = torch.randn(b, cf, lc, lh, lw, device=device)
    history = torch.randn(b, hist_f, lc, lh, lw, device=device)
    reference = torch.randn(b, 1, lc, lh, lw, device=device)
    text_ids = torch.zeros(b, cfg.model.text_len, dtype=torch.long, device=device)
    mem_state = gen.memory.init(b, device)

    t0 = time.time()
    out = gen.velocity(z, torch.full((b,), 0.5, device=device), text_ids,
                        history, reference, mem_state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[validate-real-wan] forward pass in {time.time() - t0:.1f}s, "
          f"output shape {tuple(out.shape)} (expected {tuple(z.shape)})")
    assert out.shape == z.shape, f"shape mismatch: {out.shape} != {z.shape}"

    t0 = time.time()
    out.float().sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[validate-real-wan] backward pass in {time.time() - t0:.1f}s")

    lora_grad, base_grad_leak = 0, 0
    for n, p in gen.named_parameters():
        if p.grad is None:
            continue
        if ".base." in n:
            base_grad_leak += 1
        else:
            lora_grad += 1
    print(f"[validate-real-wan] params with grad: lora/memory={lora_grad}, "
          f"frozen-base-leak={base_grad_leak}")
    assert base_grad_leak == 0, "gradient leaked into frozen base weights!"
    assert lora_grad > 0, "no gradient reached any trainable parameter!"

    sd = gen.state_dict()
    torch.save(sd, args.ckpt_out)
    size_mb = os.path.getsize(args.ckpt_out) / (1024 ** 2)
    print(f"[validate-real-wan] checkpoint: {len(sd)} tensors, "
          f"{size_mb:.1f} MB at {args.ckpt_out}")
    assert size_mb < 500, (
        "checkpoint unexpectedly large -- frozen base weights may be "
        "leaking into state_dict()")

    # Reload onto a fresh instance to confirm the lean checkpoint actually
    # restores correctly (mirrors OrbisSystem.load()'s generic call path).
    gen2 = build_backbone(cfg).to(device)
    gen2.load_state_dict(torch.load(args.ckpt_out, map_location=device))
    print("[validate-real-wan] checkpoint reload OK on a fresh instance")

    print("[validate-real-wan] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
