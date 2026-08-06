"""Progressive training: VAE, bidirectional pretrain, streaming adaptation, SR.

The staging mirrors Figure 3 of the paper -- a bidirectional short-clip prior is
adapted into a chunk-wise streaming model with history + memory and history
augmentation.  Sized for fast iteration on a single NVIDIA GPU (Docker / RunPod / Vast).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import OrbisConfig
from .dataset import RolloutSampler
from .device import device_name, get_device
from .flow import RectifiedFlow
from .system import OrbisSystem
from .vae import frames_to_tensor


def _log(msg: str, cb: Optional[Callable[[str], None]]):
    (cb or print)(msg)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------
def train_vae(system: OrbisSystem, steps: int = 800, batch: int = 64,
              lr: float = 1e-3, fg_weight: float = 8.0, log_cb=None) -> None:
    cfg = system.cfg
    device = get_device()
    system.to(device)
    vae = system.vae
    sampler = RolloutSampler(cfg, vae, seed=cfg.seed + 1)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    vae.train()
    H, W = cfg.world.height, cfg.world.width
    t0 = time.time()
    _log(f"[vae] device {device_name(device)}", log_cb)
    for step in range(steps):
        frames, _ = sampler.frame_batch(batch, 2)
        x = frames_to_tensor(frames).reshape(batch * 2, 3, H, W).to(device)
        rec, _ = vae(x)
        # Foreground-weighted reconstruction: the small bright shapes must not be
        # drowned out by the dominant dark background (which causes mean-collapse
        # under plain MSE).
        bg = x.amin(dim=(2, 3), keepdim=True)
        w = 1.0 + fg_weight * (x - bg).amax(dim=1, keepdim=True)
        loss = (w * (rec - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == steps - 1:
            _log(f"[vae] step {step:4d}/{steps} loss {loss.item():.5f}", log_cb)
    # calibrate latent scale on a large sample
    frames, _ = sampler.frame_batch(256, 2)
    x = frames_to_tensor(frames).reshape(512, 3, H, W).to(device)
    vae.calibrate(x)
    vae.eval()
    _log(f"[vae] done in {time.time()-t0:.1f}s  latent_scale={float(vae.latent_scale):.4f}",
         log_cb)


# ---------------------------------------------------------------------------
# Generator (rectified flow)
# ---------------------------------------------------------------------------
def _build_memory(gen, batch, evicted, device):
    state = gen.memory.init(batch, device)
    if evicted is not None and evicted.shape[1] > 0:
        cf = gen.cfg.chunk_frames
        n = evicted.shape[1]
        for s in range(0, n, cf):
            group = evicted[:, s:s + cf]
            tokens = gen.frame_tokens(group, role=2)
            state = gen.memory.write(state, tokens)
    return state


def _flow_step(system, flow, batch_data, device):
    gen = system.generator
    target = batch_data["target"].to(device)
    text_ids = batch_data["text_ids"].to(device)
    b = target.shape[0]
    history = batch_data["history"]
    reference = batch_data["reference"]
    evicted = batch_data["evicted"]
    history = history.to(device) if history is not None else None
    reference = reference.to(device) if reference is not None else None
    evicted = evicted.to(device) if evicted is not None else None

    memory_state = _build_memory(gen, b, evicted, device)
    noise = torch.randn_like(target)
    sigma = flow.sample_sigma(b, device)
    zt = flow.interpolate(target, noise, sigma)
    ctx = gen.encode_context(text_ids, history, reference, memory_state)
    v = gen.forward(zt, sigma, ctx)
    return flow.loss(v, target, noise)


def train_generator(system: OrbisSystem, pretrain_steps: int = 600,
                    stream_steps: int = 1400, batch: int = 48,
                    lr: float = 6e-4, history_noise: float = 0.15,
                    log_cb=None) -> None:
    cfg = system.cfg
    system.vae.eval()
    for p in system.vae.parameters():
        p.requires_grad_(False)
    gen = system.generator
    flow = RectifiedFlow(cfg.flow.train_sigma_eps)
    sampler = RolloutSampler(cfg, system.vae, seed=cfg.seed + 2)
    device = get_device()
    system.to(device)
    opt = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=1e-4)
    gen.train()
    _log(f"[generator] device {device_name(device)}", log_cb)

    def run(phase, steps, mode_fn):
        t0 = time.time()
        ema = None
        for step in range(steps):
            mode, mctx, hnoise = mode_fn(sampler.rng)
            data = sampler.training_batch(batch, mode=mode,
                                          memory_context_chunks=mctx,
                                          history_noise=hnoise)
            loss = _flow_step(system, flow, data, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % 200 == 0 or step == steps - 1:
                _log(f"[{phase}] step {step:4d}/{steps} loss {loss.item():.4f} ema {ema:.4f}",
                     log_cb)
        _log(f"[{phase}] done in {time.time()-t0:.1f}s", log_cb)

    # Stage 1: bidirectional short-clip prior (text-only, some reference).
    def pretrain_mode(rng):
        return ("reference" if rng.random() < 0.3 else "text_only"), 0, 0.0
    run("pretrain", pretrain_steps, pretrain_mode)

    # Stage 2: streaming adaptation (history + memory + augmentation).
    def stream_mode(rng):
        r = rng.random()
        if r < 0.6:
            mctx = int(rng.integers(0, 3))       # 0..2 evicted chunks
            hn = history_noise if rng.random() < 0.5 else 0.0
            return "history", mctx, hn
        if r < 0.8:
            return "text_only", 0, 0.0
        return "reference", 0, 0.0
    run("stream", stream_steps, stream_mode)
    gen.eval()


# ---------------------------------------------------------------------------
# Super-resolution
# ---------------------------------------------------------------------------
def train_sr(system: OrbisSystem, steps: int = 300, batch: int = 24,
             lr: float = 2e-3, log_cb=None) -> None:
    cfg = system.cfg
    device = get_device()
    system.to(device)
    sr = system.sr
    scale = cfg.sr.scale
    H, W = cfg.world.height, cfg.world.width
    Hs, Ws = H * scale, W * scale
    from .world import sample_scene, rollout
    rng = np.random.default_rng(cfg.seed + 3)
    opt = torch.optim.AdamW(sr.parameters(), lr=lr)
    sr.train()
    t0 = time.time()
    cf = cfg.model.chunk_frames
    _log(f"[sr] device {device_name(device)}", log_cb)
    for step in range(steps):
        hr_frames = np.empty((batch, cf, Hs, Ws, 3), dtype=np.float32)
        for b in range(batch):
            spec = sample_scene(rng)
            hi, _ = rollout(spec, cf, Hs, Ws)
            hr_frames[b] = hi
        hi = frames_to_tensor(hr_frames).to(device)            # (B,cf,3,Hs,Ws)
        # Low-res supervision is the anti-aliased downsample of the HR target,
        # so LR/HR are aligned exactly (a true low-pass pair).
        lo = F.avg_pool2d(hi.reshape(batch * cf, 3, Hs, Ws), scale)
        lo = lo.reshape(batch, cf, 3, H, W)
        out = sr(lo)
        loss = F.l1_loss(out, hi)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            _log(f"[sr] step {step:4d}/{steps} loss {loss.item():.5f}", log_cb)
    sr.eval()
    _log(f"[sr] done in {time.time()-t0:.1f}s", log_cb)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def train_all(cfg: Optional[OrbisConfig] = None, out_path: str = "orbis.pt",
              scale: float = 1.0, log_cb=None) -> OrbisSystem:
    """Train every stage and save a checkpoint.  ``scale`` shrinks all step
    counts for a fast smoke run (e.g. ``scale=0.1``)."""
    from .distill import distill
    system = OrbisSystem.build(cfg)
    s = lambda n: max(20, int(n * scale))
    train_vae(system, steps=s(800), log_cb=log_cb)
    train_generator(system, pretrain_steps=s(600), stream_steps=s(1400),
                    log_cb=log_cb)
    distill(system, steps=s(500), log_cb=log_cb)
    train_sr(system, steps=s(300), log_cb=log_cb)
    system.save(out_path)
    _log(f"[all] saved checkpoint -> {out_path}", log_cb)
    return system
