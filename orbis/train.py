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
    # Large canvases are ~95% background — balance fg/bg means or the VAE
    # collapses to black (toy 32² does not need this as strongly).
    balanced = (H * W) >= (128 * 128)
    t0 = time.time()
    _log(f"[vae] device {device_name(device)}", log_cb)
    for step in range(steps):
        frames, _ = sampler.frame_batch(batch, 2)
        x = frames_to_tensor(frames).reshape(batch * 2, 3, H, W).to(device)
        rec, _ = vae(x)
        bg = x.amin(dim=(2, 3), keepdim=True)
        fg = (x - bg).amax(dim=1, keepdim=True)
        err = (rec - x) ** 2
        if balanced:
            fg_m = fg > 0.08
            bg_m = ~fg_m
            # Equal-weight foreground / background means, then boost fg.
            loss_fg = err.mean(dim=1, keepdim=True)[fg_m].mean() if fg_m.any() else err.mean()
            loss_bg = err.mean(dim=1, keepdim=True)[bg_m].mean() if bg_m.any() else err.mean()
            loss = loss_bg + max(fg_weight, 24.0) * loss_fg
        else:
            w = 1.0 + fg_weight * fg
            loss = (w * err).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == steps - 1:
            _log(f"[vae] step {step:4d}/{steps} loss {loss.item():.5f}", log_cb)
    # Calibrate latent scale with enough samples even at 480p (encode in
    # micro-batches so we do not OOM a 24GB card).
    cal_target = 128
    micro = max(1, min(batch, 8))
    acc = []
    vae.eval()
    with torch.no_grad():
        n = 0
        while n < cal_target:
            take = min(micro, cal_target - n)
            frames, _ = sampler.frame_batch(take, 1)
            x = frames_to_tensor(frames).reshape(take, 3, H, W).to(device)
            # Raw encoder outputs (bypass latent_scale) for std estimate.
            vae.latent_scale.fill_(1.0)
            acc.append(vae.encoder(x).reshape(-1))
            n += take
        z_cat = torch.cat(acc)
        vae.latent_scale.fill_(float(z_cat.std()) + 1e-6)
    vae.eval()
    _log(f"[vae] done in {time.time()-t0:.1f}s  latent_scale={float(vae.latent_scale):.4f} "
         f"(cal_n={cal_target})",
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


def _flow_step(system, flow, batch_data, device, pixel_aux: float = 0.35,
               endpoint_aux: float = 0.0, sigma_power: float = 1.0,
               geometry_w: float = 0.0):
    """One RF step with FG-localized velocity loss (+ optional pixel/endpoint)."""
    import torch.nn.functional as F
    from .distill import _few_step
    from .flow import pixel_fg_weight_from_latents
    from .geometry_loss import geometry_aux

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
    sigma = flow.sample_sigma(b, device, power=sigma_power)
    zt = flow.interpolate(target, noise, sigma)
    ctx = gen.encode_context(text_ids, history, reference, memory_state)
    v = gen.forward(zt, sigma, ctx)

    # Prefer decoded FG mask; fall back to latent energy if decode fails.
    try:
        w = pixel_fg_weight_from_latents(system.vae, target)
    except Exception:
        w = flow.latent_fg_weight(target)
    loss = flow.loss(v, target, noise, spatial_weight=w)

    if pixel_aux > 0:
        # Clean-latent estimate: zt - sigma * v  (equals z when v = eps - z).
        s = sigma.view(-1, *([1] * (zt.dim() - 1)))
        z_hat = zt - s * v
        bf, c, lh, lw = (b * target.shape[1], target.shape[2],
                         target.shape[3], target.shape[4])
        with torch.no_grad():
            gt_pix = system.vae.decode(target.reshape(bf, c, lh, lw))
            bg = gt_pix.amin(dim=(2, 3), keepdim=True)
            fg = ((gt_pix - bg).amax(dim=1, keepdim=True) > 0.08)
        pred_pix = system.vae.decode(z_hat.reshape(bf, c, lh, lw))
        err = (pred_pix - gt_pix).pow(2).mean(dim=1, keepdim=True)
        if fg.any() and (~fg).any():
            loss_fg = err[fg].mean()
            loss_bg = err[~fg].mean()
            pred_energy = (pred_pix - pred_pix.amin(dim=(2, 3), keepdim=True)
                           ).amax(dim=1, keepdim=True)
            pred_fg = pred_energy / (pred_energy.amax(dim=(2, 3), keepdim=True)
                                     + 1e-6)
            fg_f = fg.float()
            dice = 1.0 - (2.0 * (pred_fg * fg_f).sum()
                          / (pred_fg.sum() + fg_f.sum() + 1e-6))
            # Direct latent BG hinge (bypass VAE nonlinearity).
            fg_lat = F.adaptive_max_pool2d(fg_f, (lh, lw))
            fg_lat = fg_lat.reshape(b, target.shape[1], 1, lh, lw)
            bg_lat = fg_lat < 0.5
            loss_bg_lat = z_hat.pow(2).mean(dim=2, keepdim=True)
            loss_bg_lat = loss_bg_lat[bg_lat].mean() if bg_lat.any() else 0.0
            # Kill satellite mass outside GT (bright% / n_cc failures).
            bg_energy = (pred_energy * (1.0 - fg_f)).mean()
            loss = loss + pixel_aux * (4.0 * loss_bg + 32.0 * loss_fg
                                      + 16.0 * dice + 8.0 * loss_bg_lat
                                      + 24.0 * bg_energy)
            if geometry_w > 0:
                loss = loss + geometry_w * geometry_aux(pred_pix, gt_pix)
        else:
            loss = loss + pixel_aux * err.mean()

    # Match the few-step Euler endpoint used at inference (train≠sample gap).
    if endpoint_aux > 0:
        ss = system.cfg.flow.student_steps
        z_end = _few_step(lambda zz, ss_: gen.forward(zz, ss_, ctx), noise, ss)
        err_end = (z_end - target).pow(2)
        ww = w
        while ww.dim() < err_end.dim():
            ww = ww.unsqueeze(2)
        loss_end = (ww * err_end).sum() / ww.sum().clamp_min(1e-6)
        bg = (w <= 5.0)
        if bg.any():
            loss_end = loss_end + 8.0 * (z_end.pow(2) * bg).sum() / bg.float().sum().clamp_min(1.0)
        if geometry_w > 0:
            bf = b * target.shape[1]
            c, lh, lw = target.shape[2], target.shape[3], target.shape[4]
            with torch.no_grad():
                gt_pix_e = system.vae.decode(target.reshape(bf, c, lh, lw))
            pred_pix_e = system.vae.decode(z_end.reshape(bf, c, lh, lw))
            # Geometry + off-mask energy on the inference endpoint.
            loss_end = loss_end + geometry_aux(pred_pix_e, gt_pix_e)
            with torch.no_grad():
                bg_e = gt_pix_e.amin(dim=(2, 3), keepdim=True)
                fg_e = ((gt_pix_e - bg_e).amax(dim=1, keepdim=True) > 0.08).float()
            pred_e = (pred_pix_e - pred_pix_e.amin(dim=(2, 3), keepdim=True)
                      ).amax(dim=1, keepdim=True)
            loss_end = loss_end + 12.0 * (pred_e * (1.0 - fg_e)).mean()
        loss = loss + endpoint_aux * loss_end
    return loss


def _endpoint_bootstrap_step(system, batch_data, device, geometry_w: float = 2.5):
    """Few-step Euler → GT only (no RF). Forces inference path to match GT."""
    from .distill import _few_step
    from .geometry_loss import geometry_aux

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
    ctx = gen.encode_context(text_ids, history, reference, memory_state)
    ss = system.cfg.flow.student_steps
    z_end = _few_step(lambda zz, s: gen.forward(zz, s, ctx), noise, ss)

    # Latent match is primary (must fall before pixel aux can help).
    loss = 50.0 * (z_end - target).pow(2).mean()
    bf, c, lh, lw = b * target.shape[1], target.shape[2], target.shape[3], target.shape[4]
    with torch.no_grad():
        gt_pix = system.vae.decode(target.reshape(bf, c, lh, lw))
        bg = gt_pix.amin(dim=(2, 3), keepdim=True)
        fg = ((gt_pix - bg).amax(dim=1, keepdim=True) > 0.08).float()
    pred_pix = system.vae.decode(z_end.reshape(bf, c, lh, lw))
    if fg.any() and (1 - fg).any():
        # FG: match GT color; BG: force near-black (kills wash / satellites).
        loss = loss + 20.0 * ((pred_pix - gt_pix).abs() * fg).sum() / fg.sum().clamp_min(1.0)
        loss = loss + 20.0 * (pred_pix.abs() * (1.0 - fg)).sum() / (1.0 - fg).sum().clamp_min(1.0)
    else:
        loss = loss + 10.0 * (pred_pix - gt_pix).abs().mean()
    if geometry_w > 0:
        loss = loss + geometry_w * geometry_aux(pred_pix, gt_pix)
    return loss


def train_generator(system: OrbisSystem, pretrain_steps: int = 600,
                    stream_steps: int = 1400, batch: int = 48,
                    lr: float = 6e-4, history_noise: float = 0.15,
                    pixel_aux: float = 0.35, endpoint_aux: float = 0.0,
                    sigma_power: float = 1.0, t2v_first: bool = False,
                    geometry_w: float = 0.0, bootstrap_steps: int = 0,
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
    params = list(
        gen.trainable_parameters() if hasattr(gen, "trainable_parameters")
        else gen.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    gen.train()
    _log(f"[generator] device {device_name(device)} trainable={sum(p.numel() for p in params)/1e6:.1f}M",
         log_cb)

    def run(phase, steps, mode_fn, step_fn):
        t0 = time.time()
        ema = None
        for step in range(steps):
            mode, mctx, hnoise = mode_fn(sampler.rng)
            data = sampler.training_batch(batch, mode=mode,
                                          memory_context_chunks=mctx,
                                          history_noise=hnoise)
            loss = step_fn(data)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % 200 == 0 or step == steps - 1:
                _log(f"[{phase}] step {step:4d}/{steps} loss {loss.item():.4f} ema {ema:.4f}",
                     log_cb)
        _log(f"[{phase}] done in {time.time()-t0:.1f}s", log_cb)

    def flow_fn(data):
        return _flow_step(system, flow, data, device, pixel_aux=pixel_aux,
                          endpoint_aux=endpoint_aux, sigma_power=sigma_power,
                          geometry_w=geometry_w)

    def boot_fn(data):
        return _endpoint_bootstrap_step(system, data, device, geometry_w=geometry_w)

    def text_mode(rng):
        return ("reference" if rng.random() < 0.2 else "text_only"), 0, 0.0

    # Optional: pure few-step→GT bootstrap (structure before RF mixing).
    if bootstrap_steps > 0:
        run("bootstrap", bootstrap_steps, text_mode, boot_fn)

    # Stage 1: bidirectional short-clip prior (text-only, some reference).
    def pretrain_mode(rng):
        return ("reference" if rng.random() < 0.3 else "text_only"), 0, 0.0
    run("pretrain", pretrain_steps, pretrain_mode, flow_fn)

    # Stage 2: streaming adaptation (history + memory + augmentation).
    def stream_mode(rng):
        if t2v_first:
            # Match T2V eval: mostly text-only / reference, little history.
            r = rng.random()
            if r < 0.7:
                return "text_only", 0, 0.0
            if r < 0.9:
                return "reference", 0, 0.0
            return "history", 1, 0.0
        r = rng.random()
        if r < 0.6:
            mctx = int(rng.integers(0, 3))       # 0..2 evicted chunks
            hn = history_noise if rng.random() < 0.5 else 0.0
            return "history", mctx, hn
        if r < 0.8:
            return "text_only", 0, 0.0
        return "reference", 0, 0.0
    run("stream", stream_steps, stream_mode, flow_fn)
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
