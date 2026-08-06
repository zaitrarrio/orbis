"""Few-step distillation for the real-time path.

Post-training reduces inference cost: a frozen many-step *teacher* (the trained
rectified-flow model) supervises the *student* (the same weights, fine-tuned) so
that a short ``student_steps`` Euler rollout reproduces the accurate
``teacher_steps`` trajectory endpoint from the same noise and context.  This is
the toy analogue of the paper's endpoint-anchored consistency / self-forcing
distillation; deployment then samples in a handful of steps.
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import torch

from .dataset import RolloutSampler
from .device import device_name, get_device
from .flow import RectifiedFlow
from .system import OrbisSystem
from .train import _build_memory, _log


def _few_step(vel_fn, noise, steps):
    """Grad-enabled few-step Euler rollout noise(sigma=1) -> data(sigma=0)."""
    z = noise
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
    for i in range(steps):
        s = sigmas[i].expand(z.shape[0])
        z = z - (sigmas[i] - sigmas[i + 1]) * vel_fn(z, s)
    return z


def distill(system: OrbisSystem, steps: int = 500, batch: int = 32,
            lr: float = 2e-4, log_cb=None) -> None:
    cfg = system.cfg
    student = system.generator
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    flow = RectifiedFlow(cfg.flow.train_sigma_eps)
    sampler = RolloutSampler(cfg, system.vae, seed=cfg.seed + 4)
    device = get_device()
    system.to(device)
    teacher.to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    student.train()
    t0 = time.time()
    ts, ss = cfg.flow.teacher_steps, cfg.flow.student_steps
    _log(f"[distill] device {device_name(device)}", log_cb)

    for step in range(steps):
        r = sampler.rng.random()
        mode = "history" if r < 0.7 else ("reference" if r < 0.85 else "text_only")
        mctx = int(sampler.rng.integers(0, 3)) if mode == "history" else 0
        data = sampler.training_batch(batch, mode=mode, memory_context_chunks=mctx)
        target = data["target"].to(device)
        text_ids = data["text_ids"].to(device)
        b = target.shape[0]
        history = data["history"].to(device) if data["history"] is not None else None
        reference = data["reference"].to(device) if data["reference"] is not None else None
        evicted = data["evicted"].to(device) if data["evicted"] is not None else None

        noise = torch.randn_like(target)
        with torch.no_grad():
            mem_t = _build_memory(teacher, b, evicted, device)
            ctx_t = teacher.encode_context(text_ids, history, reference, mem_t)
            z_teacher = flow.sample(lambda z, s: teacher.forward(z, s, ctx_t),
                                    target.shape, device, steps=ts, noise=noise)

        mem_s = _build_memory(student, b, evicted, device)
        ctx_s = student.encode_context(text_ids, history, reference, mem_s)
        z_student = _few_step(lambda z, s: student.forward(z, s, ctx_s), noise, ss)
        loss = (z_student - z_teacher).pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            _log(f"[distill] step {step:4d}/{steps} loss {loss.item():.5f}", log_cb)
    student.eval()
    system.distilled = True
    _log(f"[distill] done in {time.time()-t0:.1f}s", log_cb)
