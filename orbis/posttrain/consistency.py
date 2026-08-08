"""Endpoint-anchored consistency distillation with an EMA teacher."""

from __future__ import annotations

import copy
from typing import Callable

import torch

from orbis.device import device_name, get_device
from orbis.flow import RectifiedFlow
from orbis.system import OrbisSystem
from orbis.train import _build_memory, _log


@torch.no_grad()
def _ema_update(ema: torch.nn.Module, student: torch.nn.Module, decay: float):
    for e, s in zip(ema.parameters(), student.parameters()):
        e.data.mul_(decay).add_(s.data, alpha=1.0 - decay)


from orbis.distill import _few_step  # noqa: F401


def ema_consistency_distill(
    system: OrbisSystem,
    batch_fn: Callable[[], dict],
    steps: int = 200,
    lr: float = 1e-4,
    ema_decay: float = 0.999,
    log_cb=None,
) -> torch.nn.Module:
    """Train student few-step rollout to match EMA teacher many-step endpoint.

    Returns the EMA teacher module (frozen snapshot for later stages).
    """
    cfg = system.cfg
    student = system.generator
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    device = get_device()
    system.to(device)
    ema.to(device)
    flow = RectifiedFlow(cfg.flow.train_sigma_eps)
    params = list(
        student.trainable_parameters()
        if hasattr(student, "trainable_parameters")
        else student.parameters()
    )
    opt = torch.optim.AdamW(params, lr=lr)
    student.train()
    ts, ss = cfg.flow.teacher_steps, cfg.flow.student_steps
    _log(f"[ema-consistency] device {device_name(device)} decay={ema_decay}", log_cb)

    for step in range(steps):
        data = batch_fn()
        target = data["target"].to(device)
        text_ids = data["text_ids"].to(device)
        b = target.shape[0]
        history = data["history"].to(device) if data.get("history") is not None else None
        reference = data["reference"].to(device) if data.get("reference") is not None else None
        evicted = data["evicted"].to(device) if data.get("evicted") is not None else None
        noise = torch.randn_like(target)

        with torch.no_grad():
            mem_t = _build_memory(ema, b, evicted, device)
            ctx_t = ema.encode_context(text_ids, history, reference, mem_t)
            z_teacher = flow.sample(
                lambda z, s: ema.forward(z, s, ctx_t),
                target.shape, device, steps=ts, noise=noise)

        mem_s = _build_memory(student, b, evicted, device)
        ctx_s = student.encode_context(text_ids, history, reference, mem_s)
        z_student = _few_step(
            lambda z, s: student.forward(z, s, ctx_s), noise, ss)
        loss = (z_student - z_teacher).pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        _ema_update(ema, student, ema_decay)
        if step % 50 == 0 or step == steps - 1:
            _log(f"[ema-consistency] step {step:4d}/{steps} loss {loss.item():.5f}",
                 log_cb)
    student.eval()
    system.distilled = True
    return ema
