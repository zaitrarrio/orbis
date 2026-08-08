"""Guidance distillation into a conditional few-step student.

Teacher is evaluated with classifier-free guidance (cond − uncond); the student
matches the guided velocity / endpoint so deployment can drop CFG.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn

from orbis.device import device_name, get_device
from orbis.flow import RectifiedFlow
from orbis.system import OrbisSystem
from orbis.train import _build_memory, _log


from orbis.distill import _few_step  # noqa: F401  — shared Euler helper


def guidance_distill(
    system: OrbisSystem,
    batch_fn: Callable[[], dict],
    steps: int = 200,
    batch: int = 8,
    lr: float = 1e-4,
    guidance_scale: float = 3.0,
    log_cb=None,
) -> None:
    """Distill CFG teacher into the student generator/adapter."""
    cfg = system.cfg
    student = system.generator
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    device = get_device()
    system.to(device)
    teacher.to(device)
    flow = RectifiedFlow(cfg.flow.train_sigma_eps)
    params = list(
        student.trainable_parameters()
        if hasattr(student, "trainable_parameters")
        else student.parameters()
    )
    opt = torch.optim.AdamW(params, lr=lr)
    student.train()
    ts, ss = cfg.flow.teacher_steps, cfg.flow.student_steps
    _log(f"[guidance] device {device_name(device)} scale={guidance_scale}", log_cb)

    for step in range(steps):
        data = batch_fn()
        target = data["target"].to(device)
        text_ids = data["text_ids"].to(device)
        b = target.shape[0]
        history = data["history"].to(device) if data.get("history") is not None else None
        reference = data["reference"].to(device) if data.get("reference") is not None else None
        evicted = data["evicted"].to(device) if data.get("evicted") is not None else None
        null_ids = torch.zeros_like(text_ids)

        noise = torch.randn_like(target)
        with torch.no_grad():
            mem_t = _build_memory(teacher, b, evicted, device)
            ctx_c = teacher.encode_context(text_ids, history, reference, mem_t)
            ctx_u = teacher.encode_context(null_ids, history, reference, mem_t)

            def guided(z, s):
                vc = teacher.forward(z, s, ctx_c)
                vu = teacher.forward(z, s, ctx_u)
                return vu + guidance_scale * (vc - vu)

            z_teacher = flow.sample(
                guided, target.shape, device, steps=ts, noise=noise)

        mem_s = _build_memory(student, b, evicted, device)
        ctx_s = student.encode_context(text_ids, history, reference, mem_s)
        z_student = _few_step(
            lambda z, s: student.forward(z, s, ctx_s), noise, ss)
        loss = (z_student - z_teacher).pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            _log(f"[guidance] step {step:4d}/{steps} loss {loss.item():.5f}", log_cb)
    student.eval()
    system.distilled = True
