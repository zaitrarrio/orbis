"""Self-forcing distribution matching on student-generated histories.

Exposure bias is the main consistency killer in long rollouts: the model is
trained on clean GT history but must continue from its own samples at serve
time.  Self-forcing rolls the student for K chunks, then matches a teacher
endpoint on the next chunk conditioned on that *student* history.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import torch

from orbis.device import device_name, get_device
from orbis.flow import RectifiedFlow
from orbis.system import OrbisSystem
from orbis.train import _build_memory, _log


from orbis.distill import _few_step  # noqa: F401


@torch.no_grad()
def _roll_student_history(student, flow, text_ids, reference, memory_state,
                          shape, device, n_chunks: int, steps: int):
    """Autoregressive few-step rollout to build student-generated history."""
    history = None
    mem = memory_state
    cf = shape[1]
    chunks = []
    for _ in range(n_chunks):
        noise = torch.randn(shape, device=device)
        ctx = student.encode_context(text_ids, history, reference, mem)
        z = flow.sample(
            lambda zz, s: student.forward(zz, s, ctx),
            shape, device, steps=steps, noise=noise)
        chunks.append(z)
        if history is None:
            history = z
        else:
            history = torch.cat([history, z], dim=1)
            # keep last history_frames
            hf = student.cfg.history_frames
            if history.shape[1] > hf:
                evicted = history[:, : history.shape[1] - hf]
                for i in range(0, evicted.shape[1], cf):
                    group = evicted[:, i:i + cf]
                    tokens = student.frame_tokens(group, role=2)
                    mem = student.memory.write(mem, tokens)
                history = history[:, -hf:]
    return history, mem, torch.cat(chunks, dim=1) if chunks else None


def self_forcing_dmd(
    system: OrbisSystem,
    batch_fn: Callable[[], dict],
    steps: int = 200,
    lr: float = 1e-4,
    force_chunks: int = 2,
    log_cb=None,
) -> None:
    """Distribution-matching distill with student-forced histories."""
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
    _log(f"[self-forcing] device {device_name(device)} force_chunks={force_chunks}",
         log_cb)

    for step in range(steps):
        data = batch_fn()
        target = data["target"].to(device)
        text_ids = data["text_ids"].to(device)
        b = target.shape[0]
        reference = data["reference"].to(device) if data.get("reference") is not None else None
        evicted = data["evicted"].to(device) if data.get("evicted") is not None else None
        mem0 = _build_memory(student, b, evicted, device)

        # Build student history (no grad), then match next chunk
        with torch.no_grad():
            hist_s, mem_s, _ = _roll_student_history(
                student, flow, text_ids, reference, mem0,
                target.shape, device, force_chunks, ss)

        noise = torch.randn_like(target)
        with torch.no_grad():
            ctx_t = teacher.encode_context(text_ids, hist_s, reference, mem_s)
            z_teacher = flow.sample(
                lambda z, s: teacher.forward(z, s, ctx_t),
                target.shape, device, steps=ts, noise=noise)

        ctx_s = student.encode_context(text_ids, hist_s, reference, mem_s)
        z_student = _few_step(
            lambda z, s: student.forward(z, s, ctx_s), noise, ss)
        # DMD-style: match teacher endpoint under student history (consistency)
        loss = (z_student - z_teacher).pow(2).mean()
        # Optional: also pull toward GT target for stability
        if data.get("target") is not None:
            loss = loss + 0.1 * (z_student - target).pow(2).mean()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            _log(f"[self-forcing] step {step:4d}/{steps} loss {loss.item():.5f}",
                 log_cb)
    student.eval()
    system.distilled = True
