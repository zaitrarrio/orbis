"""GRPO reward alignment with world-model consistency.

Samples G candidate chunk rollouts per prompt, scores them with RewardBundle
(visual / motion / text / reference / world-model), and optimizes a
group-relative policy gradient on the student generator.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from orbis.device import device_name, get_device
from orbis.flow import RectifiedFlow
from orbis.posttrain.rewards import RewardBundle
from orbis.system import OrbisSystem
from orbis.train import _build_memory, _log


def _few_step(vel_fn, noise, steps):
    z = noise
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
    for i in range(steps):
        s = sigmas[i].expand(z.shape[0])
        z = z - (sigmas[i] - sigmas[i + 1]) * vel_fn(z, s)
    return z


def grpo_align(
    system: OrbisSystem,
    batch_fn: Callable[[], dict],
    steps: int = 100,
    group_size: int = 4,
    lr: float = 5e-5,
    log_cb=None,
) -> RewardBundle:
    cfg = system.cfg
    student = system.generator
    device = get_device()
    system.to(device)
    flow = RectifiedFlow(cfg.flow.train_sigma_eps)
    rewards = RewardBundle(
        dim=cfg.model.dim,
        latent_channels=cfg.vae.latent_channels,
        chunk_frames=cfg.model.chunk_frames,
        latent_hw=cfg.latent_hw,
    ).to(device)

    params = list(
        student.trainable_parameters()
        if hasattr(student, "trainable_parameters")
        else student.parameters()
    ) + list(rewards.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    student.train()
    rewards.train()
    ss = cfg.flow.student_steps
    _log(f"[grpo] device {device_name(device)} G={group_size}", log_cb)

    for step in range(steps):
        data = batch_fn()
        target = data["target"].to(device)
        text_ids = data["text_ids"].to(device)
        b = target.shape[0]
        history = data["history"].to(device) if data.get("history") is not None else None
        reference = data["reference"].to(device) if data.get("reference") is not None else None
        evicted = data["evicted"].to(device) if data.get("evicted") is not None else None
        mem = _build_memory(student, b, evicted, device)
        ctx = student.encode_context(text_ids, history, reference, mem)
        pooled = ctx.pooled_text

        # Sample G candidates (different noise)
        cands = []
        for _ in range(group_size):
            noise = torch.randn_like(target)
            z = _few_step(lambda zz, s: student.forward(zz, s, ctx), noise, ss)
            cands.append(z)
        stacked = torch.stack(cands, dim=1)          # (B, G, F, C, H, W)

        # Score each candidate
        score_list = []
        detail = None
        for g in range(group_size):
            sc, detail = rewards(
                stacked[:, g], target, pooled, history, mem, reference)
            score_list.append(sc)
        scores = torch.stack(score_list, dim=1)      # (B, G)

        # Group-relative advantages
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True).clamp_min(1e-3)
        adv = (scores - mean) / std                  # (B, G)

        # Surrogate: weight endpoint MSE to best-adv candidates via -adv * error
        # (higher adv → lower error pressure toward matching that sample's...
        #  actually GRPO maximizes reward; we push student toward high-adv
        #  samples by matching the stop-grad winning latent)
        best_idx = adv.argmax(dim=1)                 # (B,)
        best = stacked[torch.arange(b, device=device), best_idx].detach()

        noise = torch.randn_like(target)
        z_new = _few_step(lambda zz, s: student.forward(zz, s, ctx), noise, ss)
        # Align new sample to best candidate + world-model consistency via reward loss
        policy_loss = (z_new - best).pow(2).mean()
        # Train world model on GT targets
        wm_pred = rewards.wm(history, mem, pooled)
        wm_loss = F.mse_loss(wm_pred, target)
        # Prefer high mean reward
        reward_loss = -scores.mean()

        loss = policy_loss + 0.5 * wm_loss + 0.1 * reward_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            d = {k: float(v) for k, v in (detail or {}).items()}
            _log(f"[grpo] step {step:4d}/{steps} loss {loss.item():.5f} "
                 f"R {scores.mean().item():.4f} {d}", log_cb)

    student.eval()
    rewards.eval()
    return rewards
