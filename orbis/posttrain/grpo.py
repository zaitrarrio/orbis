"""Flow-GRPO / DanceGRPO-style clipped-ratio RL post-training (Phase 1b).

Replaces the earlier best-of-G self-distillation heuristic. That version
correctly computed group-relative advantages, but its "policy loss" was an
L2 regression of a freshly-sampled rollout toward the single winning
candidate -- imitation learning, not a policy gradient. It had no
importance ratio, no clipping, and no per-step likelihood, so nothing in it
actually implemented GRPO's clipped surrogate objective.

This module implements the real thing, following Flow-GRPO
(arXiv:2505.05470) and DanceGRPO (arXiv:2505.07818):

  1. Snapshot the current policy as a frozen "old" policy (mirrors the
     existing teacher/student pattern already used in ``orbis/distill.py``).
  2. Roll out ``group_size`` stochastic trajectories per prompt under the
     old policy via :class:`orbis.posttrain.flow_sde.FlowSDE`, using fewer
     denoising steps than final inference (Flow-GRPO's "Denoising
     Reduction" strategy -- ``cfg.grpo.train_denoise_steps``), recording the
     Gaussian transition (mean, std, sampled next-state) at every step.
  3. Decode each trajectory's final chunk to pixels and score it with a
     real reward model (default in production: CLIP image-text alignment;
     see ``orbis/posttrain/reward_models.py``) plus orbis's own
     reference/motion continuity terms.
  4. Compute group-relative advantages ``A_i = (R_i - mean(R)) / std(R)``
     (unchanged from the earlier implementation -- this part was already
     correct GRPO bookkeeping).
  5. Recompute each stored step's transition mean under the *current*
     (trainable) policy, form the per-step importance ratio
     ``rho = exp(logp_theta - logp_theta_old)``, and optimize the PPO/GRPO
     clipped surrogate ``min(rho*A, clip(rho, 1-eps, 1+eps)*A)`` minus a KL
     penalty against the frozen old policy.

The world-model consistency term from the earlier implementation is kept as
an explicitly-labeled *auxiliary* structure-learning loss (it needs a
ground-truth target, so it cannot be a GRPO "reward" -- but it is a
legitimate supervised signal for the long-horizon consistency this
streaming-video task cares about), not folded into the reward.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from orbis.device import device_name, get_device
from orbis.posttrain.flow_sde import FlowSDE
from orbis.posttrain.reward_models import (
    ClipAlignmentReward,
    CompositeReward,
    MockReward,
    RewardModel,
)
from orbis.posttrain.rewards import LatentWorldModel
from orbis.system import OrbisSystem
from orbis.train import _build_memory, _log


def _default_reward(cfg) -> CompositeReward:
    """Real-reward composite: CLIP alignment + latent-space continuity terms."""
    clip = ClipAlignmentReward()
    return CompositeReward([(clip, 1.0)],
                           w_reference=cfg.grpo.w_reference,
                           w_motion=cfg.grpo.w_motion)


def _mock_reward(cfg) -> CompositeReward:
    """CPU-testable, network-free reward. Never use for real training."""
    return CompositeReward([(MockReward(), 1.0)],
                           w_reference=cfg.grpo.w_reference,
                           w_motion=cfg.grpo.w_motion)


def grpo_align(
    system: OrbisSystem,
    batch_fn: Callable[[], dict],
    steps: int = 100,
    group_size: int = 4,
    lr: float = 5e-5,
    eta: Optional[float] = None,
    clip_eps: Optional[float] = None,
    kl_coef: Optional[float] = None,
    denoise_steps: Optional[int] = None,
    reward_model: Optional[RewardModel] = None,
    log_cb=None,
) -> RewardModel:
    """Clipped-ratio GRPO over a stochastic (SDE) flow-matching rollout.

    Backward compatible with the previous call signature (``system,
    batch_fn, steps, group_size, lr, log_cb``); new knobs default to
    ``system.cfg.grpo`` (see :class:`orbis.config.GRPOConfig`). Pass an
    explicit ``reward_model`` to override ``cfg.grpo.reward_backend``
    (e.g. a :class:`~orbis.posttrain.reward_models.MockReward`-backed
    composite for CPU unit tests).
    """
    cfg = system.cfg
    gcfg = cfg.grpo
    eta = gcfg.sde_eta if eta is None else eta
    clip_eps = gcfg.clip_eps if clip_eps is None else clip_eps
    kl_coef = gcfg.kl_coef if kl_coef is None else kl_coef
    denoise_steps = gcfg.train_denoise_steps if denoise_steps is None else denoise_steps
    if eta <= 0:
        raise ValueError(
            "grpo_align requires eta > 0 for a well-defined importance ratio "
            "(eta=0 degenerates the SDE to the original deterministic ODE, "
            "which has no transition density). Use FlowSDE directly with "
            "eta=0 only for deterministic sampling, not GRPO training.")

    student = system.generator
    device = get_device()
    system.to(device)
    sde = FlowSDE(eta=eta)

    if reward_model is None:
        reward_model = _mock_reward(cfg) if gcfg.reward_backend == "mock" else _default_reward(cfg)
    reward_model = reward_model.to(device)

    wm = LatentWorldModel(cfg.model.dim, cfg.vae.latent_channels,
                          cfg.model.chunk_frames, cfg.latent_hw).to(device)

    params = list(
        student.trainable_parameters()
        if hasattr(student, "trainable_parameters")
        else student.parameters()
    ) + list(reward_model.parameters()) + list(wm.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    student.train()
    reward_model.train()
    _log(f"[grpo] device {device_name(device)} G={group_size} eta={eta} "
        f"clip_eps={clip_eps} kl_coef={kl_coef} rollout_steps={denoise_steps} "
        f"reward={gcfg.reward_backend}", log_cb)

    for step in range(steps):
        data = batch_fn()
        target = data["target"].to(device)
        text_ids = data["text_ids"].to(device)
        prompts = data.get("prompts")
        b = target.shape[0]
        history = data["history"].to(device) if data.get("history") is not None else None
        reference = data["reference"].to(device) if data.get("reference") is not None else None
        evicted = data["evicted"].to(device) if data.get("evicted") is not None else None

        # Frozen "old" policy snapshot for this update (same deepcopy pattern
        # as the teacher/student split in orbis/distill.py).
        old_student = copy.deepcopy(student).eval()
        for p in old_student.parameters():
            p.requires_grad_(False)

        mem_old = _build_memory(old_student, b, evicted, device)
        ctx_old = old_student.encode_context(text_ids, history, reference, mem_old)

        # -- 1. Roll out G stochastic trajectories under the OLD policy -----
        all_traces = []
        all_finals = []
        for _ in range(group_size):
            noise = torch.randn_like(target)
            z_final, trace = sde.rollout(
                lambda zz, s: old_student.forward(zz, s, ctx_old),
                target.shape, device, denoise_steps, noise=noise)
            all_traces.append(trace)
            all_finals.append(z_final)
        stacked_final = torch.stack(all_finals, dim=1)  # (B, G, F, C, H, W)

        # -- 2. Score each candidate with the (black-box) reward model ------
        score_list = []
        detail = {}
        with torch.no_grad():
            for g in range(group_size):
                chunk = stacked_final[:, g]
                f = chunk.shape[1]
                try:
                    px = system.vae.decode(chunk.reshape(b * f, *chunk.shape[2:]))
                    frames = px.reshape(b, f, *px.shape[1:])
                except Exception:
                    frames = chunk  # fallback: score in latent space if decode fails
                r, d = reward_model(frames, prompts=prompts, latent_chunk=chunk,
                                    reference=reference)
                detail = d
                score_list.append(r)
        scores = torch.stack(score_list, dim=1)  # (B, G)

        # -- 3. Group-relative advantage (unchanged GRPO bookkeeping) -------
        mean_r = scores.mean(dim=1, keepdim=True)
        std_r = scores.std(dim=1, keepdim=True).clamp_min(1e-3)
        adv = (scores - mean_r) / std_r  # (B, G)

        # -- 4. Recompute means under the CURRENT policy; clipped surrogate -
        mem_new = _build_memory(student, b, evicted, device)
        ctx_new = student.encode_context(text_ids, history, reference, mem_new)

        clip_terms = []
        kl_terms = []
        ratio_log = []
        for g in range(group_size):
            adv_g = adv[:, g]  # (B,)
            for t_step in all_traces[g]:
                v_new = student.forward(t_step.z, t_step.sigma, ctx_new)
                mean_new, _ = sde.mean_and_std(t_step.z, t_step.sigma,
                                               t_step.next_sigma, v_new)
                logp_new = FlowSDE.log_prob(t_step.z_next, mean_new, t_step.std)
                logp_old = FlowSDE.log_prob(t_step.z_next, t_step.mean, t_step.std)
                ratio = torch.exp((logp_new - logp_old).clamp(-20, 20))
                surrogate = torch.min(
                    ratio * adv_g,
                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_g)
                clip_terms.append(surrogate)
                kl_terms.append(FlowSDE.gaussian_kl(mean_new, t_step.mean, t_step.std))
                ratio_log.append(ratio.detach().mean())

        policy_loss = -torch.stack(clip_terms).mean()
        kl_loss = torch.stack(kl_terms).mean()

        # -- 5. Auxiliary world-model consistency (supervised, not a reward) -
        pooled_new = ctx_new.pooled_text
        wm_pred = wm(history, mem_new, pooled_new)
        wm_loss = F.mse_loss(wm_pred, target)

        loss = policy_loss + kl_coef * kl_loss + 0.1 * wm_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 25 == 0 or step == steps - 1:
            d = {k: float(v) for k, v in (detail or {}).items()}
            ratio_mean = float(torch.stack(ratio_log).mean()) if ratio_log else float("nan")
            _log(f"[grpo] step {step:4d}/{steps} loss {loss.item():.5f} "
                f"policy {policy_loss.item():.5f} kl {kl_loss.item():.5f} "
                f"R {scores.mean().item():.4f} ratio {ratio_mean:.3f} {d}", log_cb)

    student.eval()
    reward_model.eval()
    return reward_model
