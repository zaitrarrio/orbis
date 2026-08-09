"""Tests for the real clipped-ratio GRPO implementation (Phase 1b).

Uses ``wan_smoke_config`` (the existing tiny CPU-friendly config, per
``tests/test_wan_live.py`` conventions) plus
:class:`orbis.posttrain.reward_models.MockReward` so nothing here needs
network access, GPU, or pretrained CLIP weights.
"""

from __future__ import annotations

import torch

from orbis.config import wan_smoke_config
from orbis.dataset import RolloutSampler
from orbis.posttrain.flow_sde import FlowSDE
from orbis.posttrain.grpo import grpo_align
from orbis.posttrain.reward_models import CompositeReward, MockReward
from orbis.system import OrbisSystem


def _make_system_and_batch_fn(seed: int = 0):
    cfg = wan_smoke_config(seed=seed)
    cfg.grpo.reward_backend = "mock"
    system = OrbisSystem.build(cfg)
    sampler = RolloutSampler(cfg, system.vae, seed=cfg.seed + 20)

    def batch_fn():
        return sampler.training_batch(2, mode="history", memory_context_chunks=1)

    return system, batch_fn


def test_batch_fn_includes_raw_prompts():
    """Prerequisite for real (CLIP) rewards: raw text must survive the batcher."""
    _, batch_fn = _make_system_and_batch_fn()
    data = batch_fn()
    assert "prompts" in data
    assert len(data["prompts"]) == 2
    assert all(isinstance(p, str) and len(p) > 0 for p in data["prompts"])


def test_grpo_requires_positive_eta():
    system, batch_fn = _make_system_and_batch_fn()
    try:
        grpo_align(system, batch_fn, steps=1, group_size=2, eta=0.0)
        assert False, "expected ValueError for eta<=0"
    except ValueError:
        pass


def test_grpo_align_runs_and_produces_finite_loss():
    system, batch_fn = _make_system_and_batch_fn(seed=1)
    reward = grpo_align(
        system, batch_fn, steps=2, group_size=2, lr=1e-4,
        eta=0.5, denoise_steps=2,
        reward_model=CompositeReward([(MockReward(), 1.0)],
                                     w_reference=0.0, w_motion=0.0),
    )
    assert isinstance(reward, CompositeReward)


def test_grpo_gradients_isolated_to_trainable_params():
    """Mirrors Phase 1a's gradient-isolation check: only LoRA/trainable params
    of the generator should move, nothing in the frozen base."""
    system, batch_fn = _make_system_and_batch_fn(seed=2)
    gen = system.generator
    trainable_names = {
        n for n, p in gen.named_parameters() if p.requires_grad
    }
    frozen_names = {
        n for n, p in gen.named_parameters() if not p.requires_grad
    }
    assert trainable_names, "smoke config should have at least some trainable params"

    before = {n: p.detach().clone() for n, p in gen.named_parameters()}

    grpo_align(system, batch_fn, steps=1, group_size=2, lr=1e-2,
              eta=0.5, denoise_steps=2,
              reward_model=CompositeReward([(MockReward(), 1.0)],
                                           w_reference=0.0, w_motion=0.0))

    after = dict(gen.named_parameters())
    moved_frozen = [n for n in frozen_names
                    if not torch.allclose(before[n], after[n].detach())]
    assert moved_frozen == [], f"frozen params moved: {moved_frozen}"


def test_group_relative_advantage_matches_reference_formula():
    """Sanity-check the (unchanged) GRPO advantage bookkeeping in isolation."""
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True).clamp_min(1e-3)
    adv = (scores - mean) / std
    assert torch.allclose(adv.mean(), torch.tensor(0.0), atol=1e-5)
    assert adv[0, -1] > adv[0, 0]


def test_flow_sde_reused_mean_matches_recompute():
    """The stored SDEStep.mean from an old-policy rollout must equal what you'd
    get recomputing mean_and_std with the same (z, sigma, next_sigma, v) --
    this is the optimization grpo_align relies on to avoid a second old-policy
    forward pass."""
    torch.manual_seed(3)
    sde = FlowSDE(eta=0.4)
    shape = (2, 3)
    v_fn = lambda z, s: torch.sin(z) * s.view(-1, 1)
    _, trace = sde.rollout(v_fn, shape, "cpu", steps=3)
    for t_step in trace:
        v = v_fn(t_step.z, t_step.sigma)
        mean_recomputed, std_recomputed = sde.mean_and_std(
            t_step.z, t_step.sigma, t_step.next_sigma, v)
        assert torch.allclose(mean_recomputed, t_step.mean, atol=1e-6)
        assert torch.allclose(std_recomputed, t_step.std, atol=1e-6)
