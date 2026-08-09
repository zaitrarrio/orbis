"""Tests for the cross-chunk boundary-continuity reward.

``CompositeReward``'s pre-existing ``w_motion`` term only measures
frame-to-frame smoothness *inside* a rolled-out chunk -- it is blind to a
discontinuous jump right at the seam between the last committed frame of
``history`` and the new chunk's first frame. These tests cover the new
``boundary_continuity_reward`` function directly (CPU-only, no GPU/network)
and its wiring into ``CompositeReward``/``grpo_align``.
"""

from __future__ import annotations

import torch

from orbis.config import wan_smoke_config
from orbis.dataset import RolloutSampler
from orbis.posttrain.grpo import grpo_align
from orbis.posttrain.reward_models import CompositeReward, MockReward
from orbis.posttrain.rewards import boundary_continuity_reward
from orbis.system import OrbisSystem


# -- boundary_continuity_reward: pure function -------------------------------

def test_boundary_reward_zero_for_perfectly_continuous_seam():
    """No jump at the seam -> zero penalty (max possible reward)."""
    history = torch.randn(2, 3, 4, 5, 5)
    chunk = history[:, -1:].repeat(1, 6, 1, 1, 1).clone()  # first frame == last history frame
    r = boundary_continuity_reward(chunk, history, n_frames=1)
    assert r.shape == (2,)
    assert torch.allclose(r, torch.zeros(2), atol=1e-6)


def test_boundary_reward_penalizes_discontinuous_jump():
    """A large jump at the seam must score strictly worse than continuity."""
    torch.manual_seed(0)
    history = torch.randn(2, 3, 4, 5, 5)
    continuous_chunk = history[:, -1:].repeat(1, 6, 1, 1, 1).clone()
    jumpy_chunk = continuous_chunk.clone()
    jumpy_chunk[:, 0] += 10.0  # large discontinuity at the very first frame

    r_continuous = boundary_continuity_reward(continuous_chunk, history)
    r_jumpy = boundary_continuity_reward(jumpy_chunk, history)
    assert torch.all(r_jumpy < r_continuous)


def test_boundary_reward_larger_jump_scores_worse():
    """Reward must be monotonic in seam discontinuity magnitude, not just sign-aware."""
    torch.manual_seed(1)
    history = torch.randn(1, 2, 4, 5, 5)
    base = history[:, -1:].repeat(1, 4, 1, 1, 1).clone()
    small_jump = base.clone()
    small_jump[:, 0] += 1.0
    big_jump = base.clone()
    big_jump[:, 0] += 5.0

    r_small = boundary_continuity_reward(small_jump, history)
    r_big = boundary_continuity_reward(big_jump, history)
    assert r_big.item() < r_small.item()


def test_boundary_reward_handles_missing_history():
    """First chunk of a session (no history yet) must return zeros, not error."""
    chunk = torch.randn(3, 4, 2, 5, 5)
    r_none = boundary_continuity_reward(chunk, None)
    r_empty = boundary_continuity_reward(chunk, torch.empty(3, 0, 2, 5, 5))
    assert torch.equal(r_none, torch.zeros(3))
    assert torch.equal(r_empty, torch.zeros(3))


def test_boundary_reward_respects_n_frames():
    """n_frames=2 should also weigh the second-to-last history frame."""
    torch.manual_seed(2)
    history = torch.randn(1, 3, 2, 4, 4)
    chunk = torch.randn(1, 3, 2, 4, 4)

    r_one = boundary_continuity_reward(chunk, history, n_frames=1)
    r_two = boundary_continuity_reward(chunk, history, n_frames=2)
    # Different frame windows compared -> generally different reward values.
    assert not torch.allclose(r_one, r_two)


def test_boundary_reward_handles_spatial_mismatch():
    """Defensive resize path (adaptive pool) must not crash on shape mismatch."""
    history = torch.randn(2, 2, 4, 8, 8)
    chunk = torch.randn(2, 5, 4, 4, 4)
    r = boundary_continuity_reward(chunk, history, n_frames=1)
    assert r.shape == (2,)
    assert torch.isfinite(r).all()


# -- CompositeReward wiring ---------------------------------------------------

def test_composite_reward_boundary_term_disabled_by_default():
    """w_boundary defaults to 0.0 -- opt-in, matching w_reference/w_motion."""
    reward = CompositeReward([(MockReward(), 1.0)])
    frames = torch.randn(2, 4, 3, 8, 8)
    chunk = torch.randn(2, 4, 2, 5, 5)
    history = torch.randn(2, 2, 2, 5, 5)
    total, detail = reward(frames, latent_chunk=chunk, history=history)
    assert "boundary_smooth" not in detail


def test_composite_reward_boundary_term_appears_when_enabled():
    reward = CompositeReward([(MockReward(), 1.0)], w_boundary=0.2)
    frames = torch.randn(2, 4, 3, 8, 8)
    chunk = torch.randn(2, 4, 2, 5, 5)
    history = torch.randn(2, 2, 2, 5, 5)
    total, detail = reward(frames, latent_chunk=chunk, history=history)
    assert "boundary_smooth" in detail
    assert torch.isfinite(total).all()


def test_composite_reward_boundary_term_requires_history():
    """w_boundary>0 but no history passed (e.g. very first chunk) -> no crash, no term."""
    reward = CompositeReward([(MockReward(), 1.0)], w_boundary=0.2)
    frames = torch.randn(2, 4, 3, 8, 8)
    chunk = torch.randn(2, 4, 2, 5, 5)
    total, detail = reward(frames, latent_chunk=chunk, history=None)
    assert "boundary_smooth" not in detail
    assert torch.isfinite(total).all()


def test_composite_reward_boundary_term_moves_total_reward():
    """Enabling the term must actually change the scored total for a jumpy chunk."""
    torch.manual_seed(4)
    history = torch.randn(1, 2, 2, 5, 5)
    continuous_chunk = history[:, -1:].repeat(1, 4, 1, 1, 1).clone()
    jumpy_chunk = continuous_chunk.clone()
    jumpy_chunk[:, 0] += 10.0
    frames = torch.randn(1, 4, 3, 8, 8)

    reward = CompositeReward([(MockReward(), 1.0)], w_boundary=1.0)
    total_continuous, _ = reward(frames, latent_chunk=continuous_chunk, history=history)
    total_jumpy, _ = reward(frames, latent_chunk=jumpy_chunk, history=history)
    assert total_jumpy.item() < total_continuous.item()


# -- End-to-end: grpo_align actually plumbs history into the reward call -----

def _make_system_and_batch_fn(seed: int = 0):
    cfg = wan_smoke_config(seed=seed)
    cfg.grpo.reward_backend = "mock"
    system = OrbisSystem.build(cfg)
    sampler = RolloutSampler(cfg, system.vae, seed=cfg.seed + 20)

    def batch_fn():
        return sampler.training_batch(2, mode="history", memory_context_chunks=1)

    return system, batch_fn


def test_grpo_align_logs_boundary_term_when_enabled():
    """grpo_align must pass `history=` through to the reward model, and the
    resulting detail dict (surfaced via log_cb) must include boundary_smooth
    once cfg.grpo.w_boundary > 0 -- catches any future call-site regression
    that silently drops the history= kwarg."""
    system, batch_fn = _make_system_and_batch_fn(seed=5)
    logs = []
    grpo_align(
        system, batch_fn, steps=1, group_size=2, lr=1e-4,
        eta=0.5, denoise_steps=2,
        reward_model=CompositeReward([(MockReward(), 1.0)],
                                     w_reference=0.0, w_motion=0.0,
                                     w_boundary=0.3, boundary_frames=1),
        log_cb=logs.append,
    )
    assert any("boundary_smooth" in line for line in logs)


def test_grpo_align_runs_with_default_boundary_weight():
    """The new default (w_boundary=0.15 in GRPOConfig) must not break a
    normal grpo_align step end to end."""
    system, batch_fn = _make_system_and_batch_fn(seed=6)
    reward = grpo_align(
        system, batch_fn, steps=1, group_size=2, lr=1e-4,
        eta=0.5, denoise_steps=2,
    )
    assert isinstance(reward, CompositeReward)
    assert reward.w_boundary == system.cfg.grpo.w_boundary
