"""Tests for orbis.posttrain.flow_sde.FlowSDE (Phase 1b ODE->SDE conversion).

These are the correctness checks referenced in the module docstring: the
most important one is that the SDE step reduces *exactly* to the existing
deterministic Euler step (``RectifiedFlow.sample`` / ``orbis.distill._few_step``)
when eta=0, since that's the invariant the whole derivation rests on.
"""

from __future__ import annotations

import torch

from orbis.flow import RectifiedFlow
from orbis.posttrain.flow_sde import FlowSDE


def test_eta_zero_matches_deterministic_euler_step():
    torch.manual_seed(0)
    b = 3
    shape = (b, 2, 3)
    z = torch.randn(shape)
    sigma = torch.full((b,), 0.7)
    next_sigma = torch.full((b,), 0.4)
    v = torch.randn(shape)  # arbitrary "velocity" (constant fn stand-in)

    sde = FlowSDE(eta=0.0)
    z_next, mean, std = sde.step(z, sigma, next_sigma, v)

    # Deterministic Euler step from orbis/flow.py: z - dsigma * v
    dsigma = (sigma - next_sigma).view(-1, 1, 1)
    expected = z - dsigma * v

    assert torch.allclose(z_next, expected, atol=1e-6)
    assert torch.allclose(mean, expected, atol=1e-6)
    assert torch.allclose(std, torch.zeros_like(std), atol=1e-6)


def test_eta_zero_matches_rectified_flow_sample():
    """End-to-end: a full FlowSDE(eta=0) rollout matches RectifiedFlow.sample."""
    torch.manual_seed(1)
    shape = (2, 4)
    noise = torch.randn(shape)
    c = torch.full(shape, 0.3)  # constant velocity field

    flow = RectifiedFlow()
    ode_out = flow.sample(lambda z, s: c, shape, noise.device, steps=6, noise=noise)

    sde = FlowSDE(eta=0.0)
    sde_out, trace = sde.rollout(lambda z, s: c, shape, noise.device, steps=6, noise=noise)

    assert torch.allclose(ode_out, sde_out, atol=1e-5)
    assert all(torch.allclose(t.std, torch.zeros_like(t.std)) for t in trace)


def test_positive_eta_gives_nonzero_std_and_stochastic_rollout():
    torch.manual_seed(2)
    shape = (2, 4)
    noise = torch.randn(shape)
    c = torch.full(shape, 0.2)

    sde = FlowSDE(eta=0.5)
    out_a, trace_a = sde.rollout(lambda z, s: c, shape, noise.device, steps=4, noise=noise)
    out_b, trace_b = sde.rollout(lambda z, s: c, shape, noise.device, steps=4, noise=noise)

    # Same starting noise, but independent step-noise draws -> different paths.
    assert not torch.allclose(out_a, out_b)
    assert all((t.std > 0).all() for t in trace_a)  # sigma stays > 0 at every step here


def test_std_vanishes_as_sigma_approaches_zero():
    """g(sigma) = eta*sigma must drive std -> 0 smoothly, no singularity."""
    shape = (2, 3)
    z = torch.randn(shape)
    v = torch.randn(shape)
    sde = FlowSDE(eta=0.5)
    sigma = torch.full((2,), 1e-4)
    next_sigma = torch.zeros(2)
    mean, std = sde.mean_and_std(z, sigma, next_sigma, v)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert (std < 1e-3).all()


def test_log_prob_ratio_is_one_when_means_match():
    shape = (2, 3)
    z_next = torch.randn(shape)
    mean = torch.randn(shape)
    std = torch.full(shape, 0.1)
    logp_a = FlowSDE.log_prob(z_next, mean, std)
    logp_b = FlowSDE.log_prob(z_next, mean, std)
    ratio = torch.exp(logp_a - logp_b)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


def test_kl_zero_when_means_equal_positive_otherwise():
    shape = (2, 3)
    mean = torch.randn(shape)
    std = torch.full(shape, 0.2)
    kl_same = FlowSDE.gaussian_kl(mean, mean, std)
    assert torch.allclose(kl_same, torch.zeros_like(kl_same), atol=1e-6)

    other = mean + 1.0
    kl_diff = FlowSDE.gaussian_kl(mean, other, std)
    assert (kl_diff > 0).all()
