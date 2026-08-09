"""Stochastic reformulation of orbis's rectified-flow ODE, for policy-gradient RL.

Flow-GRPO (arXiv:2505.05470, "Flow-GRPO: Training Flow Matching Models via
Online RL") and DanceGRPO (arXiv:2505.07818, "DanceGRPO: Unleashing GRPO on
Visual Generation") both need a *stochastic* sampler with a tractable
per-step transition density, because GRPO's clipped-ratio surrogate (the PPO
objective of Schulman et al. 2017, adapted to GRPO's group-relative
advantage -- Shao et al. 2024, arXiv:2402.03300) requires an importance
ratio

    rho_t = p_theta(z_{t+1} | z_t) / p_theta_old(z_{t+1} | z_t)

between the *current* and a frozen *old* policy snapshot, evaluated along
the *same* sampled trajectory. A deterministic ODE step has no such density
(it is a Dirac delta), so both papers convert the flow-matching
probability-flow ODE into an equivalent SDE that has the *same marginals* at
every noise level, by injecting a synthetic diffusion term and correcting
the drift with the process's score function -- the standard construction
from Song et al. 2021 ("Score-Based Generative Modeling through Stochastic
Differential Equations", Appendix, the "probability flow ODE <-> reverse SDE"
equivalence).

This module re-derives that construction *specifically* for orbis's own
rectified-flow convention (see ``orbis/flow.py``) rather than transcribing
the source papers' ``(x_t, t)`` notation verbatim -- we only have
summarized/third-party excerpts of their exact equations, not a
machine-checked derivation, so a from-scratch, independently-checkable
re-derivation is the more defensible engineering choice (see the PR
description for the source citations, and ``tests/test_flow_sde.py`` for the
checks this derivation must satisfy).

Recap of orbis's convention (``orbis/flow.py::RectifiedFlow``):
    z_sigma = (1 - sigma) * z0 + sigma * eps         sigma: 1 (noise) -> 0 (data)
    v(z, sigma) = eps - z0                            (regression target)
    Euler ODE step (sigma decreasing): z_next = z - (sigma - sigma_next) * v

Derivation of the SDE
----------------------
From ``z = (1-sigma) z0 + sigma eps`` and ``v = eps - z0``, solve for the
posterior mean estimate (standard rectified-flow identities):
    z0_hat(z, sigma) = z - sigma * v
    eps_hat(z, sigma) = z + (1 - sigma) * v

Tweedie's formula for an affine-Gaussian forward channel ``z = alpha*z0 +
sigma*eps`` with ``alpha = (1 - sigma)`` gives the marginal score:
    score(z, sigma) = (alpha * z0_hat(z, sigma) - z) / sigma**2
Substituting ``z0_hat``:
    score(z, sigma) = ((1-sigma)(z - sigma*v) - z) / sigma**2
                     = -(z + (1 - sigma) * v) / sigma

Starting from the (zero-base-diffusion) deterministic probability-flow ODE
``dz = -v(z, sigma) dsigma`` and injecting a synthetic diffusion coefficient
``g(sigma)`` requires a ``+ g(sigma)**2 / 2 * score`` drift correction to
keep the marginals fixed (Song et al. 2021 Eq. 6, specialized to zero base
diffusion). In orbis's decreasing-sigma parameterization this gives the
reverse-time (denoising) SDE:

    dz = [ -v(z, sigma) - (g(sigma)**2 / 2) * (z + (1-sigma)*v(z,sigma)) / sigma ] dsigma
         + g(sigma) dW

We choose ``g(sigma) = eta * sigma`` (``eta``: a dimensionless exploration
scale) so that both the score correction and the injected noise vanish
smoothly as ``sigma -> 0``. This avoids the ``1/sigma`` singularity in the
score term and guarantees the SDE step reduces *exactly* to the existing
deterministic Euler step (``orbis.distill._few_step`` / ``RectifiedFlow.sample``)
as ``eta -> 0`` -- checked directly in ``tests/test_flow_sde.py``.

Discretizing with Euler-Maruyama over one rollout step (``sigma -> next_sigma``,
``d_sigma = sigma - next_sigma > 0``) gives a Gaussian transition kernel

    p_theta(z_next | z) = N(mean_theta(z, sigma), std(sigma)**2 * I)

with a closed-form log-density -- exactly the tractable-likelihood structure
GRPO's clipped-ratio objective needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SDEStep:
    """One recorded transition of a stochastic rollout.

    ``mean``/``std`` are the Gaussian transition parameters *used to sample*
    ``z_next`` (i.e. computed under whatever policy performed the rollout --
    the frozen "old" policy during a GRPO rollout). Storing ``mean`` avoids a
    second forward pass through the old policy when recomputing the
    importance ratio later.
    """

    z: torch.Tensor            # (B, ...) state before the step
    sigma: torch.Tensor        # (B,)
    next_sigma: torch.Tensor   # (B,)
    z_next: torch.Tensor       # (B, ...) sampled next state
    mean: torch.Tensor         # (B, ...) transition mean used to sample z_next
    std: torch.Tensor          # (B, ...) or (B, 1, 1, ...) transition std (broadcastable)


class FlowSDE:
    """SDE reformulation of :class:`orbis.flow.RectifiedFlow` for policy-gradient RL.

    ``eta=0`` recovers the exact deterministic Euler step (see module
    docstring); GRPO training requires ``eta > 0`` for a well-defined
    importance ratio.
    """

    def __init__(self, eta: float = 0.3, min_sigma: float = 1e-3):
        self.eta = eta
        self.min_sigma = min_sigma

    def _score(self, z: torch.Tensor, sigma: torch.Tensor,
               v: torch.Tensor) -> torch.Tensor:
        s = sigma.view(-1, *([1] * (z.dim() - 1))).clamp_min(self.min_sigma)
        return -(z + (1 - s) * v) / s

    def mean_and_std(self, z: torch.Tensor, sigma: torch.Tensor,
                      next_sigma: torch.Tensor,
                      v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form Gaussian transition ``N(mean, std**2)`` for one SDE step."""
        s = sigma.view(-1, *([1] * (z.dim() - 1)))
        d_sigma = (sigma - next_sigma).view(-1, *([1] * (z.dim() - 1)))
        g = self.eta * s  # noise coefficient g(sigma) = eta * sigma
        score = self._score(z, sigma, v)
        drift = -v - 0.5 * g.pow(2) * score
        mean = z + d_sigma * drift
        std = (g.pow(2).clamp_min(0) * d_sigma.clamp_min(0)).sqrt()
        return mean, std

    def step(self, z: torch.Tensor, sigma: torch.Tensor, next_sigma: torch.Tensor,
              v: torch.Tensor, noise: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one Euler-Maruyama step. Returns ``(z_next, mean, std)``."""
        mean, std = self.mean_and_std(z, sigma, next_sigma, v)
        if self.eta == 0.0:
            # std is exactly zero -> exact deterministic Euler step.
            return mean, mean, std
        if noise is None:
            noise = torch.randn_like(z)
        z_next = mean + std * noise
        return z_next, mean, std

    @staticmethod
    def log_prob(z_next: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                 eps: float = 1e-8) -> torch.Tensor:
        """Per-sample Gaussian log-density, summed over all non-batch dims."""
        var = std.pow(2).clamp_min(eps)
        logp = -0.5 * ((z_next - mean).pow(2) / var + math.log(2 * math.pi) + torch.log(var))
        return logp.flatten(1).sum(dim=1)

    @staticmethod
    def gaussian_kl(mean_p: torch.Tensor, mean_q: torch.Tensor, std: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
        """``KL(N(mean_p, std^2) || N(mean_q, std^2))`` for shared std (per-sample sum).

        Both distributions share ``std`` because the noise schedule ``g(sigma)``
        is a fixed design choice, not a function of the trainable policy --
        only the mean differs between the current and old policy.
        """
        var = std.pow(2).clamp_min(eps)
        kl = 0.5 * (mean_p - mean_q).pow(2) / var
        return kl.flatten(1).sum(dim=1)

    @torch.no_grad()
    def rollout(self, velocity_fn: VelocityFn, shape, device, steps: int,
                noise: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, List[SDEStep]]:
        """Sample one stochastic trajectory ``noise(sigma=1) -> data(sigma=0)``.

        Intended to be called under a frozen ("old") policy snapshot; the
        returned trace records everything GRPO needs to later recompute the
        importance ratio under the current (trainable) policy without
        re-sampling.
        """
        if noise is None:
            noise = torch.randn(shape, device=device)
        z = noise
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
        trace: List[SDEStep] = []
        for i in range(steps):
            s = sigmas[i].expand(shape[0])
            s_next = sigmas[i + 1].expand(shape[0])
            v = velocity_fn(z, s)
            z_next, mean, std = self.step(z, s, s_next, v)
            trace.append(SDEStep(z=z, sigma=s, next_sigma=s_next,
                                 z_next=z_next, mean=mean, std=std))
            z = z_next
        return z, trace
