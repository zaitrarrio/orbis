"""Rectified-flow objective and sampler (paper Eq. 1).

Straight-line interpolation between data ``z`` (sigma=0) and noise ``eps``
(sigma=1)::

    z_sigma = (1 - sigma) * z + sigma * eps
    target  = eps - z                      # constant velocity along the path

The model regresses this velocity; sampling integrates it from noise back to
data with an explicit Euler solver.  ``teacher_steps`` gives the accurate
many-step trajectory used in pretraining/distillation; ``student_steps`` is the
few-step real-time path.
"""

from __future__ import annotations

from typing import Callable

import torch


class RectifiedFlow:
    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def sample_sigma(self, batch: int, device) -> torch.Tensor:
        return torch.rand(batch, device=device).clamp(self.eps, 1.0)

    def interpolate(self, z: torch.Tensor, noise: torch.Tensor,
                    sigma: torch.Tensor) -> torch.Tensor:
        s = sigma.view(-1, *([1] * (z.dim() - 1)))
        return (1 - s) * z + s * noise

    @staticmethod
    def target(z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - z

    def loss(self, model_velocity: torch.Tensor, z: torch.Tensor,
             noise: torch.Tensor) -> torch.Tensor:
        err = (model_velocity - self.target(z, noise)).pow(2)
        # Upweight latents that carry signal so dark background does not dominate
        # at large resolutions (same motivation as VAE fg_weight).
        w = 1.0 + 6.0 * z.detach().abs().mean(dim=2, keepdim=True)
        return (w * err).mean()

    @torch.no_grad()
    def sample(self, velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               shape, device, steps: int, noise: torch.Tensor = None,
               return_trajectory: bool = False):
        """Integrate noise (sigma=1) -> data (sigma=0) with Euler steps."""
        if noise is None:
            noise = torch.randn(shape, device=device)
        z = noise
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
        traj = [z]
        for i in range(steps):
            s = sigmas[i].expand(shape[0])
            v = velocity_fn(z, s)
            dsigma = sigmas[i] - sigmas[i + 1]
            z = z - dsigma * v
            if return_trajectory:
                traj.append(z)
        return (z, traj) if return_trajectory else z
