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

from typing import Callable, Optional

import torch
import torch.nn.functional as F


class RectifiedFlow:
    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def sample_sigma(self, batch: int, device,
                     power: float = 1.0) -> torch.Tensor:
        u = torch.rand(batch, device=device)
        if power != 1.0:
            u = u.pow(power)  # power>1 → more mass near σ≈0
        return u.clamp(self.eps, 1.0)

    def interpolate(self, z: torch.Tensor, noise: torch.Tensor,
                    sigma: torch.Tensor) -> torch.Tensor:
        s = sigma.view(-1, *([1] * (z.dim() - 1)))
        return (1 - s) * z + s * noise

    @staticmethod
    def target(z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - z

    def loss(self, model_velocity: torch.Tensor, z: torch.Tensor,
             noise: torch.Tensor,
             spatial_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        err = (model_velocity - self.target(z, noise)).pow(2)
        if spatial_weight is None:
            return err.mean()
        w = spatial_weight
        while w.dim() < err.dim():
            w = w.unsqueeze(2)
        # Proper weighted mean so sparse FG mass is not drowned by BG count.
        return (w * err).sum() / w.sum().clamp_min(1e-6)

    @staticmethod
    def latent_fg_weight(z: torch.Tensor, fg_boost: float = 24.0,
                         bg_weight: float = 1.5) -> torch.Tensor:
        """Per-location weights from clean-latent energy (B,F,1,H,W).

        FG is boosted for shape fidelity; BG stays ≥1 so residual speckles
        are still penalized (under-weighting BG left salt-and-pepper noise).
        """
        energy = z.detach().pow(2).mean(dim=2, keepdim=True)
        flat = energy.reshape(energy.shape[0], -1)
        med = flat.median(dim=1).values.view(-1, *([1] * (energy.dim() - 1)))
        fg = energy > (med * 1.5 + 1e-6)
        return torch.where(fg, torch.full_like(energy, fg_boost),
                           torch.full_like(energy, bg_weight))

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


def pixel_fg_weight_from_latents(vae, z: torch.Tensor, fg_boost: float = 24.0,
                                 bg_weight: float = 1.5) -> torch.Tensor:
    """Decode clean latents and pool an FG mask to latent resolution."""
    b, f, c, lh, lw = z.shape
    with torch.no_grad():
        frames = vae.decode(z.reshape(b * f, c, lh, lw))
        bg = frames.amin(dim=(2, 3), keepdim=True)
        fg = ((frames - bg).amax(dim=1, keepdim=True) > 0.08).float()
        # Soft dilation so the object neighborhood is also supervised.
        fg = F.max_pool2d(fg, kernel_size=5, stride=1, padding=2)
        fg_lat = F.adaptive_max_pool2d(fg, (lh, lw))
        fg_lat = fg_lat.reshape(b, f, 1, lh, lw)
    return torch.where(fg_lat > 0.5,
                       torch.full_like(fg_lat, fg_boost),
                       torch.full_like(fg_lat, bg_weight))
