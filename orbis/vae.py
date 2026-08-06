"""Small convolutional latent autoencoder.

Orbis generates in a *latent* video space; this module is the toy-scale analogue
of that VAE.  It is spatial (per-frame): temporal structure is modelled by the
chunk-wise flow generator, not here.  A deterministic autoencoder (no sampling)
is sufficient at this scale.  After training we cache a global latent scale so
the rectified-flow model operates on roughly unit-variance latents.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VAEConfig, WorldConfig


class ConvVAE(nn.Module):
    def __init__(self, vae: VAEConfig, world: WorldConfig):
        super().__init__()
        c, ch = world.channels, vae.base_channels
        lc = vae.latent_channels
        self.latent_channels = lc
        self.encoder = nn.Sequential(
            nn.Conv2d(c, ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch, ch, 4, 2, 1), nn.SiLU(),      # /2
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(),  # /4
            nn.Conv2d(ch * 2, lc, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(lc, ch * 2, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch * 2, ch, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch, c, 3, 1, 1),
        )
        # Latent normalisation scale, populated by ``calibrate`` after training.
        self.register_buffer("latent_scale", torch.ones(1))

    # -- frame tensors are (B, 3, H, W) in [0,1] ------------------------------
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames) / self.latent_scale

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(latent * self.latent_scale))

    def forward(self, frames: torch.Tensor):
        z = self.encode(frames)
        return self.decode(z), z

    @torch.no_grad()
    def calibrate(self, frames: torch.Tensor):
        """Set ``latent_scale`` to the std of raw encoder outputs."""
        self.latent_scale.fill_(1.0)
        z = self.encoder(frames)
        self.latent_scale.fill_(float(z.std()) + 1e-6)


def frames_to_tensor(frames) -> torch.Tensor:
    """``(..., H, W, 3)`` float array/tensor -> ``(..., 3, H, W)`` tensor."""
    t = torch.as_tensor(frames, dtype=torch.float32)
    return t.movedim(-1, -3)


def tensor_to_frames(t: torch.Tensor):
    """``(..., 3, H, W)`` -> ``(..., H, W, 3)`` numpy array."""
    return t.movedim(-3, -1).detach().cpu().numpy()
