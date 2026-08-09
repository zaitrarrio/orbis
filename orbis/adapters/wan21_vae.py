"""Real Wan2.1 VAE: frozen, pretrained ``AutoencoderKLWan`` behind the
``ConvVAE`` per-frame contract.

This closes the "latent distribution mismatch" fidelity gap documented in
``orbis/adapters/wan21_real.py``'s module docstring: ``RealWanBackbone``
loads Wan's real, frozen ``WanTransformer3DModel``, but until now it was
still fed latents from orbis's own independently-trained ``ConvVAE``
(``orbis/vae.py``) rather than the exact latent space Wan's transformer was
actually pretrained against.

Design (kept deliberately narrow -- see "explicitly scoped out" below):

* Wan's real ``AutoencoderKLWan`` is a genuine causal 3D VAE with temporal
  compression (``scale_factor_temporal=4`` via
  ``temperal_downsample=[False, True, True]``) plus 8x spatial compression,
  operating on ``(B, 3, T, H, W)`` pixel tensors -- fundamentally different
  from ``ConvVAE``'s per-frame-only, spatial-only design. Verified directly
  against the installed ``diffusers`` build on a real GPU box: encoding a
  single frame as a ``T=1`` "video" (``(B, 3, 1, H, W)``) always yields
  exactly one temporal latent frame (Wan's causal grouping is
  ``num_latent_frames = (T - 1) // 4 + 1``, so ``T=1`` never triggers any
  cross-frame temporal compression) with shape ``(B, 16, 1, H/8, W/8)`` --
  i.e. per-frame T=1 encode/decode is a lossless-shape special case of the
  real Wan VAE, not an approximation of it.
* This wrapper therefore calls ``AutoencoderKLWan`` **per-frame** (each
  frame reshaped to a ``T=1`` clip, then the temporal axis squeezed back
  out), preserving the exact same ``encode(frames) -> latent`` /
  ``decode(latent) -> frames`` per-frame interface every call site already
  depends on (``orbis/engine.py``, ``orbis/train.py``, ``orbis/dataset.py``,
  ``orbis/data/video_dataset.py``, ``orbis/posttrain/grpo.py``) -- none of
  them need to change. Genuine multi-frame temporal VAE compression across
  chunks (using ``T>1`` and letting Wan's causal temporal downsampling
  actually engage) is a materially larger, riskier rewrite of the chunk
  pipeline and is intentionally left for a further-deferred phase, per this
  repo's "minimize blast radius" / "document rather than silently gloss
  over" conventions.
* Wan's own diffusion transformer is pretrained on a per-channel
  *normalized* latent space, not the VAE's raw encoder output. The exact
  transform (confirmed by reading ``diffusers``' own
  ``WanPipeline``/``WanImageToVideoPipeline`` source on the GPU box) is::

      normalized = (raw - latents_mean) / latents_std   # encode direction
      raw        = normalized * latents_std + latents_mean  # decode direction

  using the real per-channel ``config.latents_mean`` / ``config.latents_std``
  (16 values each, shipped with the pretrained checkpoint) -- NOT an
  empirically-calibrated scalar the way ``ConvVAE.calibrate()`` works. This
  wrapper applies that normalization inside ``encode``/``decode`` so the
  rest of orbis (which was written against ``ConvVAE``'s "roughly
  unit-variance latents" contract) transparently gets Wan's *real*
  pretrained latent space instead.
* The VAE is off-the-shelf pretrained and kept fully frozen -- no LoRA is
  applied to it (mirrors how the real Wan2.1 pipeline itself always uses a
  frozen VAE). ``state_dict()``/``load_state_dict()`` are overridden to be
  empty/no-op (mirroring ``RealWanBackbone``'s lean-checkpoint pattern):
  nothing trainable lives here, so nothing needs to round-trip through
  orbis checkpoints -- a fresh ``from_pretrained()`` always reconstructs an
  identical frozen VAE.

Requires the optional ``wan`` extra (``uv sync --extra wan``) and, beyond
CPU shape tests, a real GPU -- see ``deploy/README.md``. CPU unit tests
exercise this against a small mock that mirrors ``AutoencoderKLWan``'s
public ``encode``/``decode``/``config`` surface (see
``tests/test_wan21_vae_adapter.py``), following the same mock-transformer
convention as ``tests/test_wan21_real_adapter.py``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RealWanVAE(nn.Module):
    """Wraps a (real or mock) ``AutoencoderKLWan``-shaped model behind the
    exact ``ConvVAE`` per-frame contract: ``encode(frames)``/``decode(latent)``
    on ``(N, 3, H, W)`` frames in ``[0, 1]`` <-> ``(N, C, H/8, W/8)`` latents.
    """

    def __init__(self, model: nn.Module, latent_channels: int):
        super().__init__()
        # Held as a plain submodule (not excluded from the module tree) but
        # frozen and never serialized -- see state_dict()/load_state_dict().
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = model.config
        z_dim = int(getattr(cfg, "z_dim", latent_channels))
        if z_dim != latent_channels:
            raise ValueError(
                f"RealWanVAE: pretrained AutoencoderKLWan z_dim={z_dim} does "
                f"not match VAEConfig.latent_channels={latent_channels}. "
                "wan21_real_config() sets latent_channels=16 to match Wan's "
                "native VAE -- check for a config mismatch.")
        self.latent_channels = z_dim
        self.downsample = int(getattr(cfg, "scale_factor_spatial", 8))

        mean = torch.tensor(list(cfg.latents_mean), dtype=torch.float32).view(1, z_dim, 1, 1)
        std = torch.tensor(list(cfg.latents_std), dtype=torch.float32).view(1, z_dim, 1, 1)
        self.register_buffer("latents_mean", mean)
        self.register_buffer("latents_std", std)
        # Kept for API parity with ConvVAE; unused (normalization here uses
        # the pretrained per-channel latents_mean/std, not a scalar).
        self.register_buffer("latent_scale", torch.ones(1))

    # -- frame tensors are (N, 3, H, W) in [0, 1], same contract as ConvVAE --
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames * 2.0 - 1.0            # [0,1] -> [-1,1] (Wan's pixel range)
        x = x.unsqueeze(2)                 # (N, 3, H, W) -> (N, 3, T=1, H, W)
        out = self.model.encode(x)
        raw = out.latent_dist.mode() if hasattr(out, "latent_dist") else out
        raw = raw.squeeze(2)               # (N, C, 1, h, w) -> (N, C, h, w)
        mean = self.latents_mean.to(raw.dtype)
        std = self.latents_std.to(raw.dtype)
        return (raw - mean) / std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(latent.dtype)
        std = self.latents_std.to(latent.dtype)
        raw = latent * std + mean
        raw = raw.unsqueeze(2)             # (N, C, h, w) -> (N, C, T=1, h, w)
        out = self.model.decode(raw)
        frames = out.sample if hasattr(out, "sample") else out
        frames = frames.squeeze(2)         # (N, 3, 1, H, W) -> (N, 3, H, W)
        frames = (frames + 1.0) / 2.0      # [-1,1] -> [0,1]
        return frames.clamp(0.0, 1.0)

    def forward(self, frames: torch.Tensor):
        z = self.encode(frames)
        return self.decode(z), z

    def calibrate(self, frames: torch.Tensor) -> None:
        """No-op: normalization uses the pretrained latents_mean/std, not an
        empirically-calibrated scalar. Kept for API parity with ConvVAE so
        any generic caller of ``vae.calibrate(...)`` doesn't need a branch.
        Note ``orbis.train.train_vae`` does NOT call this method (it
        manipulates ``ConvVAE``-specific internals -- ``.encoder``,
        ``.latent_scale`` -- directly) and must be skipped entirely for this
        VAE; see the ``real_vae`` guard in ``scripts/train-live-wan.py``.
        """
        return

    # -- checkpoint IO: nothing trainable lives here -------------------------
    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {}

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        # Frozen pretrained weights always come from from_pretrained(); a
        # fresh instance is bit-identical, so there is nothing to load.
        return

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        latent_channels: int,
        subfolder: str = "vae",
        dtype: Optional[torch.dtype] = None,
    ) -> "RealWanVAE":
        """Load the real, frozen ``AutoencoderKLWan`` from Hugging Face.

        This path is exercised on a RunPod/Vast.ai GPU box, not by CPU unit
        tests -- see ``tests/test_wan21_vae_adapter.py`` for the mock-model
        coverage of everything in this class besides the HF download.
        """
        try:
            from diffusers import AutoencoderKLWan  # type: ignore
        except ImportError as e:
            raise ImportError(
                "RealWanVAE.from_pretrained requires the 'wan' extra: "
                "`uv sync --extra wan` (diffusers>=0.32).") from e

        model = AutoencoderKLWan.from_pretrained(
            checkpoint_path, subfolder=subfolder,
            torch_dtype=dtype or torch.float32)
        return cls(model=model, latent_channels=latent_channels)
