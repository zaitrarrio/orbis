"""Chunk-wise streaming rectified-flow generator.

Realises ``p_theta(z_k | H_k, c_k, r_k, M_k)`` from the paper: the velocity field
for the current latent chunk ``z_k`` conditioned on

* ``c_k`` -- the active text instruction (token cross-attention + pooled AdaLN);
* ``H_k`` -- a bounded window of committed latent frames (recent history);
* ``r_k`` -- optional reference frames (image/video prefix for I2V / V2V);
* ``M_k`` -- fixed-capacity persistent memory tokens.

All context is clean; only the chunk tokens are noised.  The assembled sequence
is ``[memory | reference | history | chunk]`` with full attention among tokens;
causality *across* chunks is enforced by the rollout (a chunk only ever sees
past history), matching the paper's causal rollout law.

Context that does not change across the many solver steps of one chunk is
encoded once into a :class:`ModelContext` and reused -- the toy analogue of the
paper's versioned state reuse / compiled context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig, VAEConfig
from .memory import MemoryBank
from .modules import DiTBlock, FinalLayer, timestep_embedding
from .vocab import VOCAB_SIZE

# role ids for the role-embedding table
ROLE_MEMORY, ROLE_REF, ROLE_HISTORY, ROLE_CHUNK = 0, 1, 2, 3
_MAX_FRAMES = 64


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """``(B, F, C, H, W)`` -> ``(B, F, gh*gw, C*p*p)``."""
    b, f, c, h, w = x.shape
    gh, gw = h // p, w // p
    x = x.reshape(b, f, c, gh, p, gw, p)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, f, gh * gw, c * p * p)
    return x


def unpatchify(x: torch.Tensor, p: int, c: int, gh: int, gw: int) -> torch.Tensor:
    """``(B, F, gh*gw, C*p*p)`` -> ``(B, F, C, gh*p, gw*p)``."""
    b, f = x.shape[:2]
    x = x.reshape(b, f, gh, gw, c, p, p)
    x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(b, f, c, gh * p, gw * p)
    return x


@dataclass
class ModelContext:
    ctx_tokens: torch.Tensor      # (B, Nctx, dim) memory+ref+history, embedded
    text_kv: torch.Tensor         # (B, text_len, dim)
    pooled_text: torch.Tensor     # (B, dim)
    batch: int


class Generator(nn.Module):
    def __init__(self, cfg: ModelConfig, vae: VAEConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        p = cfg.patch_size
        self.patch = p
        self.latent_channels = vae.latent_channels
        self.patch_dim = vae.latent_channels * p * p

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        # grid size depends on latent hw; created lazily on first use
        self.spatial_pos = None
        self.temporal_pos = nn.Embedding(_MAX_FRAMES, dim)
        self.role_embed = nn.Embedding(4, dim)

        self.text_embed = nn.Embedding(VOCAB_SIZE, dim, padding_idx=0)
        self.text_pos = nn.Parameter(torch.randn(cfg.text_len, dim) * 0.02)

        self.cond_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))

        self.memory = MemoryBank(dim, cfg.memory_tokens, cfg.heads)

        self.blocks = nn.ModuleList(
            [DiTBlock(dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.final = FinalLayer(dim, self.patch_dim)

    # -- helpers --------------------------------------------------------------
    def build(self, latent_hw):
        """Eagerly create the spatial position table so it is registered with
        the module *before* an optimizer is constructed over its parameters."""
        lh, lw = latent_hw
        gh, gw = lh // self.patch, lw // self.patch
        if self.spatial_pos is None or self.spatial_pos.shape[0] != gh * gw:
            self.spatial_pos = nn.Parameter(
                torch.randn(gh * gw, self.cfg.dim) * 0.02)
        return self

    def _ensure_spatial(self, gh: int, gw: int, device):
        if self.spatial_pos is None or self.spatial_pos.shape[0] != gh * gw:
            self.spatial_pos = nn.Parameter(
                torch.randn(gh * gw, self.cfg.dim, device=device) * 0.02)

    def frame_tokens(self, latents: torch.Tensor, role: int,
                     frame_offset: int = 0) -> torch.Tensor:
        """Embed latent frames ``(B, F, C, lh, lw)`` -> tokens ``(B, F*grid, dim)``."""
        b, f, c, lh, lw = latents.shape
        p = self.patch
        gh, gw = lh // p, lw // p
        self._ensure_spatial(gh, gw, latents.device)
        tok = self.patch_embed(patchify(latents, p))          # (B,F,grid,dim)
        tok = tok + self.spatial_pos.view(1, 1, gh * gw, -1)
        idx = torch.arange(frame_offset, frame_offset + f, device=latents.device)
        tok = tok + self.temporal_pos(idx).view(1, f, 1, -1)
        tok = tok + self.role_embed.weight[role].view(1, 1, 1, -1)
        return tok.reshape(b, f * gh * gw, -1)

    def encode_context(self, text_ids: torch.Tensor,
                       history: Optional[torch.Tensor],
                       reference: Optional[torch.Tensor],
                       memory_state: Optional[torch.Tensor]) -> ModelContext:
        b = text_ids.shape[0]
        device = text_ids.device
        text_kv = self.text_embed(text_ids) + self.text_pos.unsqueeze(0)
        pooled = text_kv.mean(dim=1)

        groups = []
        if memory_state is not None:
            m = self.memory.read(memory_state)
            m = m + self.role_embed.weight[ROLE_MEMORY].view(1, 1, -1)
            groups.append(m)
        if reference is not None and reference.shape[1] > 0:
            groups.append(self.frame_tokens(reference, ROLE_REF))
        if history is not None and history.shape[1] > 0:
            groups.append(self.frame_tokens(history, ROLE_HISTORY))
        ctx = torch.cat(groups, dim=1) if groups else \
            torch.zeros(b, 0, self.cfg.dim, device=device)
        return ModelContext(ctx_tokens=ctx, text_kv=text_kv,
                            pooled_text=pooled, batch=b)

    def forward(self, z_noised: torch.Tensor, sigma: torch.Tensor,
                context: ModelContext) -> torch.Tensor:
        """Predict the velocity field for the noised chunk ``z_noised``."""
        b, f, c, lh, lw = z_noised.shape
        p = self.patch
        gh, gw = lh // p, lw // p
        chunk_tok = self.frame_tokens(z_noised, ROLE_CHUNK)
        n_chunk = chunk_tok.shape[1]

        x = torch.cat([context.ctx_tokens, chunk_tok], dim=1)

        cond = self.cond_mlp(timestep_embedding(sigma, self.cfg.dim)
                             .to(x.dtype)) + context.pooled_text
        for blk in self.blocks:
            x = blk(x, cond, context.text_kv)
        chunk_out = self.final(x[:, -n_chunk:], cond)         # (B, n_chunk, patch_dim)
        chunk_out = chunk_out.reshape(b, f, gh * gw, self.patch_dim)
        return unpatchify(chunk_out, p, c, gh, gw)

    # convenience for training: build context + predict in one call
    def velocity(self, z_noised, sigma, text_ids, history=None, reference=None,
                 memory_state=None):
        ctx = self.encode_context(text_ids, history, reference, memory_state)
        return self.forward(z_noised, sigma, ctx)
