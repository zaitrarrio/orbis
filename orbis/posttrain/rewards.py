"""Reward heads for GRPO post-training, including world-model consistency.

Referential integrity is scored via reference/history similarity in latent space
plus a latent world-model that predicts the next chunk from ``(H_k, M_k, c_k)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentWorldModel(nn.Module):
    """Predict next-chunk latents from pooled history, memory, and text."""

    def __init__(self, dim: int, latent_channels: int, chunk_frames: int,
                 latent_hw: tuple[int, int]):
        super().__init__()
        self.dim = dim
        lh, lw = latent_hw
        self.out_shape = (chunk_frames, latent_channels, lh, lw)
        out_dim = chunk_frames * latent_channels * lh * lw
        self.hist_pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.hist_proj = nn.Linear(latent_channels * 16, dim)
        self.mem_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), nn.SiLU(),
            nn.Linear(dim * 2, dim), nn.SiLU(),
            nn.Linear(dim, out_dim),
        )

    def forward(
        self,
        history: Optional[torch.Tensor],
        memory_state: Optional[torch.Tensor],
        pooled_text: torch.Tensor,
    ) -> torch.Tensor:
        b = pooled_text.shape[0]
        device = pooled_text.device
        if history is not None and history.numel() > 0:
            # (B,F,C,H,W) -> pool over F
            h = history.mean(dim=1)                       # (B,C,H,W)
            h = self.hist_pool(h.unsqueeze(2)).flatten(1)  # (B, C*16)
            h = self.hist_proj(h)
        else:
            h = torch.zeros(b, self.dim, device=device)
        if memory_state is not None:
            m = self.mem_proj(memory_state.mean(dim=1))
        else:
            m = torch.zeros(b, self.dim, device=device)
        t = self.text_proj(pooled_text)
        pred = self.net(torch.cat([h, m, t], dim=-1))
        return pred.view(b, *self.out_shape)


def visual_reward(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample negative MSE in latent space (higher is better)."""
    err = (pred - target).pow(2).flatten(1).mean(dim=1)
    return -err


def motion_reward(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Temporal difference agreement (higher is better)."""
    if pred.shape[1] < 2:
        return torch.zeros(pred.shape[0], device=pred.device)
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    err = (dp - dt).pow(2).flatten(1).mean(dim=1)
    return -err


def text_alignment_reward(
    pooled_text: torch.Tensor, chunk: torch.Tensor, proj: nn.Module
) -> torch.Tensor:
    """Cosine similarity between text pool and projected chunk."""
    c = chunk.flatten(1)
    e = proj(c)
    return F.cosine_similarity(pooled_text, e, dim=-1)


def reference_identity_reward(
    chunk: torch.Tensor, reference: Optional[torch.Tensor]
) -> torch.Tensor:
    """Encourage chunk appearance to stay near the anchored reference."""
    if reference is None or reference.numel() == 0:
        return torch.zeros(chunk.shape[0], device=chunk.device)
    ref = reference.mean(dim=1, keepdim=True)             # (B,1,C,H,W)
    # broadcast spatial if needed via adaptive pool
    if ref.shape[-2:] != chunk.shape[-2:]:
        ref = F.adaptive_avg_pool2d(
            ref.flatten(0, 1), chunk.shape[-2:]).view(
            chunk.shape[0], 1, chunk.shape[2], *chunk.shape[-2:])
    c = chunk.mean(dim=1, keepdim=True)
    err = (c - ref).pow(2).flatten(1).mean(dim=1)
    return -err


def world_model_consistency_reward(
    chunk: torch.Tensor, wm_pred: torch.Tensor
) -> torch.Tensor:
    """Primary long-horizon integrity lever: match world-model forecast."""
    return visual_reward(chunk, wm_pred.detach())


class RewardBundle(nn.Module):
    """Combines reward heads with consistency-heavy weights."""

    def __init__(self, dim: int, latent_channels: int, chunk_frames: int,
                 latent_hw: tuple[int, int],
                 w_visual: float = 0.2, w_motion: float = 0.15,
                 w_text: float = 0.15, w_ref: float = 0.25,
                 w_wm: float = 0.25):
        super().__init__()
        self.wm = LatentWorldModel(dim, latent_channels, chunk_frames, latent_hw)
        self.text_proj = nn.Linear(
            chunk_frames * latent_channels * latent_hw[0] * latent_hw[1], dim)
        self.w_visual = w_visual
        self.w_motion = w_motion
        self.w_text = w_text
        self.w_ref = w_ref
        self.w_wm = w_wm

    def forward(
        self,
        chunk: torch.Tensor,
        target: torch.Tensor,
        pooled_text: torch.Tensor,
        history: Optional[torch.Tensor],
        memory_state: Optional[torch.Tensor],
        reference: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        wm_pred = self.wm(history, memory_state, pooled_text)
        parts = {
            "visual": visual_reward(chunk, target),
            "motion": motion_reward(chunk, target),
            "text": text_alignment_reward(pooled_text, chunk, self.text_proj),
            "ref": reference_identity_reward(chunk, reference),
            "wm": world_model_consistency_reward(chunk, wm_pred),
        }
        total = (
            self.w_visual * parts["visual"]
            + self.w_motion * parts["motion"]
            + self.w_text * parts["text"]
            + self.w_ref * parts["ref"]
            + self.w_wm * parts["wm"]
        )
        return total, {k: v.detach().mean() for k, v in parts.items()}
