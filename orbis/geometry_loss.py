"""Explicit geometry supervision for structure-first training.

Soft FG mask Dice and centroid L2 pin *where* mass belongs. Color/RF losses
alone do not.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_fg_energy(frames: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``(B,3,H,W)`` -> soft FG energy ``(B,1,H,W)`` in [0,1]."""
    bg = frames.amin(dim=(2, 3), keepdim=True)
    energy = (frames - bg).amax(dim=1, keepdim=True)
    peak = energy.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    return energy / peak


def soft_dice_loss(pred_fg: torch.Tensor, gt_fg: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice between predicted and GT FG maps (any broadcastable shape)."""
    inter = (pred_fg * gt_fg).sum()
    return 1.0 - (2.0 * inter) / (pred_fg.sum() + gt_fg.sum() + eps)


def centroid_hw(fg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``(B,1,H,W)`` -> centroids ``(B,2)`` as (y, x) in pixel coords."""
    b, _, h, w = fg.shape
    mass = fg.reshape(b, -1).sum(dim=1).clamp_min(eps)
    ys = torch.linspace(0, h - 1, h, device=fg.device, dtype=fg.dtype)
    xs = torch.linspace(0, w - 1, w, device=fg.device, dtype=fg.dtype)
    gy = (fg * ys.view(1, 1, h, 1)).sum(dim=(1, 2, 3)) / mass
    gx = (fg * xs.view(1, 1, 1, w)).sum(dim=(1, 2, 3)) / mass
    return torch.stack([gy, gx], dim=-1)


def centroid_loss(pred_fg: torch.Tensor, gt_fg: torch.Tensor) -> torch.Tensor:
    """Normalized L2 between soft centroids (scale-invariant to H,W)."""
    c_p = centroid_hw(pred_fg)
    c_g = centroid_hw(gt_fg)
    _, _, h, w = pred_fg.shape
    scale = torch.tensor([h, w], device=pred_fg.device, dtype=pred_fg.dtype)
    return ((c_p - c_g) / scale).pow(2).mean()


def geometry_aux(pred_frames: torch.Tensor, gt_frames: torch.Tensor,
                 dice_weight: float = 1.0,
                 centroid_weight: float = 0.5) -> torch.Tensor:
    """Combined geometry loss on ``(B,3,H,W)`` (or flattened BF) frames."""
    pred_fg = soft_fg_energy(pred_frames)
    with torch.no_grad():
        gt_fg = soft_fg_energy(gt_frames)
        # Harden GT slightly so supervision focuses on the object core.
        gt_fg = (gt_fg > 0.15).float()
    return (dice_weight * soft_dice_loss(pred_fg, gt_fg)
            + centroid_weight * centroid_loss(pred_fg, gt_fg))
