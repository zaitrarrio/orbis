"""Streaming super-resolution refinement network.

A single-refinement, reference-aware upsampler: it takes a low-resolution frame
and the previously restored high-resolution frame (temporal reference) and
produces the restored frame at ``scale x`` resolution.  Frames within a chunk are
refined with full temporal context but there is *no* autoregressive dependency
across chunks -- exactly the within-chunk design the paper uses so restoration is
compatible with tiled / parallel execution.  This is a presentation stage: it
adds detail, it does not invent semantic state.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SRConfig, WorldConfig


class SuperResolution(nn.Module):
    def __init__(self, sr: SRConfig, world: WorldConfig):
        super().__init__()
        self.scale = sr.scale
        c, ch = world.channels, sr.base_channels
        # input: upsampled current LR frame (3) + temporal reference HR (3)
        self.net = nn.Sequential(
            nn.Conv2d(c * 2, ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch, c, 3, 1, 1),
        )

    def _upsample(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr, scale_factor=self.scale, mode="bilinear",
                             align_corners=False)

    def refine_frame(self, lr: torch.Tensor,
                     ref_hr: Optional[torch.Tensor]) -> torch.Tensor:
        up = self._upsample(lr)
        ref = up if ref_hr is None else ref_hr
        x = torch.cat([up, ref], dim=1)
        return torch.clamp(up + self.net(x), 0.0, 1.0)

    def forward(self, lr_chunk: torch.Tensor,
                prev_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``lr_chunk`` ``(B, T, 3, H, W)`` -> ``(B, T, 3, H*s, W*s)``.

        Temporal consistency: each frame is refined using the previous restored
        frame as reference, within the chunk.
        """
        b, t = lr_chunk.shape[:2]
        outs = []
        ref = prev_hr
        for i in range(t):
            hr = self.refine_frame(lr_chunk[:, i], ref)
            outs.append(hr)
            ref = hr
        return torch.stack(outs, dim=1)
