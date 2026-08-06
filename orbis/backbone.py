"""VideoBackbone protocol shared by the toy Generator and Wan adapters.

The live engine depends only on this surface: encode context once per chunk,
reuse it across solver steps, predict rectified-flow velocity, and consolidate
evicted frames into a bounded MemoryBank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from .config import ModelConfig
from .memory import MemoryBank


@dataclass
class ModelContext:
    """Opaque per-chunk context; immutable across Euler steps of one chunk."""

    ctx_tokens: torch.Tensor      # (B, Nctx, dim)
    text_kv: torch.Tensor         # (B, text_len, dim)
    pooled_text: torch.Tensor     # (B, dim)
    batch: int
    # optional backbone-specific bags (Wan text embeds, etc.)
    extras: Optional[dict] = None


@runtime_checkable
class VideoBackbone(Protocol):
    cfg: ModelConfig
    memory: MemoryBank

    def frame_tokens(
        self, latents: torch.Tensor, role: int, frame_offset: int = 0
    ) -> torch.Tensor: ...

    def encode_context(
        self,
        text_ids: torch.Tensor,
        history: Optional[torch.Tensor],
        reference: Optional[torch.Tensor],
        memory_state: Optional[torch.Tensor],
    ) -> ModelContext: ...

    def forward(
        self,
        z_noised: torch.Tensor,
        sigma: torch.Tensor,
        context: ModelContext,
    ) -> torch.Tensor: ...


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        if rank > 0:
            self.lora_a = nn.Linear(base.in_features, rank, bias=False)
            self.lora_b = nn.Linear(rank, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.lora_a is not None:
            out = out + self.lora_b(self.lora_a(x)) * self.scaling
        return out

    def mergeable_state_dict(self) -> dict:
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith("lora_")
        }
