"""Bounded multi-scale memory.

Recent committed frames are retained at native granularity in a short recency
window; when frames leave that window they are *consolidated* into a
fixed-capacity persistent state ``M`` rather than being recompressed from the
full prefix.  The persistent state has a constant token budget, so read/write
cost and memory footprint are independent of rollout length -- the property the
paper relies on for hour-scale generation.

The learnable write is a gated cross-attention: the persistent slots query the
evicted tokens and integrate them, preventing detailed history from mutating the
slots destructively.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import Attention, MLP


class MemoryBank(nn.Module):
    def __init__(self, dim: int, n_mem: int, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.n_mem = n_mem
        self.init_state = nn.Parameter(torch.randn(n_mem, dim) * 0.02)
        self.slot_pos = nn.Parameter(torch.randn(n_mem, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.write_attn = Attention(dim, heads)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 2)
        # Learned gates start near zero so an untrained bank is a no-op.
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_mlp = nn.Parameter(torch.zeros(1))

    def init(self, batch: int, device) -> torch.Tensor:
        return self.init_state.unsqueeze(0).expand(batch, -1, -1).to(device)

    def write(self, state: torch.Tensor, evicted: torch.Tensor) -> torch.Tensor:
        """Consolidate ``evicted`` tokens ``(B, n, dim)`` into ``state``."""
        if evicted is None or evicted.shape[1] == 0:
            return state
        q = self.norm_q(state + self.slot_pos.unsqueeze(0))
        kv = self.norm_kv(evicted)
        state = state + torch.tanh(self.gate_attn) * self.write_attn(q, context=kv)
        state = state + torch.tanh(self.gate_mlp) * self.mlp(self.norm_mlp(state))
        return state

    def read(self, state: torch.Tensor) -> torch.Tensor:
        """Return the persistent tokens (with slot positions) for the generator."""
        return state + self.slot_pos.unsqueeze(0)
