"""Reusable neural building blocks (timestep embedding, DiT block, MLPs)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(sigma: torch.Tensor, dim: int, max_period: float = 1000.0):
    """Sinusoidal embedding of the flow time ``sigma`` in ``[0, 1]``."""
    sigma = sigma.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=sigma.device) / half
    )
    args = sigma[:, None] * freqs[None] * max_period
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head attention supporting optional separate key/value source."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context=None, attn_mask=None):
        b, n, d = x.shape
        ctx = x if context is None else context
        m = ctx.shape[1]
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).view(b, m, 2, self.heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class DiTBlock(nn.Module):
    """A DiT block: AdaLN-Zero self-attention, text cross-attention, MLP.

    Global conditioning ``c`` (timestep + pooled text) drives AdaLN modulation;
    the token-level text embedding enters through cross-attention, matching the
    generator described in the paper (timestep projections + text cross-attn).
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        # 6 modulation params for self-attn + MLP, plus 1 gate for cross-attn.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 7 * dim))
        # Scratch training: open residual gates so attn/MLP receive gradients.
        # Classic AdaLN-Zero (gates=0) starves block weights of grad from step 0.
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        with torch.no_grad():
            bias = self.ada[-1].bias.view(7, dim)
            bias[2] = 1.0  # gate_msa
            bias[5] = 1.0  # gate_mlp
            bias[6] = 1.0  # gate_ca

    def forward(self, x, c, text_kv, self_mask=None):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         gate_ca) = self.ada(c).chunk(7, dim=-1)
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(h, attn_mask=self_mask)
        x = x + gate_ca.unsqueeze(1) * self.cross(self.norm_ca(x), context=text_kv)
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, out_dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        # Non-zero readout so velocity head is trainable from step 0.
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
