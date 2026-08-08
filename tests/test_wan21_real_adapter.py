"""CPU tests for RealWanBackbone against a mock Wan-shaped transformer.

``diffusers`` is not installed in CI/dev sandboxes and there is no GPU here,
so these tests build a small mock transformer that mirrors the exact module
layout of ``diffusers.models.transformers.transformer_wan.WanTransformerBlock``
(``attn1``/``attn2`` each with ``to_q``/``to_k``/``to_v``/``to_out``, an
``ffn.net`` ModuleList with a GEGLU-style ``net[0].proj`` and plain
``net[2]`` linear, all held in an ``nn.ModuleList`` of blocks) -- so the same
``_find_transformer_blocks``/``apply_lora_to_wan_blocks`` traversal exercised
here is the exact code path that runs against the real
``WanTransformer3DModel`` on a RunPod/Vast.ai GPU box (see
``deploy/README.md``); nothing in ``orbis/adapters/wan21_real.py`` is
special-cased for the mock.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from orbis.adapters.wan21_real import (RealWanBackbone,
                                        _find_transformer_blocks,
                                        apply_lora_to_wan_blocks)
from orbis.backbone import LoRALinear
from orbis.config import BackboneConfig, ModelConfig, VAEConfig

WAN_TEXT_DIM = 20


class _MockGEGLU(nn.Module):
    """Mirrors diffusers' GEGLU activation: a single Linear under `.proj`."""

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class _MockFFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.ModuleList(
            [_MockGEGLU(dim, hidden), nn.Dropout(0.0), nn.Linear(hidden, dim)])

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class _MockAttention(nn.Module):
    """Mirrors diffusers.models.attention_processor.Attention's public shape:
    to_q/to_k/to_v Linear + to_out ModuleList([Linear, Dropout])."""

    def __init__(self, dim: int, kv_dim: int | None = None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(kv_dim, dim)
        self.to_v = nn.Linear(kv_dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])

    def forward(self, x, context=None):
        kv = context if context is not None else x
        q, k, v = self.to_q(x), self.to_k(kv), self.to_v(kv)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
        out = attn @ v
        out = self.to_out[0](out)
        return self.to_out[1](out)


class _MockWanBlock(nn.Module):
    """Mirrors WanTransformerBlock's public attn1/attn2/ffn layout."""

    def __init__(self, dim: int, text_dim: int, ffn_hidden: int):
        super().__init__()
        self.attn1 = _MockAttention(dim)              # self-attention
        self.attn2 = _MockAttention(dim, kv_dim=text_dim)  # cross-attention
        self.ffn = _MockFFN(dim, ffn_hidden)

    def forward(self, x, encoder_hidden_states):
        x = x + self.attn1(x)
        x = x + self.attn2(x, context=encoder_hidden_states)
        x = x + self.ffn(x)
        return x


class MockWanTransformer(nn.Module):
    """Mirrors WanTransformer3DModel's public call signature and an
    `nn.ModuleList` of blocks, with small trainable layers so gradients
    flow end to end."""

    def __init__(self, in_channels: int = 4, dim: int = 8, num_layers: int = 2,
                 ffn_hidden: int = 12, text_dim: int = WAN_TEXT_DIM):
        super().__init__()
        self.in_channels = in_channels
        self.patch_in = nn.Linear(in_channels, dim)
        self.blocks = nn.ModuleList(
            [_MockWanBlock(dim, text_dim, ffn_hidden) for _ in range(num_layers)])
        self.patch_out = nn.Linear(dim, in_channels)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                return_dict=False):
        b, c, f, h, w = hidden_states.shape
        x = hidden_states.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)
        # Cheap timestep conditioning so `timestep` is actually consumed.
        x = self.patch_in(x) + timestep.view(b, 1, 1).to(x.dtype) * 0.0
        for block in self.blocks:
            x = block(x, encoder_hidden_states)
        x = self.patch_out(x)
        x = x.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)
        return (x,) if not return_dict else {"sample": x}


def _small_cfgs(real_weights: bool = True):
    model_cfg = ModelConfig(dim=16, depth=1, heads=2, mlp_ratio=2.0,
                             patch_size=1, chunk_frames=2, history_frames=2,
                             memory_tokens=4, text_len=6)
    vae_cfg = VAEConfig(latent_channels=4, downsample=4, base_channels=8)
    backbone_cfg = BackboneConfig(
        type="wan", lora_rank=4, lora_alpha=8.0, real_weights=real_weights,
        wan_num_train_timesteps=1000)
    latent_hw = (4, 4)
    return model_cfg, vae_cfg, backbone_cfg, latent_hw


def _mock_text_encoder_fn(prompts):
    b = len(prompts)
    return torch.randn(b, 5, WAN_TEXT_DIM)


def _build_backbone(**kwargs):
    model_cfg, vae_cfg, backbone_cfg, latent_hw = _small_cfgs()
    transformer = MockWanTransformer(in_channels=vae_cfg.latent_channels,
                                      **kwargs)
    return RealWanBackbone(
        model_cfg=model_cfg, vae_cfg=vae_cfg, backbone_cfg=backbone_cfg,
        latent_hw=latent_hw, transformer=transformer,
        text_encoder_fn=_mock_text_encoder_fn, wan_text_dim=WAN_TEXT_DIM,
    ), model_cfg, vae_cfg, backbone_cfg, latent_hw


def test_find_transformer_blocks_and_lora_targets():
    transformer = MockWanTransformer()
    blocks = _find_transformer_blocks(transformer)
    assert blocks is transformer.blocks
    wrapped = apply_lora_to_wan_blocks(blocks, rank=4, alpha=8.0)
    # 3 attn1 projs + 1 attn1 out + 3 attn2 projs + 1 attn2 out + 2 ffn
    # linears per block, over 2 blocks.
    assert len(wrapped) == 2 * (3 + 1 + 3 + 1 + 2)
    assert isinstance(blocks[0].attn1.to_q, LoRALinear)
    assert isinstance(blocks[0].ffn.net[2], LoRALinear)


def test_lora_injection_and_trainable_params():
    gen, *_ = _build_backbone()
    trainable_names = {n for n, p in gen.named_parameters() if p.requires_grad}
    frozen_names = {n for n, p in gen.named_parameters() if not p.requires_grad}

    assert any("lora_a" in n or "lora_b" in n for n in trainable_names)
    assert any("memory" in n for n in trainable_names)
    assert any("mem_to_wan" in n for n in trainable_names)
    # Frozen base transformer weights (wrapped LoRALinear.base.*) must never
    # be trainable, and un-wrapped transformer params (patch_in/patch_out)
    # must also stay frozen.
    assert any("transformer.patch_in" in n for n in frozen_names)
    assert not any(".base." in n for n in trainable_names)
    assert list(gen.trainable_parameters())
    assert all(p.requires_grad for p in gen.trainable_parameters())


def test_forward_shape_matches_z_noised_with_history_and_reference():
    gen, model_cfg, vae_cfg, backbone_cfg, latent_hw = _build_backbone()
    b, cf = 2, model_cfg.chunk_frames
    lc = vae_cfg.latent_channels
    lh, lw = latent_hw
    z = torch.randn(b, cf, lc, lh, lw)
    history = torch.randn(b, model_cfg.history_frames, lc, lh, lw)
    reference = torch.randn(b, 1, lc, lh, lw)
    text_ids = torch.zeros(b, model_cfg.text_len, dtype=torch.long)
    mem_state = gen.memory.init(b, z.device)

    ctx = gen.encode_context(text_ids, history, reference, mem_state)
    out = gen.forward(z, torch.full((b,), 0.5), ctx)
    assert out.shape == z.shape

    out2 = gen.velocity(z, torch.full((b,), 0.5), text_ids, history, reference,
                         mem_state)
    assert out2.shape == z.shape


def test_forward_without_history_or_reference():
    gen, model_cfg, vae_cfg, backbone_cfg, latent_hw = _build_backbone()
    b, cf = 1, model_cfg.chunk_frames
    lc = vae_cfg.latent_channels
    lh, lw = latent_hw
    z = torch.randn(b, cf, lc, lh, lw)
    text_ids = torch.zeros(b, model_cfg.text_len, dtype=torch.long)
    empty_hist = torch.zeros(b, 0, lc, lh, lw)
    empty_ref = torch.zeros(b, 0, lc, lh, lw)
    ctx = gen.encode_context(text_ids, empty_hist, empty_ref, None)
    out = gen.forward(z, torch.full((b,), 0.2), ctx)
    assert out.shape == z.shape


def test_gradients_flow_to_lora_and_memory_not_base():
    gen, model_cfg, vae_cfg, backbone_cfg, latent_hw = _build_backbone()
    b, cf = 1, model_cfg.chunk_frames
    lc = vae_cfg.latent_channels
    lh, lw = latent_hw
    z = torch.randn(b, cf, lc, lh, lw)
    text_ids = torch.zeros(b, model_cfg.text_len, dtype=torch.long)
    mem_state = gen.memory.init(b, z.device)
    out = gen.velocity(z, torch.full((b,), 0.5), text_ids, None, None, mem_state)
    out.sum().backward()

    lora_grad_found = False
    for n, p in gen.named_parameters():
        if ".base." in n:
            assert p.grad is None, f"frozen base param {n} unexpectedly got a grad"
        elif p.requires_grad and p.grad is not None:
            lora_grad_found = True
    assert lora_grad_found


def test_frame_tokens_shape_for_memory_bank():
    gen, model_cfg, vae_cfg, backbone_cfg, latent_hw = _build_backbone()
    b, f = 2, model_cfg.chunk_frames
    lc = vae_cfg.latent_channels
    lh, lw = latent_hw
    latents = torch.randn(b, f, lc, lh, lw)
    tok = gen.frame_tokens(latents, role=2)
    p = model_cfg.patch_size
    gh, gw = lh // p, lw // p
    assert tok.shape == (b, f * gh * gw, model_cfg.dim)

    # Also confirm the memory bank can actually consume these tokens
    # end-to-end, exactly as LiveEngine._write_memory does.
    mem_state = gen.memory.init(b, latents.device)
    mem_state = gen.memory.write(mem_state, tok)
    assert mem_state.shape == (b, model_cfg.memory_tokens, model_cfg.dim)


def test_checkpoint_roundtrip_is_lean_and_restores_lora():
    gen, model_cfg, vae_cfg, backbone_cfg, latent_hw = _build_backbone()
    full_param_count = sum(1 for _ in gen.named_parameters())
    sd = gen.state_dict()
    # Only trainable (LoRA + memory/projection) params are kept, not the
    # frozen multi-param mock base transformer.
    assert len(sd) < full_param_count
    assert all(".base." not in k for k in sd.keys())
    assert any("lora_a" in k or "lora_b" in k for k in sd.keys())

    # Mutate a LoRA weight, then confirm a fresh instance with identical
    # (untrained) init differs, and correctly matches after loading `sd`.
    with torch.no_grad():
        for n, p in gen.named_parameters():
            if "lora_b" in n:
                p.add_(1.0)
                break
    mutated_sd = gen.state_dict()

    gen2, *_ = _build_backbone()
    before = {n: p.clone() for n, p in gen2.named_parameters() if "lora_b" in n}
    gen2.load_state_dict(mutated_sd, strict=False)
    after = {n: p for n, p in gen2.named_parameters() if "lora_b" in n}
    changed = any(not torch.allclose(before[n], after[n]) for n in before)
    assert changed

    # lora_state_dict/load_lora_state_dict back-compat aliases.
    assert gen.lora_state_dict().keys() == sd.keys()
    gen3, *_ = _build_backbone()
    gen3.load_lora_state_dict(mutated_sd)
