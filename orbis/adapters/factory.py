"""Factory: build toy Generator or WanAdapter from OrbisConfig."""

from __future__ import annotations

import torch.nn as nn

from orbis.config import OrbisConfig
from orbis.model import Generator


def build_backbone(cfg: OrbisConfig) -> nn.Module:
    btype = (cfg.backbone.type or "toy").lower()
    if btype == "toy":
        return Generator(cfg.model, cfg.vae).build(cfg.latent_hw)
    if btype == "wan":
        if cfg.backbone.real_weights:
            # Real, frozen WanTransformer3DModel + LoRA + real UMT5 text
            # encoder -- see orbis/adapters/wan21_real.py for the full
            # integration design and its explicitly scoped fidelity gap
            # (orbis's own VAE, not Wan's native AutoencoderKLWan, this
            # phase). Requires the `wan` extra and, beyond CPU shape tests,
            # a real GPU (see deploy/README.md).
            from orbis.adapters.wan21_real import RealWanBackbone
            return RealWanBackbone.from_pretrained(
                cfg.model, cfg.vae, cfg.backbone, cfg.latent_hw)
        from orbis.adapters.wan_adapter import WanAdapter
        adapter = WanAdapter(
            cfg.model, cfg.vae, cfg.backbone, cfg.latent_hw)
        if not cfg.backbone.wan_stub:
            ok = adapter.try_load_hf(cfg.backbone.checkpoint_path)
            if ok:
                adapter.freeze_transformer_blocks()
        return adapter
    raise ValueError(f"Unknown backbone.type={cfg.backbone.type!r}")
