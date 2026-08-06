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
        from orbis.adapters.wan_adapter import WanAdapter
        adapter = WanAdapter(
            cfg.model, cfg.vae, cfg.backbone, cfg.latent_hw)
        if not cfg.backbone.wan_stub:
            adapter.try_load_hf(cfg.backbone.checkpoint_path)
        return adapter
    raise ValueError(f"Unknown backbone.type={cfg.backbone.type!r}")
