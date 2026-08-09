"""Container tying the trained components together, with checkpoint IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from .adapters.factory import build_backbone
from .config import OrbisConfig
from .device import get_device
from .superres import SuperResolution
from .vae import ConvVAE


def _build_vae(cfg: OrbisConfig) -> nn.Module:
    """ConvVAE (default, CPU-testable) or the real frozen Wan VAE (opt-in).

    Lazily imports ``orbis.adapters.wan21_vae`` only when
    ``backbone.real_vae`` is set, so the ``diffusers``-free toy/stub paths
    are unaffected -- mirrors how ``adapters/factory.py`` lazily imports
    ``wan21_real`` only when ``backbone.real_weights`` is set.
    """
    if cfg.backbone.type == "wan" and getattr(cfg.backbone, "real_vae", False):
        from .adapters.wan21_vae import RealWanVAE
        return RealWanVAE.from_pretrained(
            cfg.backbone.checkpoint_path, latent_channels=cfg.vae.latent_channels)
    return ConvVAE(cfg.vae, cfg.world)


@dataclass
class OrbisSystem:
    cfg: OrbisConfig
    vae: nn.Module
    generator: nn.Module
    sr: SuperResolution
    distilled: bool = False

    @staticmethod
    def build(cfg: Optional[OrbisConfig] = None) -> "OrbisSystem":
        cfg = cfg or OrbisConfig()
        torch.manual_seed(cfg.seed)
        vae = _build_vae(cfg)
        gen = build_backbone(cfg)
        sr = SuperResolution(cfg.sr, cfg.world)
        return OrbisSystem(cfg=cfg, vae=vae, generator=gen, sr=sr)

    def to(self, device: Union[str, torch.device]) -> "OrbisSystem":
        device = torch.device(device)
        self.vae.to(device)
        self.generator.to(device)
        self.sr.to(device)
        return self

    def eval(self) -> "OrbisSystem":
        self.vae.eval(); self.generator.eval(); self.sr.eval()
        return self

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "config": self.cfg.to_dict(),
            "vae": self.vae.state_dict(),
            "sr": self.sr.state_dict(),
            "distilled": self.distilled,
            "backbone_type": self.cfg.backbone.type,
        }
        gen = self.generator
        # Always persist the full generator — Wan previously saved only LoRA /
        # memory bags and dropped text_embed, cond_mlp, DiT blocks, etc.
        payload["generator"] = gen.state_dict()
        if hasattr(gen, "lora_state_dict") and self.cfg.backbone.type == "wan":
            payload["lora"] = gen.lora_state_dict()
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location=None) -> "OrbisSystem":
        device = get_device() if map_location is None else torch.device(map_location)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = OrbisConfig.from_dict(ckpt["config"])
        sys = OrbisSystem.build(cfg)
        # RealWanVAE.load_state_dict is a no-op (frozen, nothing persisted;
        # OrbisSystem.build already reconstructed an identical instance via
        # from_pretrained); ConvVAE.load_state_dict restores trained weights
        # as before.
        sys.vae.load_state_dict(ckpt["vae"])
        # Full generator first (includes LoRA tensors when present).
        if "generator" in ckpt:
            sys.generator.load_state_dict(ckpt["generator"], strict=False)
        elif "lora" in ckpt and hasattr(sys.generator, "load_lora_state_dict"):
            # Legacy Wan checkpoints that only stored the LoRA bag.
            sys.generator.load_lora_state_dict(ckpt["lora"])
        sys.sr.load_state_dict(ckpt["sr"])
        sys.distilled = ckpt.get("distilled", False)
        return sys.to(device).eval()
