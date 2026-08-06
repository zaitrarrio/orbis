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


@dataclass
class OrbisSystem:
    cfg: OrbisConfig
    vae: ConvVAE
    generator: nn.Module
    sr: SuperResolution
    distilled: bool = False

    @staticmethod
    def build(cfg: Optional[OrbisConfig] = None) -> "OrbisSystem":
        cfg = cfg or OrbisConfig()
        torch.manual_seed(cfg.seed)
        vae = ConvVAE(cfg.vae, cfg.world)
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
        # Wan path: prefer LoRA/adapter bag; toy: full generator state
        gen = self.generator
        if hasattr(gen, "lora_state_dict") and self.cfg.backbone.type == "wan":
            payload["lora"] = gen.lora_state_dict()
            payload["generator"] = {
                k: v for k, v in gen.state_dict().items()
                if "lora_" in k or k.startswith("memory.")
                or k.startswith("ref_proj.") or k == "identity_token"
            }
        else:
            payload["generator"] = gen.state_dict()
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location=None) -> "OrbisSystem":
        device = get_device() if map_location is None else torch.device(map_location)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = OrbisConfig.from_dict(ckpt["config"])
        sys = OrbisSystem.build(cfg)
        sys.vae.load_state_dict(ckpt["vae"])
        if "lora" in ckpt and hasattr(sys.generator, "load_lora_state_dict"):
            sys.generator.load_lora_state_dict(ckpt["lora"])
        else:
            sys.generator.load_state_dict(ckpt["generator"], strict=False)
        sys.sr.load_state_dict(ckpt["sr"])
        sys.distilled = ckpt.get("distilled", False)
        return sys.to(device).eval()
