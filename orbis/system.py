"""Container tying the trained components together, with checkpoint IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import OrbisConfig
from .model import Generator
from .superres import SuperResolution
from .vae import ConvVAE


@dataclass
class OrbisSystem:
    cfg: OrbisConfig
    vae: ConvVAE
    generator: Generator
    sr: SuperResolution
    distilled: bool = False

    @staticmethod
    def build(cfg: Optional[OrbisConfig] = None) -> "OrbisSystem":
        cfg = cfg or OrbisConfig()
        torch.manual_seed(cfg.seed)
        vae = ConvVAE(cfg.vae, cfg.world)
        gen = Generator(cfg.model, cfg.vae).build(cfg.latent_hw)
        sr = SuperResolution(cfg.sr, cfg.world)
        return OrbisSystem(cfg=cfg, vae=vae, generator=gen, sr=sr)

    def eval(self) -> "OrbisSystem":
        self.vae.eval(); self.generator.eval(); self.sr.eval()
        return self

    def save(self, path: str) -> None:
        torch.save({
            "config": self.cfg.to_dict(),
            "vae": self.vae.state_dict(),
            "generator": self.generator.state_dict(),
            "sr": self.sr.state_dict(),
            "distilled": self.distilled,
        }, path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "OrbisSystem":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = OrbisConfig.from_dict(ckpt["config"])
        sys = OrbisSystem.build(cfg)
        sys.vae.load_state_dict(ckpt["vae"])
        sys.generator.load_state_dict(ckpt["generator"])
        sys.sr.load_state_dict(ckpt["sr"])
        sys.distilled = ckpt.get("distilled", False)
        return sys.eval()
