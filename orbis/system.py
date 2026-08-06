"""Container tying the trained components together, with checkpoint IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .config import OrbisConfig
from .device import get_device
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
        torch.save({
            "config": self.cfg.to_dict(),
            "vae": self.vae.state_dict(),
            "generator": self.generator.state_dict(),
            "sr": self.sr.state_dict(),
            "distilled": self.distilled,
        }, path)

    @staticmethod
    def load(path: str, map_location=None) -> "OrbisSystem":
        device = get_device() if map_location is None else torch.device(map_location)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = OrbisConfig.from_dict(ckpt["config"])
        sys = OrbisSystem.build(cfg)
        sys.vae.load_state_dict(ckpt["vae"])
        sys.generator.load_state_dict(ckpt["generator"])
        sys.sr.load_state_dict(ckpt["sr"])
        sys.distilled = ckpt.get("distilled", False)
        return sys.to(device).eval()
