"""Global configuration for the Orbis reference implementation.

The numbers here are deliberately small so that the whole system -- latent
autoencoder, chunk-wise rectified-flow generator, bounded memory, few-step
distillation and streaming super-resolution -- trains and streams on a CPU in
minutes.  They are the toy-scale analogues of the production settings described
in the paper (native 832x480 generation, 4K delivery, 24 FPS).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class WorldConfig:
    """Toy video world -- the ground-truth data-generating process."""

    height: int = 32
    width: int = 32
    channels: int = 3
    fps: int = 8               # native frame rate of the toy world
    hr_scale: int = 4          # high-resolution factor used to supervise SR


@dataclass
class VAEConfig:
    """Small convolutional latent autoencoder (frames <-> latent)."""

    latent_channels: int = 8
    downsample: int = 4        # spatial downsample factor (32 -> 8)
    base_channels: int = 48


@dataclass
class ModelConfig:
    """Chunk-wise streaming rectified-flow generator (a small DiT)."""

    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 2.0
    patch_size: int = 2            # patchify the latent grid (8x8 -> 4x4 tokens)
    chunk_frames: int = 4          # latent frames generated per chunk
    history_frames: int = 4        # native committed latent frames kept as context
    memory_tokens: int = 16        # fixed-capacity persistent memory budget
    text_len: int = 8              # max prompt tokens


@dataclass
class FlowConfig:
    """Rectified-flow sampling schedule."""

    train_sigma_eps: float = 1e-4
    teacher_steps: int = 16        # multi-step ODE integration (pretrain/teacher)
    student_steps: int = 4         # few-step distilled student (real-time path)


@dataclass
class SRConfig:
    """Streaming super-resolution refinement network."""

    scale: int = 4                 # 32 -> 128 (toy analogue of 480 -> 4K)
    base_channels: int = 24


@dataclass
class OrbisConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    sr: SRConfig = field(default_factory=SRConfig)
    seed: int = 0

    @property
    def latent_hw(self) -> Tuple[int, int]:
        return (self.world.height // self.vae.downsample,
                self.world.width // self.vae.downsample)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "OrbisConfig":
        return OrbisConfig(
            world=WorldConfig(**d.get("world", {})),
            vae=VAEConfig(**d.get("vae", {})),
            model=ModelConfig(**d.get("model", {})),
            flow=FlowConfig(**d.get("flow", {})),
            sr=SRConfig(**d.get("sr", {})),
            seed=d.get("seed", 0),
        )


DEFAULT_CONFIG = OrbisConfig()
