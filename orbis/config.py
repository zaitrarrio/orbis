"""Global configuration for the Orbis reference implementation.

Toy defaults stay small for fast contract tests.  Real-scale Wan settings target
native ~480x832 generation (paper analogue of 832x480), 24 FPS, and x4 SR.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass
class WorldConfig:
    """Video frame geometry (toy world or real-scale canvas)."""

    height: int = 32
    width: int = 32
    channels: int = 3
    fps: int = 8
    hr_scale: int = 4


@dataclass
class VAEConfig:
    """Latent autoencoder (frames <-> latent)."""

    latent_channels: int = 8
    downsample: int = 4
    base_channels: int = 48


@dataclass
class ModelConfig:
    """Chunk-wise streaming rectified-flow generator."""

    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 2.0
    patch_size: int = 2
    chunk_frames: int = 4
    history_frames: int = 4
    memory_tokens: int = 16
    text_len: int = 8


@dataclass
class FlowConfig:
    """Rectified-flow sampling schedule."""

    train_sigma_eps: float = 1e-4
    teacher_steps: int = 16
    student_steps: int = 4


@dataclass
class SRConfig:
    """Streaming super-resolution refinement network."""

    scale: int = 4
    base_channels: int = 24


@dataclass
class BackboneConfig:
    """Which generative backbone the live engine drives."""

    type: str = "toy"                 # "toy" | "wan"
    # Hugging Face / local path for Wan2.1 weights (optional; stub trains without)
    checkpoint_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    # Keep reference latents conditioned on every chunk (referential integrity)
    anchor_reference: bool = False
    # Use BF16 on CUDA when available
    use_bf16: bool = True
    # Structural Wan-scale DiT when official weights are not loaded
    wan_stub: bool = True


@dataclass
class ServeConfig:
    """Progressive delivery and drift stabilization."""

    fifo_capacity: int = 4
    drift_enabled: bool = False
    drift_threshold: float = 0.35
    drift_blend: float = 0.15


@dataclass
class OrbisConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    sr: SRConfig = field(default_factory=SRConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
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
            backbone=BackboneConfig(**d.get("backbone", {})),
            serve=ServeConfig(**d.get("serve", {})),
            seed=d.get("seed", 0),
        )


def wan_real_scale_config(
    checkpoint_path: Optional[str] = None,
    stub: bool = True,
) -> OrbisConfig:
    """OrbisConfig for Wan2.1-class live training/inference at ~480p."""
    # 480x832 native (height x width), x4 SR -> ~1920x3328 presentation
    world = WorldConfig(height=480, width=832, channels=3, fps=24, hr_scale=4)
    # Wan-VAE-like spatial compression (approx x8); we use x8 for real-scale
    vae = VAEConfig(latent_channels=16, downsample=8, base_channels=64)
    model = ModelConfig(
        dim=512, depth=12, heads=8, mlp_ratio=4.0, patch_size=2,
        chunk_frames=8, history_frames=8, memory_tokens=32, text_len=64,
    )
    flow = FlowConfig(teacher_steps=16, student_steps=4)
    sr = SRConfig(scale=4, base_channels=48)
    backbone = BackboneConfig(
        type="wan",
        checkpoint_path=checkpoint_path or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        lora_rank=8,
        lora_alpha=16.0,
        anchor_reference=True,
        use_bf16=True,
        wan_stub=stub,
    )
    serve = ServeConfig(fifo_capacity=4, drift_enabled=True)
    return OrbisConfig(
        world=world, vae=vae, model=model, flow=flow, sr=sr,
        backbone=backbone, serve=serve, seed=0,
    )


def wan_smoke_config(seed: int = 0) -> OrbisConfig:
    """Small Wan-path config for CI / smoke (still backbone.type=wan)."""
    world = WorldConfig(height=64, width=64, channels=3, fps=8, hr_scale=4)
    vae = VAEConfig(latent_channels=8, downsample=4, base_channels=32)
    model = ModelConfig(
        dim=64, depth=2, heads=4, mlp_ratio=2.0, patch_size=2,
        chunk_frames=2, history_frames=2, memory_tokens=8, text_len=8,
    )
    return OrbisConfig(
        world=world, vae=vae, model=model,
        flow=FlowConfig(teacher_steps=4, student_steps=2),
        sr=SRConfig(scale=2, base_channels=16),
        backbone=BackboneConfig(
            type="wan", wan_stub=True, anchor_reference=True, lora_rank=4),
        serve=ServeConfig(fifo_capacity=2, drift_enabled=True),
        seed=seed,
    )


DEFAULT_CONFIG = OrbisConfig()
