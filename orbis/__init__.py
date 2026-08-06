"""Visko Orbis -- reference implementation of a Live Model for real-time,
interactive long-video generation.

A CPU-runnable, toy-scale but faithful realization of the paper's core
mechanisms: chunk-wise latent rectified-flow generation, bounded multi-scale
memory, progressive streaming training with few-step distillation, live
versioned prompt switching, and streaming super-resolution.
"""

from .config import OrbisConfig
from .system import OrbisSystem
from .engine import LiveEngine, Chunk
from .session import LiveSession

__all__ = ["OrbisConfig", "OrbisSystem", "LiveEngine", "Chunk", "LiveSession"]
__version__ = "1.0.0"
