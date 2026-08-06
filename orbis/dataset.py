"""Synthetic data sampling for training.

Each example is a short rollout of the toy world encoded into latents.  We
produce three conditioning modes so a single set of weights supports every
inference path in the paper:

* ``text_only``  -> first-chunk T2V (generate a plausible scene from text);
* ``history``    -> streaming continuation (recent window + populated memory);
* ``reference``  -> I2V / V2V (a clean prefix as reference context).

For the ``history`` mode the rollout is long enough that frames are evicted from
the recency window into the persistent memory, so the memory write is trained.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .config import OrbisConfig
from .world import sample_scene, rollout
from .vae import ConvVAE, frames_to_tensor


class RolloutSampler:
    def __init__(self, cfg: OrbisConfig, vae: ConvVAE, seed: int = 0):
        self.cfg = cfg
        self.vae = vae
        self.rng = np.random.default_rng(seed)
        self.H = cfg.world.height
        self.W = cfg.world.width

    # -- raw frames -----------------------------------------------------------
    def frame_batch(self, batch: int, n_frames: int):
        """Return ``(frames[B,F,H,W,3], token_ids[B,L])`` for random scenes."""
        from .text import PromptTokenizer
        tok = PromptTokenizer(self.cfg.model.text_len)
        frames = np.empty((batch, n_frames, self.H, self.W, 3), dtype=np.float32)
        ids = np.empty((batch, self.cfg.model.text_len), dtype=np.int64)
        for b in range(batch):
            spec = sample_scene(self.rng)
            fr, _ = rollout(spec, n_frames, self.H, self.W)
            frames[b] = fr
            ids[b] = tok.encode(spec.control_prompt())
        return frames, ids

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """``(B,F,H,W,3)`` -> latents ``(B,F,lc,lh,lw)``."""
        b, f = frames.shape[:2]
        device = next(self.vae.parameters()).device
        t = frames_to_tensor(frames).reshape(b * f, 3, self.H, self.W).to(device)
        z = self.vae.encode(t)
        return z.reshape(b, f, *z.shape[1:])

    # -- training batches -----------------------------------------------------
    def training_batch(self, batch: int, mode: str = "history",
                       history_frames: Optional[int] = None,
                       memory_context_chunks: int = 2,
                       history_noise: float = 0.0):
        cfg = self.cfg
        cf = cfg.model.chunk_frames
        hf = history_frames if history_frames is not None else cfg.model.history_frames

        if mode == "text_only":
            frames, ids = self.frame_batch(batch, cf)
            z = self.encode(frames)
            return {"target": z, "text_ids": torch.as_tensor(ids),
                    "history": None, "reference": None,
                    "evicted": None, "mode": mode}

        if mode == "reference":
            frames, ids = self.frame_batch(batch, 1 + cf)
            z = self.encode(frames)
            return {"target": z[:, 1:], "text_ids": torch.as_tensor(ids),
                    "history": None, "reference": z[:, :1],
                    "evicted": None, "mode": mode}

        # history / streaming mode
        n_evict = memory_context_chunks * cf
        total = n_evict + hf + cf
        frames, ids = self.frame_batch(batch, total)
        z = self.encode(frames)
        evicted = z[:, :n_evict] if n_evict > 0 else None
        history = z[:, n_evict:n_evict + hf]
        target = z[:, n_evict + hf:]
        if history_noise > 0:
            history = history + history_noise * torch.randn_like(history)
        return {"target": target, "text_ids": torch.as_tensor(ids),
                "history": history, "reference": None,
                "evicted": evicted, "mode": mode}
