"""The live inference engine -- the delivered-video runtime.

Drives a :class:`LiveSession` chunk by chunk: admit any pending prompt at the
boundary, assemble clean context (memory + reference + history), sample the chunk
with the few-step rectified-flow student, decode it, and consolidate state.
Chunks are delivered progressively (yielded as soon as they are ready) rather
than after the whole video completes.  T2V / I2V / V2V differ only in how the
first chunk's context is seeded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

from .device import get_device
from .flow import RectifiedFlow
from .serve.drift import DriftState, stabilize_chunk
from .serve.fifo import BoundedFIFO
from .session import LiveSession
from .system import OrbisSystem
from .vae import frames_to_tensor, tensor_to_frames


@dataclass
class Chunk:
    index: int
    version: int
    prompt: str
    frames: np.ndarray                 # (T, H, W, 3)
    latents: torch.Tensor              # (1, T, lc, lh, lw)
    hr_frames: Optional[np.ndarray] = None
    gen_seconds: float = 0.0
    admitted_version: Optional[int] = None


class LiveEngine:
    def __init__(self, system: OrbisSystem, steps: Optional[int] = None,
                 seed: int = 0, device=None):
        self.device = get_device(device)
        self.system = system.to(self.device).eval()
        self.cfg = system.cfg
        self.flow = RectifiedFlow(self.cfg.flow.train_sigma_eps)
        # real-time path uses the few-step student count
        self.steps = steps or self.cfg.flow.student_steps
        self.gen = system.generator
        self.vae = system.vae
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)
        self._drift = DriftState()
        self._fifo: BoundedFIFO = BoundedFIFO(
            capacity=getattr(self.cfg.serve, "fifo_capacity", 4))


    # -- latent <-> pixels ----------------------------------------------------
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        b, t = latents.shape[:2]
        frames = self.vae.decode(latents.reshape(b * t, *latents.shape[2:]))
        frames = frames.reshape(b, t, *frames.shape[1:])
        return tensor_to_frames(frames[0])          # (T, H, W, 3)

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        t = frames_to_tensor(frames).to(self.device)  # (T,3,H,W)
        z = self.vae.encode(t)
        return z.unsqueeze(0)                        # (1,T,lc,lh,lw)

    # -- session lifecycle ----------------------------------------------------
    def start(self, prompt: str, mode: str = "t2v",
              image: Optional[np.ndarray] = None,
              video: Optional[np.ndarray] = None) -> LiveSession:
        s = LiveSession(self.cfg, mode=mode)
        s.set_initial_prompt(prompt)
        s.memory_state = self.gen.memory.init(1, self.device)
        if mode == "i2v":
            if image is None:
                raise ValueError("i2v requires an image (H,W,3)")
            s.reference = self.encode(image[None])   # (1,1,...)
        elif mode == "v2v":
            if video is None:
                raise ValueError("v2v requires a video (T,H,W,3)")
            lat = self.encode(video)                 # (1,T,...)
            self._seed_history(s, lat)
        return s

    def _seed_history(self, s: LiveSession, latents: torch.Tensor) -> None:
        hf = self.cfg.model.history_frames
        n = latents.shape[1]
        if n > hf:
            self._write_memory(s, latents[:, :n - hf])
            s.history = latents[:, n - hf:]
        else:
            s.history = latents
        s.committed_frames += n

    # -- memory / history -----------------------------------------------------
    def _write_memory(self, s: LiveSession, evicted: torch.Tensor) -> None:
        cf = self.cfg.model.chunk_frames
        for i in range(0, evicted.shape[1], cf):
            group = evicted[:, i:i + cf]
            tokens = self.gen.frame_tokens(group, role=2)
            s.memory_state = self.gen.memory.write(s.memory_state, tokens)

    def _commit(self, s: LiveSession, latents: torch.Tensor) -> None:
        hf = self.cfg.model.history_frames
        combined = latents if s.history is None else \
            torch.cat([s.history, latents], dim=1)
        n = combined.shape[1]
        if n > hf:
            self._write_memory(s, combined[:, :n - hf])
            s.history = combined[:, n - hf:]
        else:
            s.history = combined
        s.chunk_index += 1
        s.committed_frames += latents.shape[1]

    # -- one chunk ------------------------------------------------------------
    @torch.no_grad()
    def generate_chunk(self, s: LiveSession) -> Chunk:
        admitted = s.admit_pending()
        cfg = self.cfg
        cf = cfg.model.chunk_frames
        lc = cfg.vae.latent_channels
        lh, lw = cfg.latent_hw
        # Referential integrity: Wan / anchor_reference keeps reference on every
        # chunk; toy default still seeds only the first chunk (empty history).
        anchor = getattr(self.cfg.backbone, "anchor_reference", False)
        if anchor:
            reference = s.reference
        else:
            reference = s.reference if s.history is None else None
        # Prefer async-encoded prompt ids when version matches (serve path)
        text_ids = s.resolve_text_ids().to(self.device)
        ctx = self.gen.encode_context(text_ids, s.history, reference,
                                      s.memory_state)

        def velocity_fn(z, sigma):
            return self.gen.forward(z, sigma, ctx)

        t0 = time.time()
        noise = torch.randn(1, cf, lc, lh, lw, device=self.device,
                            generator=self._rng)
        z = self.flow.sample(velocity_fn, (1, cf, lc, lh, lw), self.device,
                             steps=self.steps, noise=noise)
        frames = self.decode(z)
        serve = getattr(self.cfg, "serve", None)
        if serve is not None:
            frames = stabilize_chunk(
                frames, self._drift,
                threshold=serve.drift_threshold,
                blend=serve.drift_blend,
                enabled=serve.drift_enabled,
            )
        dt = time.time() - t0

        chunk = Chunk(index=s.chunk_index, version=s.active_version,
                      prompt=s.active_prompt, frames=frames, latents=z,
                      gen_seconds=dt, admitted_version=admitted)
        s.log.append({"chunk": s.chunk_index, "prompt": s.active_prompt,
                      "version": s.active_version, "admitted": admitted})
        self._commit(s, z)
        return chunk

    # -- streaming ------------------------------------------------------------
    def stream(self, s: LiveSession, n_chunks: int,
               superres: bool = False) -> Iterator[Chunk]:
        """Generate chunks into a bounded FIFO and yield in delivery order.

        Never drops frames: when the FIFO is full, the oldest chunk is delivered
        before enqueueing the next (progressive decode backpressure).
        """
        for _ in range(n_chunks):
            chunk = self.generate_chunk(s)
            if superres:
                chunk.hr_frames = self.upscale(chunk.frames)
            while self._fifo.full:
                oldest = self._fifo.pop()
                if oldest is not None:
                    yield oldest
            self._fifo.push(chunk)
        yield from self._fifo.drain()

    @torch.no_grad()
    def upscale(self, frames: np.ndarray) -> np.ndarray:
        lr = frames_to_tensor(frames).unsqueeze(0).to(self.device)  # (1,T,3,H,W)
        hr = self.system.sr(lr)
        return tensor_to_frames(hr[0])

    # -- scripted convenience (for demos / eval) ------------------------------
    def generate_video(self, prompt: str, n_chunks: int, mode: str = "t2v",
                       image=None, video=None,
                       schedule: Optional[Dict[int, str]] = None,
                       superres: bool = False, seed: Optional[int] = None):
        """Generate a full rollout, applying scheduled prompt switches.

        ``schedule`` maps a chunk index -> new prompt text; the switch is queued
        just before that chunk so it is admitted at that boundary (earlier,
        already-delivered chunks are unaffected)."""
        if seed is not None:
            self._rng.manual_seed(seed)
        schedule = schedule or {}
        s = self.start(prompt, mode=mode, image=image, video=video)
        all_frames: List[np.ndarray] = []
        hr_frames: List[np.ndarray] = []
        events = []
        for k in range(n_chunks):
            if k in schedule:
                upd = s.set_prompt(schedule[k])
                events.append({"chunk": k, "type": "prompt_switch",
                               "text": schedule[k], "version": upd.version})
            chunk = self.generate_chunk(s)
            if superres:
                chunk.hr_frames = self.upscale(chunk.frames)
                hr_frames.append(chunk.hr_frames)
            all_frames.append(chunk.frames)
        frames = np.concatenate(all_frames, axis=0)
        result = {"frames": frames, "events": events, "session": s,
                  "log": s.log}
        if superres:
            result["hr_frames"] = np.concatenate(hr_frames, axis=0)
        return result
