"""Public / local video clip loader for Wan-scale mid and post-training.

Expects a directory of clips (``.mp4`` / ``.webm`` / frame folders) with optional
JSONL captions::

    {"path": "clip_0001.mp4", "caption": "a dog runs across a field",
     "events": [{"t": 0.0, "text": "a dog stands"},
                {"t": 1.5, "text": "the dog runs to the right"}]}

When no real media is present, ``SyntheticVideoDataset`` falls back to the
toy world so CI and smoke runs stay self-contained.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from orbis.config import OrbisConfig
from orbis.text import PromptTokenizer
from orbis.world import rollout, sample_scene


@dataclass
class EventCaption:
    t: float
    text: str


@dataclass
class ClipRecord:
    path: str
    caption: str
    events: List[EventCaption] = field(default_factory=list)


def load_manifest(root: str | Path) -> List[ClipRecord]:
    root = Path(root)
    manifest = root / "manifest.jsonl"
    records: List[ClipRecord] = []
    if manifest.exists():
        with manifest.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                events = [
                    EventCaption(t=float(e["t"]), text=e["text"])
                    for e in obj.get("events", [])
                ]
                records.append(ClipRecord(
                    path=str(root / obj["path"]),
                    caption=obj.get("caption", ""),
                    events=events,
                ))
        return records
    # Discover loose video files
    for p in sorted(root.glob("**/*")):
        if p.suffix.lower() in {".mp4", ".webm", ".mov", ".avi"}:
            records.append(ClipRecord(path=str(p), caption=p.stem.replace("_", " ")))
    return records


def _load_frames_ffmpeg(path: str, max_frames: int, h: int, w: int) -> Optional[np.ndarray]:
    """Best-effort decode via imageio/ffmpeg; returns None if unavailable."""
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception:
        return None
    try:
        frames = []
        for i, frame in enumerate(iio.imiter(path)):
            if i >= max_frames:
                break
            arr = frame.astype(np.float32) / 255.0
            if arr.shape[0] != h or arr.shape[1] != w:
                # nearest resize without torchvision
                ys = (np.linspace(0, arr.shape[0] - 1, h)).astype(np.int64)
                xs = (np.linspace(0, arr.shape[1] - 1, w)).astype(np.int64)
                arr = arr[ys][:, xs]
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            frames.append(arr)
        if not frames:
            return None
        return np.stack(frames, axis=0)
    except Exception:
        return None


class VideoClipDataset:
    """Iterable of (frames, caption, events) for streaming / mid-training."""

    def __init__(self, root: str | Path, cfg: OrbisConfig, seed: int = 0):
        self.root = Path(root)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.records = load_manifest(self.root)
        self.tok = PromptTokenizer(cfg.model.text_len)

    def __len__(self) -> int:
        return len(self.records)

    def sample(self) -> Tuple[np.ndarray, str, List[EventCaption]]:
        if not self.records:
            return SyntheticVideoDataset(self.cfg, seed=int(self.rng.integers(1e9))).sample()
        rec = self.records[int(self.rng.integers(len(self.records)))]
        n = self.cfg.model.chunk_frames * 4
        frames = _load_frames_ffmpeg(
            rec.path, n, self.cfg.world.height, self.cfg.world.width)
        if frames is None:
            return SyntheticVideoDataset(self.cfg, seed=int(self.rng.integers(1e9))).sample()
        return frames, rec.caption, rec.events


class SyntheticVideoDataset:
    """Toy-world stand-in that emits event-aligned captions for mid-training."""

    def __init__(self, cfg: OrbisConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.tok = PromptTokenizer(cfg.model.text_len)

    def sample(self) -> Tuple[np.ndarray, str, List[EventCaption]]:
        H, W = self.cfg.world.height, self.cfg.world.width
        spec = sample_scene(self.rng)
        n = self.cfg.model.chunk_frames * 4
        # Mid-rollout attribute change at halfway (color and/or shape+direction)
        switch_t = n // 2
        patch = {"color": self.rng.choice(
            ["red", "blue", "green", "yellow", "cyan", "magenta"])}
        if self.rng.random() < 0.6:
            patch["shape"] = self.rng.choice(["circle", "square", "triangle"])
        if self.rng.random() < 0.5:
            patch["direction"] = self.rng.choice(
                ["left", "right", "up", "down"])
        schedule = {switch_t: patch}
        frames, specs = rollout(spec, n, H, W, control_schedule=schedule)
        cap0 = f"a {spec.color} {spec.shape} moving {spec.direction}"
        spec1 = specs[min(switch_t, len(specs) - 1)]
        cap1 = f"a {spec1.color} {spec1.shape} moving {spec1.direction}"
        events = [
            EventCaption(t=0.0, text=cap0),
            EventCaption(t=switch_t / max(self.cfg.world.fps, 1), text=cap1),
        ]
        return frames, cap0, events


def event_condition_schedule(
    events: Sequence[EventCaption], fps: int, chunk_frames: int
) -> Dict[int, str]:
    """Map event times to chunk-index prompt switches for mid-training."""
    schedule: Dict[int, str] = {}
    for ev in events:
        frame = int(ev.t * fps)
        chunk = frame // max(chunk_frames, 1)
        if chunk > 0:
            schedule[chunk] = ev.text
    return schedule


class MidTrainBatcher:
    """Builds training batches with mid-rollout condition changes."""

    def __init__(self, cfg: OrbisConfig, vae, dataset, seed: int = 0):
        self.cfg = cfg
        self.vae = vae
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.tok = PromptTokenizer(cfg.model.text_len)

    @torch.no_grad()
    def training_batch(self, batch: int, mode: str = "history") -> dict:
        from orbis.vae import frames_to_tensor

        cf = self.cfg.model.chunk_frames
        hf = self.cfg.model.history_frames
        device = next(self.vae.parameters()).device
        targets, texts, histories, refs, evicteds = [], [], [], [], []
        event_schedules = []

        for _ in range(batch):
            frames, caption, events = self.dataset.sample()
            # Encode full clip to latents
            x = frames_to_tensor(frames).to(device)       # (T,3,H,W)
            z = self.vae.encode(x).unsqueeze(0)          # (1,T,C,lh,lw)
            t = z.shape[1]
            # Pick a target chunk index past the first
            max_start = max(1, t - cf)
            start = int(self.rng.integers(0, max_start))
            # Align start to event boundary when available
            sched = event_condition_schedule(
                events, self.cfg.world.fps, cf)
            if sched and mode == "event":
                start = min(sched.keys()) * cf
                start = min(start, max_start)
            target = z[:, start:start + cf]
            if target.shape[1] < cf:
                pad = cf - target.shape[1]
                target = torch.cat(
                    [target, target[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)

            prefix = z[:, :start]
            hist = prefix[:, -hf:] if prefix.shape[1] > 0 else None
            evicted = prefix[:, :-hf] if prefix.shape[1] > hf else None
            ref = z[:, :1] if mode in ("reference", "event") else None

            # Active caption: event text at this chunk if any
            chunk_idx = start // cf
            text = caption
            for ev in events:
                if int(ev.t * self.cfg.world.fps) // cf <= chunk_idx:
                    text = ev.text

            targets.append(target.squeeze(0))
            texts.append(self.tok.encode(text))
            histories.append(hist.squeeze(0) if hist is not None else None)
            refs.append(ref.squeeze(0) if ref is not None else None)
            evicteds.append(evicted.squeeze(0) if evicted is not None else None)
            event_schedules.append(sched)

        def _stack_opt(items):
            if all(x is None for x in items):
                return None
            proto = next(x for x in items if x is not None)
            filled = [
                x if x is not None else torch.zeros_like(proto) for x in items]
            tlen = min(x.shape[0] for x in filled)
            filled = [x[:tlen] for x in filled]
            return torch.stack(filled, dim=0)

        return {
            "target": torch.stack(targets, dim=0),
            "text_ids": torch.stack(
                [torch.as_tensor(t, dtype=torch.long) for t in texts], dim=0),
            "history": _stack_opt(histories),
            "reference": _stack_opt(refs),
            "evicted": _stack_opt(evicteds),
            "event_schedules": event_schedules,
        }
