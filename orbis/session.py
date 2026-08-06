"""Live session state and the prompt-versioning contract.

A session is a *persistent process*: it carries visual state (history window +
persistent memory) across chunks and accepts instruction updates against the
output clock.  The central rule from the paper is realised here exactly:

    a prompt update is admitted at the next uncommitted chunk boundary and
    affects only uncommitted chunks; delivered frames are never revised.

A *rolling prompt summary* (``rolling_summary``) carries established entities so
a partial update ("moving up") keeps the rest of the scene ("the red triangle").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from .config import OrbisConfig
from .text import PromptTokenizer, parse_controls, controls_to_prompt


@dataclass
class PromptUpdate:
    version: int
    controls: Dict[str, str]
    text: str
    submit_chunk: int          # chunk index that was in flight when submitted


@dataclass
class LiveSession:
    cfg: OrbisConfig
    mode: str = "t2v"
    # active (committed-at-boundary) condition
    active_controls: Dict[str, str] = field(default_factory=dict)
    active_ids: Optional[torch.Tensor] = None
    active_version: int = 0
    rolling_summary: Dict[str, str] = field(default_factory=dict)
    pending: Optional[PromptUpdate] = None
    _version_counter: int = 0
    # persistent visual state
    chunk_index: int = 0
    committed_frames: int = 0
    history: Optional[torch.Tensor] = None      # (1, n<=hf, lc, lh, lw)
    memory_state: Optional[torch.Tensor] = None  # (1, n_mem, dim)
    reference: Optional[torch.Tensor] = None     # (1, r, lc, lh, lw) or None
    # bookkeeping for the UI / eval
    log: list = field(default_factory=list)

    def __post_init__(self):
        self._tok = PromptTokenizer(self.cfg.model.text_len)

    # -- prompt control -------------------------------------------------------
    def set_initial_prompt(self, text: str) -> None:
        self.rolling_summary = parse_controls(text)
        self.active_controls = dict(self.rolling_summary)
        self.active_ids = self._encode(self.active_controls)
        self.active_version = 0
        self.pending = None

    def set_prompt(self, text: str) -> PromptUpdate:
        """Queue a prompt update; it will be admitted at the next chunk boundary.

        The new controls merge onto the rolling summary, so unspecified
        attributes persist.  Returns the update (with its version and the chunk
        it will first affect)."""
        self.rolling_summary = parse_controls(text, defaults=self.rolling_summary)
        self._version_counter += 1
        upd = PromptUpdate(version=self._version_counter,
                           controls=dict(self.rolling_summary),
                           text=text, submit_chunk=self.chunk_index)
        self.pending = upd            # latest pending overwrites stale pending
        return upd

    def admit_pending(self) -> Optional[int]:
        """Admit the latest pending prompt at a chunk boundary.  Returns the
        admitted version, or ``None`` if nothing was pending."""
        if self.pending is None:
            return None
        self.active_controls = dict(self.pending.controls)
        self.active_ids = self._encode(self.active_controls)
        self.active_version = self.pending.version
        v = self.pending.version
        self.pending = None
        return v

    @property
    def active_prompt(self) -> str:
        return controls_to_prompt(self.active_controls)

    def _encode(self, controls: Dict[str, str]) -> torch.Tensor:
        return torch.as_tensor(self._tok.encode_controls(controls)).unsqueeze(0)
