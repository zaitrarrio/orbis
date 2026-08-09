"""Prompt parsing and tokenisation.

Two views of a prompt are produced, mirroring the paper's split between the
symbolic *control* carried by an instruction and the learned *embedding* the
generator conditions on:

* :func:`parse_controls` -> a canonical ``{shape,color,direction,speed,size}``
  dict (used for data generation, evaluation and the rolling prompt summary);
* :meth:`PromptTokenizer.encode` -> a padded array of canonical token ids the
  model embeds and cross-attends to.

Both are multilingual: synonyms in several languages collapse onto the same
canonical token before anything else happens.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import torch

from .vocab import (CANONICAL_TOKENS, COLORS, DIRECTIONS, SHAPES, SIZES,
                    SPEEDS, SYNONYMS, TOKEN_TO_ID)

ID_TO_TOKEN: Dict[int, str] = {i: t for i, t in enumerate(CANONICAL_TOKENS)}

_CATEGORIES = {
    "shape": set(SHAPES),
    "color": set(COLORS),
    "direction": set(DIRECTIONS),
    "speed": set(SPEEDS),
    "size": set(SIZES),
}

# Split on whitespace and common punctuation; keep CJK characters as their own
# tokens so "红色圆" tokenises even without spaces.
_TOKEN_RE = re.compile(r"[A-Za-z\-]+|[一-鿿]")


def _normalize_tokens(text: str) -> List[str]:
    raw = _TOKEN_RE.findall(text.lower())
    # Also try to catch multi-char CJK synonyms (e.g. "圆形") by scanning the
    # raw string for the longest synonym keys.
    canon: List[str] = []
    joined = text.lower()
    # greedy multi-char CJK synonym pass
    i = 0
    cjk = re.findall(r"[一-鿿]+", joined)
    for run in cjk:
        j = 0
        while j < len(run):
            matched = None
            for length in (3, 2, 1):
                sub = run[j:j + length]
                if sub in SYNONYMS:
                    matched = (sub, length)
                    break
            if matched:
                canon.append(SYNONYMS[matched[0]])
                j += matched[1]
            else:
                j += 1
    # latin / pinyin tokens
    for tok in raw:
        if tok in SYNONYMS:
            canon.append(SYNONYMS[tok])
    return canon


def parse_controls(text: str,
                   defaults: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Extract canonical controls from a free-form (multilingual) prompt.

    Categories not mentioned by the prompt inherit from ``defaults`` (the rolling
    prompt summary), which is how established entities persist across a prompt
    switch that only names *some* attributes.
    """
    controls: Dict[str, str] = dict(defaults or {})
    for canon in _normalize_tokens(text):
        for cat, members in _CATEGORIES.items():
            if canon in members:
                controls[cat] = canon
    return controls


def controls_to_prompt(controls: Dict[str, str]) -> str:
    shape = controls.get("shape", "circle")
    color = controls.get("color", "white")
    direction = controls.get("direction", "right")
    speed = controls.get("speed", "medium")
    size = controls.get("size", "medium")
    mods = []
    if size != "medium":
        mods.append(size)
    mods += [color, shape]
    s = "a " + " ".join(mods) + f" moving {direction}"
    if speed != "medium":
        s += f" {speed}"
    return s


class PromptTokenizer:
    """Maps prompts to fixed-length arrays of canonical token ids."""

    def __init__(self, max_len: int = 8):
        self.max_len = max_len

    def encode(self, text: str,
               defaults: Optional[Dict[str, str]] = None) -> np.ndarray:
        controls = parse_controls(text, defaults=defaults)
        # Deterministic slot order so the same meaning always tokenises the same
        # way regardless of the surface prompt.
        order = ["shape", "color", "direction", "speed", "size"]
        ids = [TOKEN_TO_ID[controls[c]] for c in order if c in controls]
        ids = ids[: self.max_len]
        ids += [0] * (self.max_len - len(ids))  # pad
        return np.array(ids, dtype=np.int64)

    def encode_controls(self, controls: Dict[str, str]) -> np.ndarray:
        order = ["shape", "color", "direction", "speed", "size"]
        ids = [TOKEN_TO_ID[controls[c]] for c in order if c in controls]
        ids = ids[: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return np.array(ids, dtype=np.int64)


def ids_to_controls(ids) -> Dict[str, str]:
    """Invert :meth:`PromptTokenizer.encode_controls` back into a controls dict.

    Classifies each non-pad id by which canonical category it belongs to
    (shape/color/direction/speed/size) — safe because ``CANONICAL_TOKENS`` is a
    disjoint union of those five closed vocabularies plus filler words.
    """
    controls: Dict[str, str] = {}
    for i in ids:
        i = int(i)
        if i <= 0 or i not in ID_TO_TOKEN:
            continue
        tok = ID_TO_TOKEN[i]
        for cat, members in _CATEGORIES.items():
            if tok in members and cat not in controls:
                controls[cat] = tok
    return controls


def ids_to_prompt_strings(text_ids) -> List[str]:
    """Bridge the toy symbolic vocab to a real natural-language text encoder.

    ``orbis``'s own vocab (see ``orbis/vocab.py``) is a small closed set of
    canonical control tokens (shape/color/direction/speed/size, e.g. ``red``,
    ``circle``, ``right``), not a natural-language tokenizer. Real Wan2.1 is
    conditioned by a UMT5 text encoder that expects an actual prompt string.
    Rather than inventing a second, incompatible text pipeline, this rebuilds
    a natural-language prompt from the recovered controls via
    :func:`controls_to_prompt` (e.g. ``[circle_id, red_id, right_id] ->
    "a red circle moving right"``), so a real pretrained text encoder can be
    dropped in without changing any upstream call sites (:class:`LiveSession`,
    :class:`LiveEngine`, the dataset/batcher code) that only ever produce and
    pass around ``text_ids`` tensors.

    Accepts a ``(B, L)`` int tensor/array of token ids; returns a list of
    ``B`` prompt strings.
    """
    if isinstance(text_ids, torch.Tensor):
        text_ids = text_ids.detach().cpu().numpy()
    text_ids = np.asarray(text_ids)
    if text_ids.ndim == 1:
        text_ids = text_ids[None, :]
    out: List[str] = []
    for row in text_ids:
        controls = ids_to_controls(row)
        out.append(controls_to_prompt(controls) if controls else "")
    return out
