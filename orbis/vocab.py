"""Shared, canonical vocabulary for the toy world and the prompt encoder.

The prompt grammar is intentionally small and *compositional*: a prompt names a
subject (shape), an attribute (color), a motion (direction) and optional
modifiers (speed, size).  Because these controls map onto visible properties of
the rendered video, text->video alignment and mid-rollout prompt switching are
directly observable rather than a matter of faith.

Multilingual synonyms are included so that the "multilingual prompting" claim of
the paper has a concrete (if toy) realization: e.g. ``rojo`` and ``红`` both map
to the canonical ``red`` token.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical tokens
# ---------------------------------------------------------------------------
SHAPES: List[str] = ["circle", "square", "triangle"]

COLORS: Dict[str, Tuple[float, float, float]] = {
    "red": (0.90, 0.15, 0.15),
    "green": (0.15, 0.75, 0.25),
    "blue": (0.20, 0.35, 0.95),
    "yellow": (0.95, 0.85, 0.15),
    "cyan": (0.15, 0.85, 0.85),
    "magenta": (0.90, 0.20, 0.80),
    "white": (0.95, 0.95, 0.95),
}

# Unit-ish velocity vectors in normalized (x right, y down) coordinates.
DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "right": (1.0, 0.0),
    "left": (-1.0, 0.0),
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "upright": (0.7, -0.7),
    "upleft": (-0.7, -0.7),
    "downright": (0.7, 0.7),
    "downleft": (-0.7, 0.7),
}

SPEEDS: Dict[str, float] = {"slow": 0.6, "medium": 1.0, "fast": 1.6}
SIZES: Dict[str, float] = {"small": 0.7, "medium": 1.0, "big": 1.4}

BACKGROUND = (0.05, 0.06, 0.09)  # near-black navy

# ---------------------------------------------------------------------------
# Multilingual / synonym table -> canonical token
# ---------------------------------------------------------------------------
SYNONYMS: Dict[str, str] = {}


def _register(canonical: str, *aliases: str) -> None:
    SYNONYMS[canonical] = canonical
    for a in aliases:
        SYNONYMS[a] = canonical


# shapes (en / es / zh / fr)
_register("circle", "circulo", "círculo", "圆", "圆形", "cercle", "ball", "dot", "disc")
_register("square", "cuadrado", "方", "方形", "carre", "carré", "box")
_register("triangle", "triangulo", "triángulo", "三角", "三角形")

# colors
_register("red", "rojo", "红", "红色", "rouge")
_register("green", "verde", "绿", "绿色", "vert")
_register("blue", "azul", "蓝", "蓝色", "bleu")
_register("yellow", "amarillo", "黄", "黄色", "jaune")
_register("cyan", "cian", "青", "青色")
_register("magenta", "rosa", "品红", "洋红")
_register("white", "blanco", "白", "白色", "blanc")

# directions
_register("right", "derecha", "右", "droite", "rightward", "east")
_register("left", "izquierda", "左", "gauche", "leftward", "west")
_register("up", "arriba", "上", "haut", "upward", "north")
_register("down", "abajo", "下", "bas", "downward", "south")
_register("upright", "up-right", "noreste")
_register("upleft", "up-left", "noroeste")
_register("downright", "down-right", "sureste")
_register("downleft", "down-left", "suroeste")

# modifiers
_register("slow", "lento", "慢", "lent")
_register("medium", "normal", "中")
_register("fast", "rapido", "rápido", "快", "rapide", "quick")
_register("small", "pequeno", "pequeño", "小", "petit", "tiny")
_register("big", "grande", "大", "gros", "large", "huge")

# Full canonical token list (for the embedding table).  Index 0 is reserved for
# padding / unknown.
CANONICAL_TOKENS: List[str] = (
    ["<pad>"]
    + SHAPES
    + list(COLORS.keys())
    + list(DIRECTIONS.keys())
    + list(SPEEDS.keys())
    + list(SIZES.keys())
    + ["moving", "a", "the"]  # ignorable filler tokens, still embedded
)
TOKEN_TO_ID: Dict[str, int] = {t: i for i, t in enumerate(CANONICAL_TOKENS)}
VOCAB_SIZE = len(CANONICAL_TOKENS)
