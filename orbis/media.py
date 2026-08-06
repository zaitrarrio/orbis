"""Zero-dependency image / video export (PNG and animated GIF).

We cannot rely on Pillow or imageio being installed, so both encoders are
implemented on top of the standard library only (``zlib`` for PNG, a hand-rolled
LZW coder for GIF).  Frames are ``float [0,1]`` or ``uint8`` ``(H, W, 3)``
arrays; helpers upscale with nearest-neighbour so tiny toy frames are viewable.
"""

from __future__ import annotations

import struct
import zlib
from typing import List, Sequence

import numpy as np


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def upscale_nearest(img: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return img
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png(img: np.ndarray) -> bytes:
    img = to_uint8(img)
    h, w, _ = img.shape
    # prepend a zero filter byte to each scanline
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (sig + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(raw, 9))
            + _png_chunk(b"IEND", b""))


def save_png(path: str, img: np.ndarray, scale: int = 1) -> None:
    img = upscale_nearest(to_uint8(img), scale)
    with open(path, "wb") as f:
        f.write(encode_png(img))


# ---------------------------------------------------------------------------
# Animated GIF (GIF89a + LZW)
# ---------------------------------------------------------------------------
_R_LEVELS, _G_LEVELS, _B_LEVELS = 6, 7, 6  # 252 palette entries (<=256)


def _build_palette() -> np.ndarray:
    pal = []
    for r in range(_R_LEVELS):
        for g in range(_G_LEVELS):
            for b in range(_B_LEVELS):
                pal.append((round(r * 255 / (_R_LEVELS - 1)),
                            round(g * 255 / (_G_LEVELS - 1)),
                            round(b * 255 / (_B_LEVELS - 1))))
    while len(pal) < 256:
        pal.append((0, 0, 0))
    return np.array(pal, dtype=np.uint8)


_PALETTE = _build_palette()


def _quantize(img: np.ndarray) -> np.ndarray:
    img = to_uint8(img).astype(np.int32)
    ri = np.round(img[..., 0] * (_R_LEVELS - 1) / 255).astype(np.int32)
    gi = np.round(img[..., 1] * (_G_LEVELS - 1) / 255).astype(np.int32)
    bi = np.round(img[..., 2] * (_B_LEVELS - 1) / 255).astype(np.int32)
    return (ri * (_G_LEVELS * _B_LEVELS) + gi * _B_LEVELS + bi).astype(np.uint8)


def _lzw_encode(indices: np.ndarray, min_code_size: int) -> bytes:
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1
    code_size = min_code_size + 1
    table = {(i,): i for i in range(clear_code)}
    next_code = eoi_code + 1

    out_bits = 0
    out_nbits = 0
    out = bytearray()

    def emit(code: int, nbits: int):
        nonlocal out_bits, out_nbits
        out_bits |= code << out_nbits
        out_nbits += nbits
        while out_nbits >= 8:
            out.append(out_bits & 0xFF)
            out_bits >>= 8
            out_nbits -= 8

    emit(clear_code, code_size)
    data = indices.tobytes()
    cur = (data[0],)
    for b in data[1:]:
        nxt = cur + (b,)
        if nxt in table:
            cur = nxt
        else:
            emit(table[cur], code_size)
            table[nxt] = next_code
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
            if next_code >= 4096:
                emit(clear_code, code_size)
                table = {(i,): i for i in range(clear_code)}
                next_code = eoi_code + 1
                code_size = min_code_size + 1
            cur = (b,)
    emit(table[cur], code_size)
    emit(eoi_code, code_size)
    if out_nbits > 0:
        out.append(out_bits & 0xFF)

    # chunk into <=255 byte sub-blocks
    blocks = bytearray()
    for i in range(0, len(out), 255):
        chunk = out[i:i + 255]
        blocks.append(len(chunk))
        blocks.extend(chunk)
    blocks.append(0)
    return bytes(blocks)


def encode_gif(frames: Sequence[np.ndarray], fps: int = 8, scale: int = 1,
               loop: bool = True) -> bytes:
    frames = [upscale_nearest(to_uint8(f), scale) for f in frames]
    h, w, _ = frames[0].shape
    delay = max(1, round(100 / fps))  # in 1/100 s

    out = bytearray(b"GIF89a")
    out += struct.pack("<HH", w, h)
    out += bytes([0xF7, 0, 0])  # global color table, 256 entries, 8 bpp
    out += _PALETTE.tobytes()
    if loop:
        out += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00"

    min_code_size = 8
    for f in frames:
        idx = _quantize(f)
        out += b"\x21\xF9\x04\x00" + struct.pack("<H", delay) + b"\x00\x00"
        out += b"\x2C" + struct.pack("<HHHH", 0, 0, w, h) + b"\x00"
        out += bytes([min_code_size])
        out += _lzw_encode(idx, min_code_size)
    out += b"\x3B"
    return bytes(out)


def save_gif(path: str, frames: Sequence[np.ndarray], fps: int = 8,
             scale: int = 1) -> None:
    with open(path, "wb") as f:
        f.write(encode_gif(frames, fps=fps, scale=scale))


def png_data_uri(img: np.ndarray, scale: int = 1) -> str:
    import base64
    png = encode_png(upscale_nearest(to_uint8(img), scale))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
