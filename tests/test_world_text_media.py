import numpy as np

from orbis.world import SceneSpec, rollout, render_frame, sample_scene
from orbis.text import parse_controls, controls_to_prompt, PromptTokenizer
from orbis import media


def test_rollout_deterministic_and_moves():
    spec = SceneSpec(shape="circle", color="red", direction="right")
    f1, _ = rollout(spec, 8, 32, 32)
    f2, _ = rollout(spec, 8, 32, 32)
    assert np.array_equal(f1, f2)
    # object should move: frames not all identical
    assert not np.allclose(f1[0], f1[-1])


def test_prompt_switch_changes_color():
    spec = SceneSpec(color="red", direction="right")
    frames, _ = rollout(spec, 12, 32, 32, control_schedule={6: {"color": "blue"}})
    r_before = frames[3].reshape(-1, 3).max(0)
    r_after = frames[9].reshape(-1, 3).max(0)
    assert r_before[0] > r_before[2]      # red dominant before
    assert r_after[2] > r_after[0]        # blue dominant after


def test_shapes_are_distinct():
    a = render_frame(SceneSpec(shape="circle"), 32, 32)
    b = render_frame(SceneSpec(shape="square"), 32, 32)
    assert not np.allclose(a, b)


def test_multilingual_parse():
    assert parse_controls("rojo cuadrado")["color"] == "red"
    assert parse_controls("蓝色三角形")["shape"] == "triangle"
    assert parse_controls("un cercle vert rapide")["speed"] == "fast"


def test_rolling_summary_merge():
    merged = parse_controls("moving up blue",
                            defaults={"shape": "triangle", "color": "red",
                                      "direction": "right"})
    assert merged["shape"] == "triangle"     # persisted
    assert merged["color"] == "blue"          # updated
    assert merged["direction"] == "up"


def test_tokenizer_padding_and_stability():
    tok = PromptTokenizer(8)
    a = tok.encode("a red circle moving right")
    b = tok.encode("circle red right")        # same meaning, different order
    assert a.shape == (8,)
    assert np.array_equal(a, b)               # order-independent canonicalisation


def test_png_and_gif_headers():
    frames = [render_frame(SceneSpec(color=c), 32, 32)
              for c in ("red", "green", "blue")]
    png = media.encode_png(frames[0])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    gif = media.encode_gif(frames, fps=8, scale=2)
    assert gif[:6] == b"GIF89a"
    assert gif[-1:] == b"\x3b"                 # trailer


def test_gif_lzw_roundtrip_full_frame():
    """Regression: early code-size bump made viewers show a single scanline."""
    try:
        from PIL import Image
    except ImportError:
        import pytest
        pytest.skip("Pillow not installed")

    frames = []
    for i in range(6):
        f = np.zeros((32, 32, 3), np.float32)
        f[8:24, (i * 3) % 20:(i * 3) % 20 + 10] = (1.0, 0.2, 0.1)
        frames.append(f)
    # scale=12 → 384², large enough to exercise 10–12-bit LZW + clears
    gif = media.encode_gif(frames, fps=8, scale=12)
    from io import BytesIO
    im = Image.open(BytesIO(gif))
    assert im.size == (384, 384)
    for i in range(im.n_frames):
        im.seek(i)
        arr = np.asarray(im.convert("RGB"))
        ys, xs = np.where(arr.sum(-1) > 40)
        assert len(ys) > 1000, f"frame {i} too sparse (broken LZW?)"
        assert len(set(ys.tolist())) > 50, f"frame {i} looks like a single line"
