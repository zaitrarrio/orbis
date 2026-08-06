import numpy as np
import torch

from orbis.config import OrbisConfig
from orbis.engine import LiveEngine
from orbis.session import LiveSession
from orbis.system import OrbisSystem


def _small_system():
    cfg = OrbisConfig()
    cfg.model.depth = 2                # keep tests fast
    return OrbisSystem.build(cfg)


def test_prompt_versioning_admitted_at_boundary():
    cfg = OrbisConfig()
    s = LiveSession(cfg)
    s.set_initial_prompt("a red circle moving right")
    assert s.active_controls["color"] == "red"
    upd = s.set_prompt("a blue square moving up")
    # queued but NOT yet active
    assert s.pending is not None
    assert s.active_controls["color"] == "red"
    admitted = s.admit_pending()
    assert admitted == upd.version
    assert s.active_controls["color"] == "blue"
    assert s.pending is None


def test_rolling_summary_partial_update():
    s = LiveSession(OrbisConfig())
    s.set_initial_prompt("a red triangle moving right")
    s.set_prompt("moving up")          # only names direction
    s.admit_pending()
    assert s.active_controls["shape"] == "triangle"   # persisted
    assert s.active_controls["color"] == "red"        # persisted
    assert s.active_controls["direction"] == "up"     # updated


def test_switch_only_affects_uncommitted_chunks():
    # The core, training-independent contract: a switch queued at a boundary is
    # admitted there, already-delivered frames are byte-identical, and the
    # version log flips exactly at the boundary.
    eng = LiveEngine(_small_system(), steps=2, seed=7)
    cf = eng.cfg.model.chunk_frames
    boundary = 3
    a = eng.generate_video("a red circle moving right", n_chunks=6, seed=7)
    eng2 = LiveEngine(eng.system, steps=2, seed=7)
    b = eng2.generate_video("a red circle moving right", n_chunks=6, seed=7,
                            schedule={boundary: "a blue square moving up"})
    fa, fb = a["frames"], b["frames"]
    # frames before the boundary are byte-identical (delivered => immutable)
    assert np.allclose(fa[:boundary * cf], fb[:boundary * cf])
    # version log: v0 before the boundary, the switch version from the boundary on
    log = b["log"]
    assert all(e["version"] == 0 for e in log[:boundary])
    assert all(e["version"] == log[boundary]["version"] for e in log[boundary:])
    assert log[boundary]["version"] > 0
    assert log[boundary]["admitted"] == log[boundary]["version"]


def test_trained_text_conditioning_changes_output():
    # With trained weights, different prompts must yield different video. Uses the
    # committed checkpoint if present; skipped otherwise so the suite stays fast.
    import os
    import pytest
    if not os.path.exists("orbis.pt"):
        pytest.skip("no trained checkpoint (run `orbis train`)")
    eng = LiveEngine(OrbisSystem.load("orbis.pt"), seed=1)
    red = eng.generate_video("a red circle moving right", n_chunks=2, seed=1)
    blue = eng.generate_video("a blue square moving down", n_chunks=2, seed=1)
    assert not np.allclose(red["frames"], blue["frames"], atol=1e-3)


def test_streaming_state_is_bounded():
    eng = LiveEngine(_small_system(), steps=2, seed=0)
    s = eng.start("a green square moving down")
    hf = eng.cfg.model.history_frames
    n_mem = eng.cfg.model.memory_tokens
    for _ in range(12):
        eng.generate_chunk(s)
        assert s.history.shape[1] <= hf              # recency window bounded
        assert s.memory_state.shape[1] == n_mem      # memory budget fixed


def test_modes_run_and_shapes():
    eng = LiveEngine(_small_system(), steps=2, seed=0)
    H, W = eng.cfg.world.height, eng.cfg.world.width
    # T2V
    c = eng.generate_chunk(eng.start("a red circle moving right"))
    assert c.frames.shape == (eng.cfg.model.chunk_frames, H, W, 3)
    # I2V
    img = np.random.rand(H, W, 3).astype(np.float32)
    c = eng.generate_chunk(eng.start("a red circle", mode="i2v", image=img))
    assert c.frames.shape[1:] == (H, W, 3)
    # V2V
    vid = np.random.rand(6, H, W, 3).astype(np.float32)
    c = eng.generate_chunk(eng.start("a red circle", mode="v2v", video=vid))
    assert c.frames.shape[1:] == (H, W, 3)


def test_superres_upscales():
    eng = LiveEngine(_small_system(), steps=2, seed=0)
    c = eng.generate_chunk(eng.start("a red circle moving right"))
    hr = eng.upscale(c.frames)
    scale = eng.cfg.sr.scale
    assert hr.shape[1] == c.frames.shape[1] * scale
    assert hr.shape[2] == c.frames.shape[2] * scale
