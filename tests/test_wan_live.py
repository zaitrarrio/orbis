"""Tests for Wan backbone, post-training, serve contracts, consistency."""

from __future__ import annotations

import numpy as np
import torch

from orbis.adapters.factory import build_backbone
from orbis.adapters.wan_adapter import WanAdapter
from orbis.config import OrbisConfig, wan_smoke_config
from orbis.data.video_dataset import (
    MidTrainBatcher,
    SyntheticVideoDataset,
    event_condition_schedule,
    EventCaption,
)
from orbis.engine import LiveEngine
from orbis.posttrain.rewards import RewardBundle, LatentWorldModel
from orbis.serve.drift import DriftState, stabilize_chunk, drift_score
from orbis.serve.fifo import BoundedFIFO
from orbis.session import LiveSession, EncodedPrompt
from orbis.system import OrbisSystem


def test_wan_adapter_shapes_and_context_reuse():
    cfg = wan_smoke_config(seed=0)
    gen = build_backbone(cfg)
    assert isinstance(gen, WanAdapter)
    b, cf = 2, cfg.model.chunk_frames
    lc = cfg.vae.latent_channels
    lh, lw = cfg.latent_hw
    z = torch.randn(b, cf, lc, lh, lw)
    text = torch.zeros(b, cfg.model.text_len, dtype=torch.long)
    hist = torch.randn(b, cfg.model.history_frames, lc, lh, lw)
    ref = torch.randn(b, 1, lc, lh, lw)
    mem = gen.memory.init(b, z.device)
    ctx = gen.encode_context(text, hist, ref, mem)
    v1 = gen.forward(z, torch.ones(b) * 0.5, ctx)
    v2 = gen.forward(z, torch.ones(b) * 0.5, ctx)
    assert v1.shape == z.shape
    assert torch.allclose(v1, v2)


def test_wan_memory_bounded():
    cfg = wan_smoke_config()
    gen = build_backbone(cfg)
    mem = gen.memory.init(1, "cpu")
    lc = cfg.vae.latent_channels
    lh, lw = cfg.latent_hw
    for _ in range(5):
        lat = torch.randn(1, cfg.model.chunk_frames, lc, lh, lw)
        tok = gen.frame_tokens(lat, role=2)
        mem = gen.memory.write(mem, tok)
    assert mem.shape == (1, cfg.model.memory_tokens, cfg.model.dim)


def test_wan_anchor_reference_across_chunks():
    cfg = wan_smoke_config(seed=1)
    assert cfg.backbone.anchor_reference
    sys = OrbisSystem.build(cfg)
    eng = LiveEngine(sys, steps=2, seed=3, device="cpu")
    # I2V with a solid reference frame
    img = np.zeros((cfg.world.height, cfg.world.width, 3), dtype=np.float32)
    img[10:40, 10:40] = (1.0, 0.0, 0.0)
    s = eng.start("a red square moving right", mode="i2v", image=img)
    assert s.reference is not None
    c0 = eng.generate_chunk(s)
    c1 = eng.generate_chunk(s)
    # Reference must still be set after history fills (anchored)
    assert s.reference is not None
    assert c0.frames.shape[0] == cfg.model.chunk_frames
    assert c1.frames.shape[0] == cfg.model.chunk_frames


def test_wan_switch_immutability():
    cfg = wan_smoke_config(seed=2)
    sys = OrbisSystem.build(cfg)
    eng = LiveEngine(sys, steps=2, seed=7, device="cpu")
    base = eng.generate_video(
        "a red circle moving right", n_chunks=4, seed=7)
    eng2 = LiveEngine(sys, steps=2, seed=7, device="cpu")
    switched = eng2.generate_video(
        "a red circle moving right", n_chunks=4, seed=7,
        schedule={2: "a blue square moving up"})
    cf = cfg.model.chunk_frames
    # Pre-boundary frames identical
    assert np.allclose(
        base["frames"][: 2 * cf], switched["frames"][: 2 * cf])
    assert switched["session"].log[2]["version"] >= 1


def test_async_prompt_version_guard():
    cfg = OrbisConfig()
    s = LiveSession(cfg)
    s.set_initial_prompt("a red circle moving right")
    upd = s.set_prompt("moving up")
    # Stale encoding for wrong version must be rejected after admit
    s.admit_pending()
    stale = EncodedPrompt(version=upd.version - 1,
                          ids=torch.zeros(1, cfg.model.text_len),
                          session_epoch=s._session_epoch)
    assert s.apply_async_encoding(stale) is False
    good = EncodedPrompt(version=s.active_version,
                         ids=s.active_ids.clone(),
                         session_epoch=s._session_epoch)
    assert s.apply_async_encoding(good) is True


def test_bounded_fifo_no_drop_when_drained():
    q = BoundedFIFO(capacity=2)
    assert q.push("a")
    assert q.push("b")
    assert q.full
    # capacity-full push drops oldest unless consumer drains first
    q.pop()
    assert q.push("c")
    assert list(q.drain()) == ["b", "c"]


def test_drift_stabilize_conservative():
    state = DriftState()
    a = np.ones((4, 8, 8, 3), dtype=np.float32) * 0.2
    b = np.ones((4, 8, 8, 3), dtype=np.float32) * 0.9
    out0 = stabilize_chunk(a, state, enabled=True)
    assert out0 is a or np.allclose(out0, a)
    out1 = stabilize_chunk(b, state, threshold=0.1, blend=0.5, enabled=True)
    # Corrected toward previous darker mean
    assert out1[:2].mean() < b.mean()


def test_world_model_and_rewards():
    cfg = wan_smoke_config()
    lh, lw = cfg.latent_hw
    rb = RewardBundle(
        cfg.model.dim, cfg.vae.latent_channels,
        cfg.model.chunk_frames, (lh, lw))
    b = 2
    chunk = torch.randn(b, cfg.model.chunk_frames, cfg.vae.latent_channels, lh, lw)
    target = torch.randn_like(chunk)
    text = torch.randn(b, cfg.model.dim)
    hist = torch.randn(b, 2, cfg.vae.latent_channels, lh, lw)
    mem = torch.randn(b, cfg.model.memory_tokens, cfg.model.dim)
    ref = torch.randn(b, 1, cfg.vae.latent_channels, lh, lw)
    total, parts = rb(chunk, target, text, hist, mem, ref)
    assert total.shape == (b,)
    assert "wm" in parts and "ref" in parts


def test_event_schedule_and_mid_batch():
    events = [
        EventCaption(0.0, "a red circle moving right"),
        EventCaption(1.0, "a blue circle moving right"),
    ]
    sched = event_condition_schedule(events, fps=8, chunk_frames=2)
    assert 4 in sched or 3 in sched or len(sched) >= 1
    cfg = wan_smoke_config(seed=5)
    sys = OrbisSystem.build(cfg)
    ds = SyntheticVideoDataset(cfg, seed=5)
    batcher = MidTrainBatcher(cfg, sys.vae, ds, seed=5)
    batch = batcher.training_batch(2, mode="event")
    assert batch["target"].shape[0] == 2
    assert batch["text_ids"].ndim == 2


def test_system_wan_checkpoint_roundtrip(tmp_path):
    cfg = wan_smoke_config(seed=9)
    sys = OrbisSystem.build(cfg)
    # Mutate a trainable tensor so roundtrip is meaningful.
    with torch.no_grad():
        sys.generator.text_embed.weight[1].add_(0.123)
        if hasattr(sys.generator, "blocks") and len(sys.generator.blocks):
            sys.generator.blocks[0].ada[-1].bias.add_(0.05)
    p = tmp_path / "wan.pt"
    sys.save(str(p))
    loaded = OrbisSystem.load(str(p), map_location="cpu")
    assert loaded.cfg.backbone.type == "wan"
    assert isinstance(loaded.generator, WanAdapter)
    assert torch.allclose(
        sys.generator.text_embed.weight, loaded.generator.text_embed.weight)
    assert torch.allclose(
        sys.generator.blocks[0].ada[-1].bias,
        loaded.generator.blocks[0].ada[-1].bias)


def test_wan_stub_blocks_are_trainable():
    """Stub must train DiT blocks — zero-init gates otherwise stay dead."""
    cfg = wan_smoke_config(seed=0)
    sys = OrbisSystem.build(cfg)
    trainable = {n for n, p in sys.generator.named_parameters() if p.requires_grad}
    assert any(n.startswith("blocks.0.ada") for n in trainable)
    assert any("cross" in n for n in trainable)
