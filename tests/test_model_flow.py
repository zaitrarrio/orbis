import torch

from orbis.config import OrbisConfig
from orbis.flow import RectifiedFlow
from orbis.model import Generator, patchify, unpatchify
from orbis.system import OrbisSystem


def test_patchify_roundtrip():
    x = torch.randn(2, 3, 8, 8, 8)   # (B,F,C,H,W)
    p = 2
    tok = patchify(x, p)
    assert tok.shape == (2, 3, 16, 32)
    back = unpatchify(tok, p, 8, 4, 4)
    assert torch.allclose(back, x, atol=1e-5)


def test_flow_interpolation_endpoints():
    flow = RectifiedFlow()
    z = torch.randn(4, 3)
    noise = torch.randn(4, 3)
    z0 = flow.interpolate(z, noise, torch.zeros(4))
    z1 = flow.interpolate(z, noise, torch.ones(4))
    assert torch.allclose(z0, z, atol=1e-5)
    assert torch.allclose(z1, noise, atol=1e-5)
    assert torch.allclose(flow.target(z, noise), noise - z)


def test_flow_sample_constant_velocity():
    # Integrating a constant velocity c from sigma=1 to 0 gives noise - c.
    flow = RectifiedFlow()
    noise = torch.randn(2, 5)
    c = torch.full((2, 5), 0.3)
    out = flow.sample(lambda z, s: c, (2, 5), noise.device, steps=8, noise=noise)
    assert torch.allclose(out, noise - c, atol=1e-5)


def test_generator_shapes_and_context_reuse():
    cfg = OrbisConfig()
    gen = Generator(cfg.model, cfg.vae).build(cfg.latent_hw)
    lh, lw = cfg.latent_hw
    lc = cfg.vae.latent_channels
    B, T = 2, cfg.model.chunk_frames
    z = torch.randn(B, T, lc, lh, lw)
    sigma = torch.rand(B)
    ids = torch.randint(0, 20, (B, cfg.model.text_len))
    hist = torch.randn(B, cfg.model.history_frames, lc, lh, lw)
    mem = gen.memory.init(B, z.device)
    ctx = gen.encode_context(ids, hist, None, mem)
    v1 = gen.forward(z, sigma, ctx)
    v2 = gen.velocity(z, sigma, ids, history=hist, memory_state=mem)
    assert v1.shape == z.shape
    assert torch.allclose(v1, v2, atol=1e-6)   # context-reuse == one-shot


def test_memory_is_bounded():
    cfg = OrbisConfig()
    gen = Generator(cfg.model, cfg.vae).build(cfg.latent_hw)
    lh, lw = cfg.latent_hw
    lc = cfg.vae.latent_channels
    state = gen.memory.init(1, torch.device("cpu"))
    n_mem = cfg.model.memory_tokens
    for _ in range(20):                        # many writes
        frames = torch.randn(1, cfg.model.chunk_frames, lc, lh, lw)
        tokens = gen.frame_tokens(frames, role=2)
        state = gen.memory.write(state, tokens)
        assert state.shape == (1, n_mem, cfg.model.dim)   # fixed budget


def test_checkpoint_roundtrip(tmp_path):
    sysm = OrbisSystem.build(OrbisConfig())
    p = tmp_path / "ck.pt"
    sysm.save(str(p))
    loaded = OrbisSystem.load(str(p), map_location="cpu")
    for a, b in zip(sysm.generator.parameters(), loaded.generator.parameters()):
        assert torch.allclose(a, b)
