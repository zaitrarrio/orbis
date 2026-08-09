"""CPU tests for RealWanVAE against a mock AutoencoderKLWan.

``diffusers`` is not installed in CI/dev sandboxes and there is no GPU here
(mirrors ``tests/test_wan21_real_adapter.py``'s approach for the backbone),
so this builds a small mock model that mirrors the exact public surface of
``diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan`` that
``RealWanVAE`` depends on:

* ``model.config.z_dim`` / ``.latents_mean`` / ``.latents_std`` /
  ``.scale_factor_spatial``
* ``model.encode(x)`` returning an object with ``.latent_dist.mode()``
* ``model.decode(z)`` returning an object with ``.sample``
* ``model.parameters()`` (plain ``nn.Module``)

Values used for ``latents_mean``/``latents_std``/``z_dim`` below are the
*real* Wan2.1 VAE config values (confirmed against the installed
``diffusers`` build on a real GPU box), not arbitrary placeholders --
so the normalization-math tests below exercise the exact transform that
runs against the real pretrained checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from orbis.adapters.wan21_vae import RealWanVAE

# Real Wan2.1-1.3B AutoencoderKLWan config values (see module docstring).
WAN_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
WAN_LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
]
Z_DIM = 16


class _MockLatentDist:
    def __init__(self, z: torch.Tensor):
        self._z = z

    def mode(self) -> torch.Tensor:
        return self._z


class _MockEncodeOutput:
    def __init__(self, z: torch.Tensor):
        self.latent_dist = _MockLatentDist(z)


class _MockDecodeOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class _MockConfig:
    def __init__(self, z_dim: int = Z_DIM):
        self.z_dim = z_dim
        self.latents_mean = list(WAN_LATENTS_MEAN)
        self.latents_std = list(WAN_LATENTS_STD)
        self.scale_factor_spatial = 8
        self.scale_factor_temporal = 4


class _MockAutoencoderKLWan(nn.Module):
    """Mirrors AutoencoderKLWan's causal-VAE-at-T=1 behavior: a (B,3,1,H,W)
    input always yields a (B,z_dim,1,H/8,W/8) latent (real Wan's causal
    grouping never compresses a single input frame), via a real (tiny,
    randomly-initialized) strided conv so shapes/gradients are exercised
    exactly like the real 3D conv stack would be.
    """

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.config = _MockConfig(z_dim)
        self.enc_conv = nn.Conv3d(3, z_dim, kernel_size=(1, 8, 8), stride=(1, 8, 8))
        self.dec_conv = nn.ConvTranspose3d(z_dim, 3, kernel_size=(1, 8, 8), stride=(1, 8, 8))

    def encode(self, x: torch.Tensor) -> _MockEncodeOutput:
        assert x.dim() == 5 and x.shape[2] == 1, f"expected (B,3,1,H,W), got {tuple(x.shape)}"
        z = self.enc_conv(x)
        assert z.shape[2] == 1
        return _MockEncodeOutput(z)

    def decode(self, z: torch.Tensor) -> _MockDecodeOutput:
        assert z.dim() == 5 and z.shape[2] == 1, f"expected (B,C,1,h,w), got {tuple(z.shape)}"
        frames = self.dec_conv(z)
        return _MockDecodeOutput(frames)


def _make_vae(z_dim: int = Z_DIM) -> RealWanVAE:
    return RealWanVAE(model=_MockAutoencoderKLWan(z_dim), latent_channels=z_dim)


def test_encode_decode_shapes():
    vae = _make_vae()
    frames = torch.rand(3, 3, 64, 96)  # (N, 3, H, W) in [0,1]
    z = vae.encode(frames)
    assert z.shape == (3, Z_DIM, 8, 12)
    rec = vae.decode(z)
    assert rec.shape == frames.shape


def test_decode_output_in_unit_range():
    vae = _make_vae()
    z = torch.randn(2, Z_DIM, 4, 4) * 5.0  # exaggerated range
    rec = vae.decode(z)
    assert rec.min() >= 0.0 and rec.max() <= 1.0


def test_encode_matches_manual_normalization():
    """encode() must equal (raw_model_output - latents_mean) / latents_std."""
    vae = _make_vae()
    frames = torch.rand(2, 3, 16, 16)
    z = vae.encode(frames)

    with torch.no_grad():
        x = (frames * 2.0 - 1.0).unsqueeze(2)
        raw = vae.model.encode(x).latent_dist.mode().squeeze(2)
        mean = torch.tensor(WAN_LATENTS_MEAN).view(1, Z_DIM, 1, 1)
        std = torch.tensor(WAN_LATENTS_STD).view(1, Z_DIM, 1, 1)
        expected = (raw - mean) / std
    assert torch.allclose(z, expected, atol=1e-6)


def test_decode_matches_manual_denormalization():
    """decode() must feed model.decode() with latent*std + mean (raw space)."""
    vae = _make_vae()
    z = torch.randn(2, Z_DIM, 4, 4)

    mean = torch.tensor(WAN_LATENTS_MEAN).view(1, Z_DIM, 1, 1)
    std = torch.tensor(WAN_LATENTS_STD).view(1, Z_DIM, 1, 1)
    raw_expected = (z * std + mean).unsqueeze(2)
    with torch.no_grad():
        expected_frames = vae.model.decode(raw_expected).sample.squeeze(2)
        expected_frames = ((expected_frames + 1.0) / 2.0).clamp(0.0, 1.0)

    rec = vae.decode(z)
    assert torch.allclose(rec, expected_frames, atol=1e-6)


def test_round_trip_recovers_shape_and_stays_finite():
    vae = _make_vae()
    frames = torch.rand(2, 3, 32, 32)
    rec, z = vae(frames)
    assert rec.shape == frames.shape
    assert torch.isfinite(z).all()
    assert torch.isfinite(rec).all()


def test_model_is_frozen():
    vae = _make_vae()
    assert all(not p.requires_grad for p in vae.model.parameters())


def test_gradient_flows_through_frozen_decode_to_latent_input():
    """LoRA/generator training needs grad w.r.t. the *latent*, not the
    frozen VAE's own weights -- standard autograd semantics through a
    frozen-weight module, exercised explicitly here."""
    vae = _make_vae()
    z = torch.randn(1, Z_DIM, 4, 4, requires_grad=True)
    rec = vae.decode(z)
    rec.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_state_dict_is_empty_and_load_is_noop():
    vae = _make_vae()
    sd = vae.state_dict()
    assert sd == {}
    # Must not raise even with an arbitrary/garbage dict (frozen base is
    # always reconstructed fresh via from_pretrained(), never loaded).
    vae.load_state_dict({"nonsense.key": torch.zeros(1)})


def test_z_dim_mismatch_raises():
    try:
        RealWanVAE(model=_MockAutoencoderKLWan(z_dim=8), latent_channels=16)
        assert False, "expected ValueError on z_dim mismatch"
    except ValueError as e:
        assert "z_dim" in str(e)


def test_calibrate_is_noop():
    vae = _make_vae()
    before = vae.latent_scale.clone()
    vae.calibrate(torch.rand(2, 3, 16, 16))
    assert torch.equal(vae.latent_scale, before)
