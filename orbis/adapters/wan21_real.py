"""Real Wan2.1-1.3B backbone: frozen pretrained transformer + LoRA.

This is the "highest fidelity" alternative to ``adapters/wan_adapter.py``'s
``WanAdapter``, which trains a *structurally similar but randomly
initialized* custom DiT at Wan-scale tensor sizes. ``WanAdapter.try_load_hf``
loads a real ``diffusers.WanPipeline`` but, by its own docstring, never maps
any of its tensors into the custom stub ("Full tensor remap is
version-sensitive") -- so today the stub never actually benefits from
pretraining.

``RealWanBackbone`` instead wraps Wan's own real ``WanTransformer3DModel``
directly and drives it with orbis's memory/history/reference conditioning,
rather than attempting a brittle weight transplant into a differently shaped
architecture. Concretely:

* Wan2.1's block is structurally different from ``orbis.modules.DiTBlock``
  (3D RoPE self-attention + learned ``scale_shift_table`` AdaLN-style
  modulation vs this repo's learned absolute positions + a 7-way AdaLN-Zero
  MLP), so a tensor-level remap would silently produce a model that runs but
  computes something unrelated to what Wan was actually trained to do. There
  is no safe partial remap here -- either use Wan's real block verbatim, or
  don't claim to use pretrained Wan weights at all.
* Reference + history + the noised chunk are concatenated along the
  *temporal frame axis* of ``hidden_states`` before Wan's own
  ``patch_embedding``, so Wan's native 3D RoPE self-attention naturally
  attends across committed context and the chunk being denoised -- no
  changes are needed inside ``WanTransformerBlock`` itself. Only the
  chunk-frame slice of the output is used for the flow loss (the
  already-committed context frames are read-only conditioning, matching the
  paper's "H_k is committed, clean content" semantics).
* orbis's persistent ``MemoryBank`` tokens are projected into Wan's own
  text/cross-attention space and concatenated onto the real UMT5 prompt
  embedding sequence, so persistent memory rides Wan's *existing*
  cross-attention path instead of requiring a second attention mechanism.
* Wan2.1 is itself trained with a flow-matching objective
  (``FlowMatchEulerDiscreteScheduler``) using the same rectified-flow
  parameterization as ``orbis.flow.RectifiedFlow`` (predict ``noise - z``
  along a straight-line path) -- so orbis's own flow trainer/sampler can
  drive Wan's transformer directly; only the timestep needs rescaling from
  orbis's ``sigma in [0, 1]`` into Wan's ``num_train_timesteps`` space.
* The frozen base transformer's own weights are pretrained; orbis only
  trains LoRA adapters on ``attn1``/``attn2``/``ffn`` projections plus the
  small memory-bank and cross-attention projection heads described above.

Known, explicitly-scoped fidelity gap (documented rather than silently
glossed over): this phase reuses orbis's own already-trained ``ConvVAE``
(``orbis/vae.py``) for encode/decode -- NOT Wan's own ``AutoencoderKLWan``.
The channel count is made to match by construction
(``wan21_real_config()`` sets ``VAEConfig.latent_channels=16``, mirroring
Wan's transformer ``in_channels=out_channels=16``), so the real transformer
receives correctly-shaped input, but orbis's VAE was trained independently
and its latent *distribution* will differ from what Wan's transformer was
actually pretrained against. The LoRA fine-tune's job includes adapting the
frozen transformer to this distribution shift -- a materially smaller domain
gap than training from random initialization, but not the same as feeding
Wan its own native VAE latents end-to-end. Swapping in the real
``AutoencoderKLWan`` for the full pipeline is intentionally left for a
follow-up phase (see PR description) to keep this change reviewable.

Requires the optional ``wan`` extra (``uv sync --extra wan``) and, for
anything beyond CPU shape/plumbing tests, a real GPU to download and run the
1.3B checkpoint -- see ``deploy/README.md``. CPU unit tests
(``tests/test_wan21_real_adapter.py``) exercise every code path in this
module (LoRA injection, frame concatenation/slicing, memory-token
cross-attention injection, checkpoint round-trip) against a tiny mock
transformer that mirrors ``diffusers``' real ``WanTransformerBlock`` module
layout (``attn1``/``attn2`` with ``to_q``/``to_k``/``to_v``/``to_out``,
``ffn.net``), so the exact same traversal/injection code is expected to work
unmodified against the true ``diffusers.WanTransformer3DModel``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from orbis.backbone import LoRALinear, ModelContext
from orbis.config import BackboneConfig, ModelConfig, VAEConfig
from orbis.memory import MemoryBank
from orbis.model import patchify
from orbis.text import ids_to_prompt_strings

ROLE_MEMORY, ROLE_REF, ROLE_HISTORY, ROLE_CHUNK = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Generic, version-resilient LoRA injection over a real diffusers Wan module.
#
# Rather than hardcoding "blocks.{i}.attn1.to_q" style paths against one
# exact diffusers release (the brittleness `WanAdapter.try_load_hf` itself
# flagged as the reason it never attempted a remap), this walks the actual
# module tree and matches on structural shape (an `nn.ModuleList` of blocks
# each exposing `.attn1`/`.attn2` with `.to_q/.to_k/.to_v/.to_out`, and an
# `.ffn.net`), which is stable across diffusers' Wan/Flux/SD3-style DiT
# implementations even as exact class names or file layouts change.
# ---------------------------------------------------------------------------


def _find_transformer_blocks(transformer: nn.Module) -> nn.ModuleList:
    """Locate the ``nn.ModuleList`` of per-layer Wan transformer blocks."""
    for _, child in transformer.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            first = child[0]
            if hasattr(first, "attn1") and hasattr(first, "attn2"):
                return child
    raise RuntimeError(
        "Could not locate the Wan transformer block list (expected an "
        "nn.ModuleList of blocks each exposing .attn1/.attn2 self-/"
        "cross-attention submodules). diffusers' internal Wan module layout "
        "may have changed; update "
        "orbis.adapters.wan21_real._find_transformer_blocks().")


def _lora_targets_for_block(block: nn.Module) -> List[str]:
    """Dotted-path suffixes (relative to one block) worth LoRA-adapting."""
    targets: List[str] = []
    for attn_name in ("attn1", "attn2"):
        attn = getattr(block, attn_name, None)
        if attn is None:
            continue
        for proj_name in ("to_q", "to_k", "to_v"):
            proj = getattr(attn, proj_name, None)
            if isinstance(proj, nn.Linear):
                targets.append(f"{attn_name}.{proj_name}")
        to_out = getattr(attn, "to_out", None)
        if to_out is not None and len(to_out) > 0 and isinstance(to_out[0], nn.Linear):
            targets.append(f"{attn_name}.to_out.0")
    ffn = getattr(block, "ffn", None)
    net = getattr(ffn, "net", None) if ffn is not None else None
    if net is not None:
        for idx, sub in enumerate(net):
            if isinstance(sub, nn.Linear):
                targets.append(f"ffn.net.{idx}")
            elif isinstance(getattr(sub, "proj", None), nn.Linear):
                # GEGLU-style activation wraps its linear projection in `.proj`.
                targets.append(f"ffn.net.{idx}.proj")
    return targets


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def apply_lora_to_wan_blocks(
    blocks: nn.ModuleList, rank: int, alpha: float,
) -> List[str]:
    """Wrap attn/ffn projections in every block with :class:`LoRALinear`.

    Returns the list of wrapped dotted paths (``"{block_idx}.{suffix}"``) for
    logging/testing. No-op (returns ``[]``) if ``rank <= 0``.
    """
    if rank <= 0:
        return []
    wrapped: List[str] = []
    for i, block in enumerate(blocks):
        for suffix in _lora_targets_for_block(block):
            path = f"{i}.{suffix}"
            base = _get_submodule(blocks, path)
            if not isinstance(base, nn.Linear):
                continue
            _set_submodule(blocks, path, LoRALinear(base, rank=rank, alpha=alpha))
            wrapped.append(path)
    return wrapped


class RealWanBackbone(nn.Module):
    """Drives a real, frozen ``WanTransformer3DModel`` + LoRA + orbis state.

    See module docstring for the full integration design and its explicitly
    scoped-out gap (orbis's own VAE, not Wan's native one, this phase).

    Implements the same surface the live engine depends on
    (:class:`orbis.backbone.VideoBackbone`): ``frame_tokens``,
    ``encode_context``, ``forward``, plus ``velocity``/``trainable_parameters``
    for parity with :class:`orbis.adapters.wan_adapter.WanAdapter`.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        vae_cfg: VAEConfig,
        backbone_cfg: BackboneConfig,
        latent_hw: Tuple[int, int],
        transformer: nn.Module,
        text_encoder_fn: Optional[Callable[[List[str]], torch.Tensor]] = None,
        wan_text_dim: int = 4096,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = model_cfg
        self.backbone_cfg = backbone_cfg
        self.vae_cfg = vae_cfg
        self.latent_hw = latent_hw
        self.wan_text_dim = wan_text_dim
        self.num_train_timesteps = (
            num_train_timesteps or backbone_cfg.wan_num_train_timesteps)

        self.transformer = transformer
        # Kept outside the module tree deliberately: frozen, not checkpointed
        # (always reloaded fresh from `checkpoint_path`/`text_encoder_path`),
        # and callable with plain prompt strings.
        self._text_encoder_fn = text_encoder_fn

        # -- orbis-side bookkeeping (independent of Wan's internal hidden
        # size; used only for the MemoryBank and its projection into Wan's
        # cross-attention space) --------------------------------------------
        p = max(model_cfg.patch_size, 1)
        self.patch = p
        self.latent_channels = vae_cfg.latent_channels
        self.patch_dim = vae_cfg.latent_channels * p * p
        lh, lw = latent_hw
        gh, gw = max(lh // p, 1), max(lw // p, 1)

        self.memory = MemoryBank(model_cfg.dim, model_cfg.memory_tokens, model_cfg.heads)
        self._mem_patch_embed = nn.Linear(self.patch_dim, model_cfg.dim)
        self._mem_spatial_pos = nn.Parameter(torch.randn(gh * gw, model_cfg.dim) * 0.02)
        self._mem_temporal_pos = nn.Embedding(256, model_cfg.dim)
        self._mem_role_embed = nn.Embedding(4, model_cfg.dim)
        # Persistent memory tokens ride Wan's own cross-attention path
        # alongside the real UMT5 prompt embedding.
        self.mem_to_wan = nn.Sequential(
            nn.LayerNorm(model_cfg.dim),
            nn.Linear(model_cfg.dim, wan_text_dim),
        )

        self._lora_paths: List[str] = []
        self._freeze_and_apply_lora(backbone_cfg.lora_rank, backbone_cfg.lora_alpha)

    # -- setup ----------------------------------------------------------------
    def _freeze_and_apply_lora(self, rank: int, alpha: float) -> None:
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        blocks = _find_transformer_blocks(self.transformer)
        self._lora_paths = apply_lora_to_wan_blocks(blocks, rank, alpha)

    # -- checkpoint IO: keep only the trainable delta ------------------------
    #
    # The frozen base transformer is multi-GB; re-serializing it into every
    # orbis checkpoint would make iteration on a rented GPU box painfully
    # slow. Since `OrbisSystem.save/load` call `generator.state_dict()` /
    # `generator.load_state_dict()` unconditionally, overriding them here
    # (rather than touching `system.py`) makes checkpoints for this backbone
    # naturally lean: only LoRA deltas + memory/projection heads are saved.
    # A fresh `RealWanBackbone.from_pretrained(...)` always recreates the
    # identical frozen base before the lean state is overlaid on load.
    def _trainable_key_set(self) -> set:
        return {n for n, p in self.named_parameters() if p.requires_grad}

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        full = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        keep = {prefix + k for k in self._trainable_key_set()}
        return {k: v for k, v in full.items() if k in keep}

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        # `state_dict` is the lean (trainable-only) bag produced above; the
        # frozen base weights already came from `from_pretrained`, so a
        # strict load would always fail on the (intentionally) missing keys.
        return super().load_state_dict(state_dict, strict=False)

    def lora_state_dict(self) -> dict:
        """Back-compat with `WanAdapter`'s explicit LoRA-bag save path."""
        return self.state_dict()

    def load_lora_state_dict(self, state: dict) -> None:
        self.load_state_dict(state, strict=False)

    def trainable_parameters(self):
        for _, p in self.named_parameters():
            if p.requires_grad:
                yield p

    # -- orbis-side frame tokens (MemoryBank bookkeeping only) ---------------
    def frame_tokens(self, latents: torch.Tensor, role: int,
                     frame_offset: int = 0) -> torch.Tensor:
        """Tokenize latents at orbis's own (small) dim, for the memory bank.

        The real Wan transformer never sees these tokens directly -- it
        consumes raw latent frames through its own patch embedding (see
        :meth:`forward`). This exists so ``LiveEngine._write_memory`` (which
        calls ``gen.frame_tokens(...)`` to consolidate evicted history into
        the bounded :class:`MemoryBank`) keeps working unchanged against this
        backbone, exactly as it does against :class:`WanAdapter`.
        """
        b, f, c, lh, lw = latents.shape
        p = self.patch
        gh, gw = max(lh // p, 1), max(lw // p, 1)
        tok = self._mem_patch_embed(patchify(latents, p))
        tok = tok + self._mem_spatial_pos.view(1, 1, gh * gw, -1)
        idx = torch.arange(frame_offset, frame_offset + f, device=latents.device)
        idx = idx.clamp(max=255)
        tok = tok + self._mem_temporal_pos(idx).view(1, f, 1, -1)
        tok = tok + self._mem_role_embed.weight[role].view(1, 1, 1, -1)
        return tok.reshape(b, f * gh * gw, -1)

    # -- context / forward -----------------------------------------------------
    def encode_context(
        self,
        text_ids: torch.Tensor,
        history: Optional[torch.Tensor],
        reference: Optional[torch.Tensor],
        memory_state: Optional[torch.Tensor],
    ) -> ModelContext:
        if self._text_encoder_fn is None:
            raise RuntimeError(
                "RealWanBackbone has no text encoder attached; build via "
                "RealWanBackbone.from_pretrained(...) or pass a "
                "`text_encoder_fn=` (used by tests with a mock encoder).")
        b = text_ids.shape[0]
        device = text_ids.device
        prompts = ids_to_prompt_strings(text_ids)
        text_embeds = self._text_encoder_fn(prompts).to(device)  # (B, L, wan_text_dim)

        groups = [text_embeds]
        if memory_state is not None:
            mem = self.memory.read(memory_state)     # (B, n_mem, orbis_dim)
            mem = self.mem_to_wan(mem)                # (B, n_mem, wan_text_dim)
            groups.insert(0, mem)
        encoder_hidden_states = torch.cat(groups, dim=1)

        has_reference = reference is not None and reference.shape[1] > 0
        extras = {
            "encoder_hidden_states": encoder_hidden_states,
            "history": history if (history is not None and history.shape[1] > 0) else None,
            "reference": reference if has_reference else None,
        }
        # ctx_tokens/text_kv/pooled_text are unused by this backbone's own
        # forward() (real conditioning lives in `extras`); kept zero-shaped
        # only so `ModelContext` stays a valid, introspectable dataclass for
        # any shared/generic call site.
        zero_ctx = torch.zeros(b, 0, self.cfg.dim, device=device)
        zero_text_kv = torch.zeros(b, 1, self.cfg.dim, device=device)
        zero_pooled = torch.zeros(b, self.cfg.dim, device=device)
        return ModelContext(ctx_tokens=zero_ctx, text_kv=zero_text_kv,
                             pooled_text=zero_pooled, batch=b, extras=extras)

    def forward(self, z_noised: torch.Tensor, sigma: torch.Tensor,
                context: ModelContext) -> torch.Tensor:
        extras = context.extras or {}
        encoder_hidden_states = extras.get("encoder_hidden_states")
        if encoder_hidden_states is None:
            raise RuntimeError(
                "ModelContext.extras missing 'encoder_hidden_states'; call "
                "encode_context() first (RealWanBackbone.forward cannot be "
                "driven from a stub-style ctx_tokens/text_kv context).")
        history = extras.get("history")
        reference = extras.get("reference")

        frame_groups = []
        if reference is not None:
            frame_groups.append(reference)
        if history is not None:
            frame_groups.append(history)
        n_ctx_frames = sum(g.shape[1] for g in frame_groups)
        frame_groups.append(z_noised)
        # orbis latents are (B, F, C, H, W); Wan expects (B, C, F, H, W).
        cat = torch.cat(frame_groups, dim=1)
        hidden_states = cat.permute(0, 2, 1, 3, 4).contiguous()

        timestep = sigma.reshape(-1).to(hidden_states.dtype) * self.num_train_timesteps

        out = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = out.permute(0, 2, 1, 3, 4).contiguous()  # back to (B, F_total, C, H, W)
        # History/reference frames are read-only conditioning (already
        # "committed", per the paper's streaming semantics); only the
        # noised-chunk slice is supervised by the flow loss.
        return out[:, n_ctx_frames:]

    def velocity(self, z_noised, sigma, text_ids, history=None, reference=None,
                 memory_state=None):
        ctx = self.encode_context(text_ids, history, reference, memory_state)
        return self.forward(z_noised, sigma, ctx)

    # -- real loader ------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_cfg: ModelConfig,
        vae_cfg: VAEConfig,
        backbone_cfg: BackboneConfig,
        latent_hw: Tuple[int, int],
    ) -> "RealWanBackbone":
        """Load the real Wan2.1 transformer + UMT5 text encoder from HF.

        Requires the ``wan`` extra (``uv sync --extra wan``; also needs
        ``sentencepiece`` for the UMT5 tokenizer) and, practically, a GPU
        with enough VRAM to hold the 1.3B transformer (~8GB) plus the UMT5
        text encoder. This path is exercised on a RunPod/Vast.ai box, not by
        CPU unit tests -- see ``tests/test_wan21_real_adapter.py`` for the
        mock-transformer coverage of everything downstream of this loader.
        """
        try:
            from diffusers import WanTransformer3DModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "RealWanBackbone.from_pretrained requires the 'wan' extra: "
                "`uv sync --extra wan` (diffusers>=0.32, transformers>=4.45, "
                "accelerate, sentencepiece).") from e
        try:
            from transformers import AutoTokenizer, UMT5EncoderModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "RealWanBackbone.from_pretrained requires `transformers` and "
                "`sentencepiece` for the UMT5 text encoder/tokenizer.") from e

        dtype = torch.bfloat16 if backbone_cfg.use_bf16 else torch.float32
        transformer = WanTransformer3DModel.from_pretrained(
            backbone_cfg.checkpoint_path, subfolder="transformer",
            torch_dtype=dtype)

        text_path = backbone_cfg.text_encoder_path or backbone_cfg.checkpoint_path
        tokenizer = AutoTokenizer.from_pretrained(text_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            text_path, subfolder="text_encoder", torch_dtype=dtype)
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad_(False)

        wan_text_dim = getattr(transformer.config, "text_dim", 4096)

        def text_encoder_fn(prompts: List[str]) -> torch.Tensor:
            device = next(text_encoder.parameters()).device
            enc = tokenizer(
                prompts, padding="max_length", truncation=True,
                max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = text_encoder(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"))
            return out.last_hidden_state.float()

        backbone = cls(
            model_cfg=model_cfg, vae_cfg=vae_cfg, backbone_cfg=backbone_cfg,
            latent_hw=latent_hw, transformer=transformer,
            text_encoder_fn=text_encoder_fn, wan_text_dim=wan_text_dim,
            num_train_timesteps=backbone_cfg.wan_num_train_timesteps,
        )
        # Kept as plain attributes (not submodules) so they stay out of
        # state_dict()/named_parameters() -- see the checkpoint-IO note above.
        backbone._hf_text_encoder = text_encoder
        backbone._hf_tokenizer = tokenizer
        return backbone

    def to(self, *args, **kwargs):  # type: ignore[override]
        result = super().to(*args, **kwargs)
        te = getattr(self, "_hf_text_encoder", None)
        if te is not None:
            te.to(*args, **kwargs)
        return result
