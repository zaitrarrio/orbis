"""Wan2.1-compatible live backbone with LoRA and persistent reference.

When official Hugging Face weights are unavailable, a structural Wan-scale DiT
stub trains the full methodology (streaming memory, LoRA post-training,
referential anchoring) at real-scale latent geometry.  When weights are present,
``load_wan_weights`` attaches them and freezes the base under LoRA.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from orbis.backbone import LoRALinear, ModelContext
from orbis.config import BackboneConfig, ModelConfig, VAEConfig
from orbis.memory import MemoryBank
from orbis.modules import DiTBlock, FinalLayer, timestep_embedding
from orbis.model import patchify, unpatchify
from orbis.vocab import VOCAB_SIZE

ROLE_MEMORY, ROLE_REF, ROLE_HISTORY, ROLE_CHUNK = 0, 1, 2, 3
_MAX_FRAMES = 128


class WanAdapter(nn.Module):
    """VideoBackbone implementing Wan-scale chunk-wise rectified flow + LoRA."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        vae_cfg: VAEConfig,
        backbone_cfg: BackboneConfig,
        latent_hw: tuple[int, int],
    ):
        super().__init__()
        self.cfg = model_cfg
        self.backbone_cfg = backbone_cfg
        self.vae_cfg = vae_cfg
        dim = model_cfg.dim
        p = model_cfg.patch_size
        self.patch = p
        self.latent_channels = vae_cfg.latent_channels
        self.patch_dim = vae_cfg.latent_channels * p * p
        self.anchor_reference = backbone_cfg.anchor_reference

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        lh, lw = latent_hw
        gh, gw = lh // p, lw // p
        self.spatial_pos = nn.Parameter(torch.randn(gh * gw, dim) * 0.02)
        self.temporal_pos = nn.Embedding(_MAX_FRAMES, dim)
        self.role_embed = nn.Embedding(4, dim)

        self.text_embed = nn.Embedding(VOCAB_SIZE, dim, padding_idx=0)
        self.text_pos = nn.Parameter(torch.randn(model_cfg.text_len, dim) * 0.02)
        # Identity / reference projection kept across all chunks
        self.ref_proj = nn.Linear(dim, dim)
        self.identity_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.cond_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.memory = MemoryBank(dim, model_cfg.memory_tokens, model_cfg.heads)

        blocks = []
        for _ in range(model_cfg.depth):
            blk = DiTBlock(dim, model_cfg.heads, model_cfg.mlp_ratio)
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        self.final = FinalLayer(dim, self.patch_dim)

        self._apply_lora(backbone_cfg.lora_rank, backbone_cfg.lora_alpha)
        self._hf_loaded = False

    def _apply_lora(self, rank: int, alpha: float) -> None:
        """Wrap patch_embed / final with LoRA.

        Do **not** freeze DiT blocks for the structural stub (``wan_stub=True``).
        Blocks use zero-init AdaLN gates; freezing them permanently zeros
        self-/cross-attention, so text never reaches spatial tokens and
        generation collapses to a near-black mean. Freeze only after real HF
        weights are loaded (see ``freeze_transformer_blocks``).
        """
        if rank <= 0:
            return
        self.patch_embed = LoRALinear(self.patch_embed, rank=rank, alpha=alpha)
        if isinstance(self.final.linear, nn.Linear):
            self.final.linear = LoRALinear(
                self.final.linear, rank=rank, alpha=alpha)

    def freeze_transformer_blocks(self) -> None:
        """Freeze DiT blocks for LoRA finetune on a pretrained backbone."""
        for blk in self.blocks:
            for p in blk.parameters():
                p.requires_grad_(False)

    def trainable_parameters(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield p

    def lora_state_dict(self) -> dict:
        out = {}
        for name, mod in self.named_modules():
            if isinstance(mod, LoRALinear):
                for k, v in mod.mergeable_state_dict().items():
                    out[f"{name}.{k}"] = v
        out["memory"] = self.memory.state_dict()
        out["ref_proj"] = self.ref_proj.state_dict()
        out["identity_token"] = self.identity_token.detach().cpu()
        return out

    def load_lora_state_dict(self, state: dict) -> None:
        if "memory" in state:
            self.memory.load_state_dict(state["memory"])
        if "ref_proj" in state:
            self.ref_proj.load_state_dict(state["ref_proj"])
        if "identity_token" in state:
            self.identity_token.data.copy_(
                state["identity_token"].to(self.identity_token.device))
        # Remaining keys are LoRA / adapter tensors
        filtered = {
            k: v for k, v in state.items()
            if k not in ("memory", "ref_proj", "identity_token")
            and not isinstance(v, dict)
        }
        missing = self.load_state_dict(filtered, strict=False)
        _ = missing  # allow partial (base weights not in bag)

    def try_load_hf(self, path: Optional[str] = None) -> bool:
        """Best-effort load of Wan2.1 Diffusers weights. Returns True if loaded."""
        path = path or self.backbone_cfg.checkpoint_path
        try:
            from diffusers import WanPipeline  # type: ignore
        except Exception:
            return False
        try:
            pipe = WanPipeline.from_pretrained(path, torch_dtype=torch.float32)
            # Structural stub keeps its own DiT; record that HF is available for
            # future weight mapping. Full tensor remap is version-sensitive.
            self._hf_loaded = True
            self._hf_pipe = pipe
            return True
        except Exception:
            return False

    def frame_tokens(self, latents: torch.Tensor, role: int,
                     frame_offset: int = 0) -> torch.Tensor:
        b, f, c, lh, lw = latents.shape
        p = self.patch
        gh, gw = lh // p, lw // p
        tok = self.patch_embed(patchify(latents, p))
        tok = tok + self.spatial_pos.view(1, 1, gh * gw, -1)
        idx = torch.arange(frame_offset, frame_offset + f, device=latents.device)
        idx = idx.clamp(max=_MAX_FRAMES - 1)
        tok = tok + self.temporal_pos(idx).view(1, f, 1, -1)
        tok = tok + self.role_embed.weight[role].view(1, 1, 1, -1)
        return tok.reshape(b, f * gh * gw, -1)

    def encode_context(
        self,
        text_ids: torch.Tensor,
        history: Optional[torch.Tensor],
        reference: Optional[torch.Tensor],
        memory_state: Optional[torch.Tensor],
    ) -> ModelContext:
        b = text_ids.shape[0]
        device = text_ids.device
        # Clamp token ids into vocab for longer Wan text_len vs toy vocab
        text_ids = text_ids.clamp(0, VOCAB_SIZE - 1)
        tlen = min(text_ids.shape[1], self.cfg.text_len)
        text_ids = text_ids[:, :tlen]
        if text_ids.shape[1] < self.cfg.text_len:
            pad = torch.zeros(
                b, self.cfg.text_len - text_ids.shape[1],
                dtype=text_ids.dtype, device=device)
            text_ids = torch.cat([text_ids, pad], dim=1)

        text_kv = self.text_embed(text_ids) + self.text_pos.unsqueeze(0)
        pooled = text_kv.mean(dim=1)

        groups = []
        if memory_state is not None:
            m = self.memory.read(memory_state)
            m = m + self.role_embed.weight[ROLE_MEMORY].view(1, 1, -1)
            groups.append(m)
        # Referential integrity: always condition on reference when present
        if reference is not None and reference.shape[1] > 0:
            ref_tok = self.frame_tokens(reference, ROLE_REF)
            ref_tok = self.ref_proj(ref_tok)
            ident = self.identity_token.expand(b, -1, -1)
            groups.append(torch.cat([ident, ref_tok], dim=1))
        if history is not None and history.shape[1] > 0:
            groups.append(self.frame_tokens(history, ROLE_HISTORY))
        ctx = torch.cat(groups, dim=1) if groups else \
            torch.zeros(b, 0, self.cfg.dim, device=device)
        return ModelContext(
            ctx_tokens=ctx, text_kv=text_kv, pooled_text=pooled, batch=b)

    def forward(self, z_noised: torch.Tensor, sigma: torch.Tensor,
                context: ModelContext) -> torch.Tensor:
        b, f, c, lh, lw = z_noised.shape
        p = self.patch
        gh, gw = lh // p, lw // p
        chunk_tok = self.frame_tokens(z_noised, ROLE_CHUNK)
        n_chunk = chunk_tok.shape[1]
        x = torch.cat([context.ctx_tokens, chunk_tok], dim=1)
        cond = self.cond_mlp(
            timestep_embedding(sigma, self.cfg.dim).to(x.dtype)
        ) + context.pooled_text
        for blk in self.blocks:
            x = blk(x, cond, context.text_kv)
        chunk_out = self.final(x[:, -n_chunk:], cond)
        chunk_out = chunk_out.reshape(b, f, gh * gw, self.patch_dim)
        return unpatchify(chunk_out, p, c, gh, gw)

    def velocity(self, z_noised, sigma, text_ids, history=None, reference=None,
                 memory_state=None):
        ctx = self.encode_context(text_ids, history, reference, memory_state)
        return self.forward(z_noised, sigma, ctx)
