"""Real, pretrained reward models for GRPO post-training.

The pre-Phase-1b :class:`orbis.posttrain.rewards.RewardBundle` scores every
candidate against the *ground-truth target latent* (negative MSE / cosine
similarity to a freshly-initialized, untrained linear projection). That is a
disguised supervised-reconstruction loss, not a reward model in the RLHF
sense Flow-GRPO (arXiv:2505.05470) and DanceGRPO (arXiv:2505.07818) use --
it requires a ground-truth target, which defeats the purpose of
policy-gradient RL for open-ended generation, and none of its heads are
pretrained on external human-preference or vision-language data.

This module adds reward models that score a decoded chunk against only the
*prompt* (no access to a ground-truth clean latent), matching how both
papers score their rollouts:

* :class:`ClipAlignmentReward` -- CLIP (Radford et al. 2021,
  arXiv:2103.00020) image-text cosine similarity ("CLIP Score"), the
  text-alignment reward both papers use. Lazily loads a pretrained
  ``transformers`` CLIP checkpoint on first call; requires network access to
  fetch weights the first time (cached afterwards).
* :class:`AestheticReward` -- a linear probe on CLIP image embeddings,
  matching the architecture of the public LAION aesthetic-predictor
  checkpoints (the generic quality reward DanceGRPO, DDPO, and DPOK use).
  Ships with **no bundled checkpoint** -- callers must supply a real
  ``state_dict`` path (see the class docstring for where to obtain one).
  This is a deliberate scope limitation, not an oversight: shipping a
  randomly-initialized head as if it were "the aesthetic score" would repeat
  exactly the flaw being fixed in this phase (see module docstring above).
* :class:`MockReward` -- a deterministic, network-free reward for CPU unit
  tests only. Never claims to measure anything real.
* :class:`CompositeReward` -- weighted sum of pixel-space reward components
  plus orbis's own *legitimate* auxiliary continuity terms (reference
  identity, temporal smoothness), which are reused as-is since they compare
  a rollout to its own reference/history rather than to a withheld
  ground-truth target, and remain useful continuity signals for this
  streaming-video task.

All reward models are used purely as scalar (black-box) rewards -- GRPO's
policy gradient flows through the *sampling* log-probabilities (see
``orbis/posttrain/flow_sde.py``), not through the reward function, so every
``forward`` here runs under ``torch.no_grad()`` and never needs to be
differentiable.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP's published per-channel normalization constants (Radford et al. 2021).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _to_three_channel(px: torch.Tensor) -> torch.Tensor:
    if px.shape[1] == 1:
        return px.repeat(1, 3, 1, 1)
    if px.shape[1] != 3:
        return px[:, :3]
    return px


class RewardModel(nn.Module):
    """Common interface: score ``(frames, prompts, ...) -> per-sample reward (B,)``."""

    def forward(self, frames: torch.Tensor, prompts: Optional[Sequence[str]] = None,
                **kwargs) -> torch.Tensor:
        raise NotImplementedError


class MockReward(RewardModel):
    """Deterministic, network-free reward for CPU unit tests.

    Rewards frames whose mean is close to a fixed synthetic target and whose
    successive frames change smoothly. Enough structure to exercise
    group-relative-advantage and gradient-flow tests without any pretrained
    weights or network access. **Never use for real training** -- the
    reward values are meaningless outside pipeline-shape testing.
    """

    def __init__(self, target_mean: float = 0.0):
        super().__init__()
        self.target_mean = target_mean

    def forward(self, frames: torch.Tensor, prompts=None, **kwargs) -> torch.Tensor:
        b = frames.shape[0]
        flat = frames.flatten(1)
        level = -(flat.mean(dim=1) - self.target_mean).pow(2)
        if frames.dim() >= 2 and frames.shape[1] > 1:
            smooth = -(frames[:, 1:] - frames[:, :-1]).pow(2).flatten(1).mean(dim=1)
        else:
            smooth = torch.zeros(b, device=frames.device)
        return level + 0.1 * smooth


class ClipAlignmentReward(RewardModel):
    """CLIP image-text cosine similarity ("CLIP Score"), averaged over frames.

    Lazily loads ``openai/clip-vit-base-patch32`` via ``transformers`` on
    first call (requires network access the first time; cached afterwards).
    Preprocessing (resize + CLIP's published mean/std normalization) is done
    manually with plain tensor ops rather than ``transformers``' PIL-based
    image processor, to avoid CHW-tensor/PIL round-tripping and keep this
    robust to run inside a training loop.

    ``frames``: ``(B, F, C, H, W)`` in ``[-1, 1]`` pixel space (matches
    ``orbis.vae.ConvVAE.decode`` output convention).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 image_size: int = 224):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self._model = None
        self._tokenizer = None

    def _lazy_load(self, device):
        if self._model is None:
            from transformers import CLIPModel, CLIPTokenizer
            self._model = CLIPModel.from_pretrained(self.model_name).to(device).eval()
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model, self._tokenizer

    @torch.no_grad()
    def forward(self, frames: torch.Tensor, prompts: Optional[Sequence[str]] = None,
                **kwargs) -> torch.Tensor:
        if prompts is None:
            raise ValueError(
                "ClipAlignmentReward requires raw text prompts (pass "
                "prompts=... -- e.g. batch['prompts'] from RolloutSampler); "
                "orbis's own token_ids use a different, incompatible vocabulary.")
        device = frames.device
        model, tok = self._lazy_load(device)
        b, f = frames.shape[:2]
        px = frames.reshape(b * f, *frames.shape[2:])
        px = _to_three_channel(px)
        px = (px.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        px = F.interpolate(px, size=(self.image_size, self.image_size),
                           mode="bilinear", align_corners=False)
        mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)
        px = (px - mean) / std

        img_feat = model.get_image_features(pixel_values=px)
        img_feat = F.normalize(img_feat, dim=-1).view(b, f, -1)

        text_inputs = tok(list(prompts), padding=True, truncation=True,
                          return_tensors="pt").to(device)
        txt_feat = model.get_text_features(**text_inputs)
        txt_feat = F.normalize(txt_feat, dim=-1)  # (B, D)

        sim = (img_feat * txt_feat.unsqueeze(1)).sum(dim=-1)  # (B, F)
        return sim.mean(dim=1)


class AestheticReward(RewardModel):
    """Linear probe on CLIP ViT-L/14 image embeddings (LAION aesthetic-predictor
    architecture: https://github.com/LAION-AI/aesthetic-predictor).

    This class does **not** ship a trained checkpoint. Pass a real
    ``checkpoint_path`` (a ``state_dict`` for a single ``nn.Linear(768, 1)``
    head, e.g. the public ``sac+logos+ava1-l14-linearMSE.pth`` weights) to
    get real aesthetic scores. Without one, construction raises unless
    ``allow_untrained_head=True`` is passed explicitly -- which loudly warns
    on every call that the scores are meaningless, for pipeline-shape
    testing only (never for real training decisions).
    """

    def __init__(self, checkpoint_path: Optional[str] = None,
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 image_size: int = 224, allow_untrained_head: bool = False):
        super().__init__()
        if checkpoint_path is None and not allow_untrained_head:
            raise FileNotFoundError(
                "AestheticReward requires a real checkpoint (a state_dict for "
                "nn.Linear(768, 1)), e.g. the public LAION aesthetic-predictor "
                "weights (https://github.com/LAION-AI/aesthetic-predictor). "
                "Pass checkpoint_path=... , or explicitly opt into an "
                "untrained head via allow_untrained_head=True for "
                "pipeline-shape testing only (scores will be meaningless).")
        self.checkpoint_path = checkpoint_path
        self.clip_model_name = clip_model_name
        self.image_size = image_size
        self._untrained = checkpoint_path is None
        self.head = nn.Linear(768, 1)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.head.load_state_dict(state)
        for p in self.head.parameters():
            p.requires_grad_(False)
        self._model = None

    def _lazy_load(self, device):
        if self._model is None:
            from transformers import CLIPModel
            self._model = CLIPModel.from_pretrained(self.clip_model_name).to(device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            self.head.to(device)
        return self._model

    @torch.no_grad()
    def forward(self, frames: torch.Tensor, prompts=None, **kwargs) -> torch.Tensor:
        if self._untrained:
            warnings.warn(
                "AestheticReward is running with an UNTRAINED head -- scores "
                "are meaningless and must not be used for real training.",
                RuntimeWarning)
        device = frames.device
        model = self._lazy_load(device)
        b, f = frames.shape[:2]
        px = frames.reshape(b * f, *frames.shape[2:])
        px = _to_three_channel(px)
        px = (px.clamp(-1, 1) + 1) / 2
        px = F.interpolate(px, size=(self.image_size, self.image_size),
                           mode="bilinear", align_corners=False)
        mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)
        px = (px - mean) / std
        img_feat = model.get_image_features(pixel_values=px)
        score = self.head(img_feat).squeeze(-1).view(b, f)
        return score.mean(dim=1)


class CompositeReward(RewardModel):
    """Weighted sum of pixel-space reward components plus latent-space
    continuity auxiliaries (reference-identity, temporal smoothness) reused
    from :mod:`orbis.posttrain.rewards` -- those terms need no ground-truth
    target, only the rollout's own reference/history, so they remain valid
    signals here.

    ``forward`` returns ``(total_reward[B], detail_dict)`` (matching the
    previous ``RewardBundle`` interface for compatibility with existing
    logging code).
    """

    def __init__(self, components: Sequence[Tuple[RewardModel, float]],
                 w_reference: float = 0.0, w_motion: float = 0.0):
        super().__init__()
        self.components = nn.ModuleList([c for c, _ in components])
        self.weights = [w for _, w in components]
        self.w_reference = w_reference
        self.w_motion = w_motion

    def forward(self, frames: torch.Tensor, prompts: Optional[Sequence[str]] = None,
                latent_chunk: Optional[torch.Tensor] = None,
                reference: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b = frames.shape[0]
        device = frames.device
        total = torch.zeros(b, device=device)
        detail: Dict[str, torch.Tensor] = {}
        for comp, w in zip(self.components, self.weights):
            r = comp(frames, prompts=prompts, **kwargs)
            total = total + w * r
            detail[type(comp).__name__] = r.detach().mean()

        if self.w_reference and latent_chunk is not None and reference is not None \
                and reference.numel() > 0:
            from orbis.posttrain.rewards import reference_identity_reward
            r = reference_identity_reward(latent_chunk, reference)
            total = total + self.w_reference * r
            detail["reference"] = r.detach().mean()

        if self.w_motion and latent_chunk is not None and latent_chunk.shape[1] > 1:
            # Self-consistency temporal-smoothness proxy (no ground truth
            # needed): penalize large frame-to-frame latent jumps.
            d = latent_chunk[:, 1:] - latent_chunk[:, :-1]
            r = -d.pow(2).flatten(1).mean(dim=1)
            total = total + self.w_motion * r
            detail["motion_smooth"] = r.detach().mean()

        return total, detail
