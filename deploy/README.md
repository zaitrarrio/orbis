# Deploy: GitHub → GHCR → RunPod / Vast

Source of truth is this repository: [zaitrarrio/orbis](https://github.com/zaitrarrio/orbis).

Pushes to `main` (and version tags `v*`) build the Linux + CUDA 12.8 image via
[`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml)
and publish to **GitHub Container Registry**:

| Tag | Meaning |
|-----|---------|
| `ghcr.io/zaitrarrio/orbis:cuda128` | Latest `main` (stable deploy tag) |
| `ghcr.io/zaitrarrio/orbis:main` | Branch tip |
| `ghcr.io/zaitrarrio/orbis:sha-<short>` | Exact commit |
| `ghcr.io/zaitrarrio/orbis:vX.Y.Z` | Release tag |

Prefer **RTX 4090**, **RTX 5090**, or **H100**. Host driver **≥ 570** for CUDA 12.8 / Blackwell.

## Local (same image definition)

```bash
docker compose build
docker compose run --rm orbis nvidia-smi
```

## RunPod

1. Pod template image: `ghcr.io/zaitrarrio/orbis:cuda128`
2. If the package is private: add a GHCR pull credential / registry auth in the template.
3. GPU: 4090 / 5090 / H100 · container disk ≥ 40 GB · volume `/workspace`
4. Optional start command:

```bash
bash -lc 'cd /workspace && uv run python scripts/train_all.py orbis.pt 0.1'
```

Mirror fields from `deploy/runpod/template.json`.

## Vast.ai

1. Template image: `ghcr.io/zaitrarrio/orbis:cuda128` (same GHCR tags as above).
2. Filter offers for 4090 / 5090 / H100.
3. **Entrypoint** launch keeps the image `CMD`; **SSH/Jupyter** → paste `deploy/vast/onstart.sh` into On-start.
4. Docker options: see `deploy/vast/template.env`. Private GHCR: configure registry login on the account/template.

### Scripted deploy (Strobe-style)

```bash
# .env needs VAST_API_KEY (never commit .env)
cp .env.example .env   # then fill VAST_API_KEY

bash scripts/deploy/vast-create.sh    # rent RTX 4090; slim PyTorch image + clone onstart
bash scripts/deploy/vast-test.sh      # SSH-poll until /workspace/.orbis_smoke_ok
bash scripts/deploy/vast-destroy.sh --all-labelled   # stop billing
```

Defaults use `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` (fast pull). Onstart
fetches a GitHub **tarball** of `main` (avoids stalled `git clone`), reuses the
image CUDA torch via `--system-site-packages`, then smoke-tests.
Force GHCR with `ORBIS_VAST_USE_GHCR=1`.

Overrides: `VAST_GPU_NAME='RTX 5090'`, `VAST_MAX_DPH=2.0`, `VAST_MIN_INET=500`,
`VAST_LABEL=orbis-gpu`, `VAST_IMAGE=...`.
SSH key default: `~/.ssh/id_strobe_vast` (`VAST_SSH_KEY` to override).

## Make the GHCR package public (optional)

Repo **Settings → Packages** (or the package page after the first workflow run) →
package visibility **Public**, so RunPod/Vast can pull without a token.

## Wan2.1 live training (real-scale path)

The Live methodology on Wan uses a structural Wan-scale DiT + LoRA by default
(`backbone.wan_stub=True`). Official weights are **not** shipped in git.

### Hugging Face cache on the pod volume

Mount a persistent volume at `/workspace` and cache weights under
`/workspace/hf-cache`:

```bash
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache
# optional extras for Diffusers Wan + video IO
uv sync --extra wan
```

Default checkpoint id: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (see
`orbis.config.BackboneConfig.checkpoint_path`). Prefer **H100** for 14B; **4090 /
5090** for the 1.3B stub/LoRA path.

### Train

```bash
# Full methodology smoke (CI-sized Wan stub geometry):
uv run python scripts/train-live-wan.py orbis-wan.pt 1.0 --smoke

# Real-scale config (480x832); still stub unless --load-hf:
uv run python scripts/train-live-wan.py orbis-wan.pt 0.1

# Attempt to attach official Diffusers Wan weights:
uv run python scripts/train-live-wan.py orbis-wan.pt 0.1 --load-hf

# Toy path (unchanged):
uv run python scripts/train_all.py orbis.pt 0.1
uv run python scripts/train_all.py orbis-wan-smoke.pt 0.1 --backbone wan
```

Clips for mid-training: put `manifest.jsonl` + videos under e.g.
`/workspace/data/openvid` and pass `--data /workspace/data/openvid`.

### Real Wan2.1-1.3B backbone (frozen transformer + LoRA)

`--load-hf` above only affects the legacy structural Wan-*scale* stub
(`orbis/adapters/wan_adapter.py`); by its own docstring it never actually
remaps any pretrained tensors into the stub. `--real-wan` instead drives the
real `diffusers.WanTransformer3DModel` directly, frozen, with LoRA adapters
on its attention/FFN projections plus orbis's memory-bank and text
projection heads as the only trainable parameters
(`orbis/adapters/wan21_real.py`). This is the higher-fidelity path.

```bash
uv sync --extra wan   # diffusers, transformers, accelerate, sentencepiece (UMT5 tokenizer)
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache

# Real Wan2.1-T2V-1.3B transformer + UMT5 text encoder, frozen + LoRA:
uv run python scripts/train-live-wan.py orbis-real-wan.pt 0.1 --real-wan

# Override the HF checkpoint / text-encoder repo if needed:
uv run python scripts/train-live-wan.py orbis-real-wan.pt 0.1 --real-wan \
  --wan-checkpoint Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --wan-text-encoder Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

Requires a real GPU (~8GB+ VRAM for the 1.3B transformer plus the UMT5 text
encoder; prefer 4090/5090/H100) — this sandbox has no GPU and no `diffusers`
installed, so this path is validated here only via CPU mock-transformer unit
tests (`tests/test_wan21_real_adapter.py`); please run the commands above on
your rented pod to confirm an end-to-end forward/backward pass and inspect
sample rollouts.

For a fast, cheap fidelity check before committing to a full training run,
use the standalone smoke test instead of the full pipeline:

```bash
uv sync --extra wan
export HF_HOME=/workspace/hf-cache
uv run python scripts/validate_real_wan.py
# Also validate the real, frozen Wan VAE (see below) end to end:
uv run python scripts/validate_real_wan.py --real-vae
```

Only export `HF_HOME` — `huggingface_hub` resolves the actual cache
directory to `$HF_HOME/hub` by default. Also exporting
`HUGGINGFACE_HUB_CACHE=/workspace/hf-cache` (i.e. the *same* path as
`HF_HOME`, missing the `/hub` suffix) makes `huggingface_hub` cache into a
different, flat `$HF_HOME/models--...` layout instead of reusing the
existing `$HF_HOME/hub/models--...` cache — silently triggering a full
multi-GB re-download of the transformer, text encoder, and VAE on every
run, and can fill a small instance's disk. Confirmed on real hardware
while validating the VAE swap below; if you see repeated "Fetching N
files" downloads despite weights already being cached, check for this
env var mismatch first.

This loads the real Wan2.1-1.3B transformer + UMT5 text encoder, runs one
real forward + backward pass at the configured shapes, checks that
gradients only reach LoRA/memory parameters (never the frozen base), and
confirms the saved checkpoint is small (megabytes, not gigabytes) and
reloads correctly. It does not run VAE pretraining or any posttrain stage
-- see `scripts/train-live-wan.py --real-wan` above for the full pipeline.

**Real Wan VAE (closes the fidelity gap below):** pass `--real-vae` to
`validate_real_wan.py`, or `--real-wan-vae` (alongside `--real-wan`) to
`train-live-wan.py`, to encode/decode with Wan's own real, frozen,
pretrained `AutoencoderKLWan` (`orbis/adapters/wan21_vae.py`) instead of
orbis's toy `ConvVAE`. It wraps `AutoencoderKLWan` per-frame (each frame
as a `T=1` "video", verified on real hardware to always yield exactly one
temporal latent — no cross-frame temporal compression is engaged), and
applies the checkpoint's own real per-channel `latents_mean`/`latents_std`
normalization, so `RealWanBackbone` sees the *exact* latent space it was
pretrained against. It is fully frozen (no LoRA), and its `state_dict()`
is intentionally empty (nothing trainable to checkpoint — a fresh
`from_pretrained()` always reconstructs an identical instance). Confirmed
end to end on a real H100: `vae encode`/`vae decode` at the configured
480x832 shape, a combined `RealWanBackbone` + `RealWanVAE` forward +
backward pass with zero gradient leakage into either frozen component,
and correct lean-checkpoint save/reload.

Genuine multi-frame temporal VAE compression across chunks (using `T>1`
so Wan's causal temporal downsampling actually engages, rather than the
`T=1` per-frame special case used here) is a materially larger, riskier
rewrite of the chunk pipeline and remains an explicitly scoped-out,
further-deferred fidelity gap.

**Previously known, now-closeable fidelity gap:** by default (`real_vae`
not set), this phase still keeps orbis's own already-trained `ConvVAE`
(`orbis/vae.py`) for encode/decode, not Wan's native `AutoencoderKLWan`.
Channel counts are matched (`wan21_real_config()` sets 16 latent channels
to mirror Wan's transformer `in_channels=16`), so shapes are compatible,
but the *latent distribution* orbis's VAE produces wasn't what Wan was
pretrained against — LoRA fine-tuning is expected to adapt the frozen
transformer to this shift. Pass `--real-vae`/`--real-wan-vae` (see above)
to close this gap; it defaults to off to keep the base `--real-wan` path's
existing (CPU-testable) behavior unchanged.

### Flow-GRPO / DanceGRPO-style RL post-training (Phase 1b)

`orbis/posttrain/grpo.py` implements clipped-ratio GRPO over a *stochastic*
reformulation of the rectified-flow sampler (`orbis/posttrain/flow_sde.py`),
following [Flow-GRPO](https://arxiv.org/abs/2505.05470) and
[DanceGRPO](https://arxiv.org/abs/2505.07818) (official DanceGRPO code:
https://github.com/XueZeyue/DanceGRPO, which explicitly supports Wan2.1 via
`scripts/finetune/finetune_wan_2_1_grpo.sh`). This replaces the earlier
`grpo_align` (best-of-G self-distillation — group-relative advantages were
computed correctly, but the "policy loss" was an L2 regression toward the
winning candidate, not a real policy gradient; the previous
`RewardBundle` scored every candidate against a withheld ground-truth
latent, not a real pretrained reward model).

What's new:

* **`FlowSDE`** (`orbis/posttrain/flow_sde.py`) converts orbis's
  deterministic Euler sampler into an equivalent SDE with a tractable
  per-step Gaussian transition density (derivation and citations in the
  module docstring; `eta=0` reduces exactly to the original deterministic
  step — checked in `tests/test_flow_sde.py`). This is what makes the GRPO
  importance ratio `rho = exp(logp_theta - logp_theta_old)` well-defined.
* **Real reward models** (`orbis/posttrain/reward_models.py`):
  `ClipAlignmentReward` (real pretrained CLIP image-text alignment,
  `openai/clip-vit-base-patch32` by default) scores decoded pixel frames
  against the batch's *raw* prompt text (`RolloutSampler` now also returns
  `"prompts"` alongside `"text_ids"`, since CLIP's tokenizer is unrelated to
  orbis's own canonical-token vocabulary). `AestheticReward` mirrors the
  public LAION aesthetic-predictor architecture but ships **no bundled
  checkpoint** — pass a real one via `checkpoint_path`, or it refuses to run
  unless you explicitly opt into an untrained head for shape-testing only.
  `MockReward` is network-free and CPU-testable but meaningless — used only
  in `tests/test_grpo_real.py` and as the pipeline's `--grpo-reward mock`
  default (safe for `--smoke`/CI).
* Orbis's own reference-identity and temporal-smoothness continuity terms
  are kept as auxiliary reward components (`CompositeReward`) since they
  compare a rollout to its own reference/history, not to a withheld
  ground-truth target — they didn't need to be replaced.
* New `GRPOConfig` (`orbis/config.py`): `sde_eta`, `clip_eps`, `kl_coef`,
  `train_denoise_steps` (Flow-GRPO's "Denoising Reduction" — fewer steps at
  RL-rollout time than final inference), `reward_backend`.

```bash
# CPU-testable, network-free (default -- matches --smoke/CI):
uv run python scripts/train-live-wan.py orbis-wan.pt 0.1 --smoke

# Real reward model (downloads a pretrained CLIP checkpoint on first run):
uv run python scripts/train-live-wan.py orbis-real-wan.pt 0.1 --real-wan --grpo-reward clip

# Tune SDE exploration noise if rollouts look degenerate/unstable:
uv run python scripts/train-live-wan.py orbis-real-wan.pt 0.1 --real-wan --grpo-reward clip --grpo-eta 0.5
```

CPU unit tests (`tests/test_flow_sde.py`, `tests/test_grpo_real.py`) cover
the SDE derivation invariants (ODE-equivalence at `eta=0`, log-prob/KL
correctness, ratio=1 when policies match) and the GRPO loop itself
(gradient isolation to trainable params only, finite loss, group-relative
advantage bookkeeping) using `wan_smoke_config()` + `MockReward` — no GPU or
network access required. As with the real-Wan backbone above, please run a
short real-GPU pass (`--grpo-reward clip`) on your rented pod to confirm the
CLIP download/preprocessing path and inspect real reward values and
sample rollouts; this sandbox has no GPU and no network access to fetch
pretrained CLIP weights.
