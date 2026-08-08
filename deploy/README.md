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

**Known, explicitly scoped fidelity gap:** this phase keeps orbis's own
already-trained `ConvVAE` (`orbis/vae.py`) for encode/decode, not Wan's
native `AutoencoderKLWan`. Channel counts are matched (`wan21_real_config()`
sets 16 latent channels to mirror Wan's transformer `in_channels=16`), so
shapes are compatible, but the *latent distribution* orbis's VAE produces
wasn't what Wan was pretrained against — LoRA fine-tuning is expected to
adapt the frozen transformer to this shift. Swapping in the real
`AutoencoderKLWan` for the full pipeline is a larger, riskier change and is
intentionally deferred to a follow-up PR (see PR description) to keep this
change reviewable. Flow-GRPO/DanceGRPO-based RL post-training
(Phase 1b) is also a separate, later PR.
