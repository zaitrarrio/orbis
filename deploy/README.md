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
