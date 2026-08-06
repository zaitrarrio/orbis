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
