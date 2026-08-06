# Deploy: RunPod + Vast.ai

Orbis ships as a **Linux + CUDA 12.8** Docker image. Prefer **RTX 4090**, **RTX 5090**, or **H100** pods. Host driver should be **≥ 570** for CUDA 12.8 / Blackwell.

## Build & push

```bash
docker build -t YOUR_REGISTRY/orbis:cuda128 .
docker push YOUR_REGISTRY/orbis:cuda128
```

Local GPU check:

```bash
docker compose run --rm orbis nvidia-smi
docker compose run --rm orbis python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## RunPod

1. Create a **Pod template** with image `YOUR_REGISTRY/orbis:cuda128`.
2. GPU: RTX 4090 / RTX 5090 / H100 (match VRAM to workload).
3. Container disk ≥ 40 GB; volume mount `/workspace` for checkpoints.
4. Start command (interactive) or override, e.g.:

```bash
bash -lc 'cd /workspace && uv run python scripts/train_all.py orbis.pt 0.1'
```

See `deploy/runpod/template.json` for field defaults you can mirror in the console/API.

## Vast.ai

1. Create a template with the same image and `CUDA` filter for 4090 / 5090 / H100.
2. Launch mode: **Entrypoint** (keeps image `CMD`) or **SSH** with `deploy/vast/onstart.sh` pasted into **On-start**.
3. Docker options example: `-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility`

On SSH/Jupyter modes Vast replaces the entrypoint — always run `onstart.sh` (or call your train/serve command from it).
