#!/usr/bin/env bash
# Launch a Vast.ai GPU instance for Orbis smoke train/test.
#
# Pattern mirrors strobe/scripts/deploy/vast-create.sh:
#   search offers → rent → onstart smoke train/test → SSH details.
#
# Default image is a slim public PyTorch CUDA runtime (fast pull). The custom
# GHCR orbis:cuda128 image is CUDA-devel based and often stalls on marketplace
# hosts — set ORBIS_VAST_USE_GHCR=1 to force it.
#
# Requires: VAST_API_KEY in .env
# Optional: VAST_GPU_NAME (default "RTX 4090"), VAST_NUM_GPUS, VAST_MAX_DPH,
#           VAST_DISK_GB, VAST_MIN_INET, VAST_LABEL, VAST_IMAGE_LOGIN, HF_TOKEN,
#           ORBIS_VAST_USE_GHCR, VAST_IMAGE (full image:tag override)
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_cmd curl
require_cmd jq
require_env VAST_API_KEY

api="https://console.vast.ai/api/v0"
auth=(-H "Authorization: Bearer ${VAST_API_KEY}" -H "Content-Type: application/json")

# REST API uses spaces in GPU names ("RTX 4090"), not underscores.
gpu_name="${VAST_GPU_NAME:-RTX 4090}"
num_gpus="${VAST_NUM_GPUS:-1}"
max_dph="${VAST_MAX_DPH:-1.5}"
disk_gb="${VAST_DISK_GB:-50}"
# Higher default: slow downlink is what stuck prior GHCR pulls.
min_inet="${VAST_MIN_INET:-500}"
label="${VAST_LABEL:-orbis-gpu}"

if [[ -n "${VAST_IMAGE:-}" ]]; then
  image="${VAST_IMAGE}"
elif [[ "${ORBIS_VAST_USE_GHCR:-0}" == "1" ]]; then
  image="${GHCR_IMAGE}:${GHCR_TAG}"
else
  # ~slim runtime; onstart clones orbis + uv sync (avoids multi-GB devel pulls)
  image="${ORBIS_VAST_SLIM_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}"
fi

query="$(jq -n \
  --arg gpu "$gpu_name" \
  --argjson n "$num_gpus" \
  --argjson dph "$max_dph" \
  --argjson disk "$disk_gb" \
  --argjson inet "$min_inet" \
  '{
    limit: 20,
    type: "ondemand",
    rentable: {eq: true},
    verified: {eq: true},
    gpu_name: {eq: $gpu},
    num_gpus: {eq: $n},
    dph_total: {lte: $dph},
    disk_space: {gte: $disk},
    cuda_max_good: {gte: 12.4},
    reliability: {gte: 0.98},
    direct_port_count: {gte: 1},
    inet_down: {gte: $inet},
    order: [["dph_total", "asc"]]
  }')"

echo "searching offers: ${num_gpus}x ${gpu_name} <= \$${max_dph}/hr, >=${disk_gb}GB disk, inet>=${min_inet}..."
offers="$(curl -sS --max-time 60 -X POST "${api}/bundles" "${auth[@]}" -d "$query")"

if ! echo "$offers" | jq -e '.offers' >/dev/null 2>&1; then
  echo "error: offer search failed" >&2
  echo "$offers" | jq -r '.msg // .' >&2
  exit 1
fi

offer_count="$(echo "$offers" | jq '.offers | length')"
if [[ "$offer_count" -eq 0 ]]; then
  echo "error: no offers matched" >&2
  echo "hint: raise VAST_MAX_DPH or try VAST_GPU_NAME='RTX 5090' / 'H100 SXM'" >&2
  exit 1
fi

echo "top matches:"
echo "$offers" | jq -r '.offers[:5][] | "  id=\(.id)  \(.num_gpus)x\(.gpu_name)  \(.gpu_ram//0)MB  $\(.dph_total|.*1000|round/1000)/hr  \(.geolocation // "?")  inet=\(.inet_down // 0|floor)  rel=\(.reliability|.*1000|round/1000)"'

offer_id="$(echo "$offers" | jq -r '.offers[0].id')"
offer_dph="$(echo "$offers" | jq -r '.offers[0].dph_total')"
echo "renting offer ${offer_id} at \$${offer_dph}/hr"
echo "image: ${image}"

# Under ssh_* runtypes Vast replaces PID 1 with sshd — image CMD never runs.
# onstart bootstraps Orbis from a GitHub tarball (git clone often stalls on
# marketplace hosts), reuses the image CUDA torch, then smoke-tests.
onstart="$(cat <<'ONSTART'
set -eu
mkdir -p /var/log /workspace
exec >/var/log/orbis-smoke.log 2>&1
echo "[orbis] onstart $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi || true

if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL --retry 5 --retry-delay 2 https://astral.sh/uv/0.8.4/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:${PATH:-}"

echo "[orbis] fetch source tarball (no git clone)"
rm -rf /tmp/orbis-src /tmp/orbis.tgz
curl -fL --retry 8 --retry-delay 3 --retry-all-errors \
  --connect-timeout 30 --max-time 180 \
  -o /tmp/orbis.tgz \
  https://github.com/zaitrarrio/orbis/archive/refs/heads/main.tar.gz
mkdir -p /tmp/orbis-src
tar -xzf /tmp/orbis.tgz -C /tmp/orbis-src --strip-components=1
cd /tmp/orbis-src

echo "[orbis] venv with system-site-packages (reuse image torch)"
python -m venv --system-site-packages .venv
# shellcheck disable=SC1091
. .venv/bin/activate
# Install project + test deps only — do NOT re-download CUDA torch via uv lock
uv pip install -e . 'numpy>=1.23,<2' pytest

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
echo "[orbis] pytest"
python -m pytest -q
echo "[orbis] smoke train"
python scripts/train_all.py /workspace/orbis-smoke.pt 0.1
echo "[orbis] generate"
orbis generate --ckpt /workspace/orbis-smoke.pt \
  --prompt "a red circle moving right" --chunks 4 --out /workspace/out.gif
echo "[orbis] live switch"
orbis live --ckpt /workspace/orbis-smoke.pt \
  --prompt "a red circle moving right" \
  --switch "2:a blue square moving up" --chunks 4 --out /workspace/live.gif
ls -la /workspace/out.gif /workspace/live.gif /workspace/orbis-smoke.pt
touch /workspace/.orbis_smoke_ok
echo "[orbis] smoke OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec sleep infinity
ONSTART
)"

env_obj='{}'
add_env() {
  [[ -n "${2:-}" ]] || return 0
  env_obj="$(echo "$env_obj" | jq --arg k "$1" --arg v "$2" '. + {($k): $v}')"
}
add_env NVIDIA_VISIBLE_DEVICES all
add_env NVIDIA_DRIVER_CAPABILITIES compute,utility
add_env PYTHONUNBUFFERED 1
add_env HF_TOKEN "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
add_env HF_HOME /workspace/hf-cache
add_env HUGGINGFACE_HUB_CACHE /workspace/hf-cache

payload="$(jq -n \
  --arg image "$image" \
  --arg label "$label" \
  --arg onstart "$onstart" \
  --argjson env "$env_obj" \
  --argjson disk "$disk_gb" \
  '{
    client_id: "me",
    image: $image,
    label: $label,
    disk: $disk,
    runtype: "ssh_direct",
    target_state: "running",
    onstart: $onstart,
    env: $env
  }')"

if [[ -n "${VAST_IMAGE_LOGIN:-}" ]]; then
  payload="$(echo "$payload" | jq --arg l "$VAST_IMAGE_LOGIN" '. + {image_login: $l}')"
fi

resp="$(curl -sS --max-time 120 -X PUT "${api}/asks/${offer_id}/" "${auth[@]}" -d "$payload")"

if [[ "$(echo "$resp" | jq -r '.success // false')" != "true" ]]; then
  echo "error: instance creation failed" >&2
  echo "$resp" | jq -r '.msg // .' >&2
  exit 1
fi

instance_id="$(echo "$resp" | jq -r '.new_contract')"
echo "created Vast instance id=${instance_id} label=${label}"

if [[ -f .env ]]; then
  tmp="$(mktemp)"
  grep -vE '^(VAST_INSTANCE_ID|VAST_INSTANCE_IP|VAST_SSH_PORT)=' .env >"$tmp" || true
  echo "VAST_INSTANCE_ID=${instance_id}" >>"$tmp"
  mv "$tmp" .env
fi

echo "waiting for instance to start (slim image pull should be minutes, not tens)..."
for _ in $(seq 1 90); do
  detail="$(curl -sS --max-time 30 "${api}/instances/${instance_id}/" "${auth[@]}" \
    | tr -d '\000-\010\013\014\016-\037' || true)"
  # Prefer python-safe parse if jq chokes on residual controls
  st="$(echo "$detail" | jq -r '.instances.actual_status // empty' 2>/dev/null || true)"
  ip="$(echo "$detail" | jq -r '.instances.public_ipaddr // empty' 2>/dev/null || true)"
  ssh_port="$(echo "$detail" | jq -r '.instances.ports."22/tcp"[0].HostPort // empty' 2>/dev/null || true)"
  if [[ "$st" == "running" && -n "$ip" && -n "$ssh_port" ]]; then
    ip="${ip%/}"
    if [[ -f .env ]]; then
      tmp="$(mktemp)"
      grep -vE '^(VAST_INSTANCE_ID|VAST_INSTANCE_IP|VAST_SSH_PORT)=' .env >"$tmp" || true
      {
        cat "$tmp"
        echo "VAST_INSTANCE_ID=${instance_id}"
        echo "VAST_INSTANCE_IP=${ip}"
        echo "VAST_SSH_PORT=${ssh_port}"
      } >.env
      rm -f "$tmp"
    fi
    ssh_key="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
    echo "instance running: ${ip}  ssh_port=${ssh_port}"
    echo "  ssh:     ssh -i ${ssh_key} -p ${ssh_port} root@${ip}"
    echo "  logs:    ssh ... 'tail -f /var/log/orbis-smoke.log'"
    echo "  destroy: bash scripts/deploy/vast-destroy.sh ${instance_id}"
    echo "  test:    bash scripts/deploy/vast-test.sh"
    exit 0
  fi
  sleep 10
done

echo "warning: instance ${instance_id} did not report running in time" >&2
echo "check https://cloud.vast.ai/instances/" >&2
exit 1
