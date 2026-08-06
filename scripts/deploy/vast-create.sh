#!/usr/bin/env bash
# Launch a Vast.ai GPU instance running ghcr.io/zaitrarrio/orbis:cuda128.
#
# Pattern mirrors strobe/scripts/deploy/vast-create.sh:
#   search offers → rent → onstart smoke train/test → SSH details.
#
# Requires: VAST_API_KEY in .env
# Optional: VAST_GPU_NAME (default "RTX 4090"), VAST_NUM_GPUS, VAST_MAX_DPH,
#           VAST_DISK_GB, VAST_MIN_INET, VAST_LABEL, VAST_IMAGE_LOGIN, HF_TOKEN
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
min_inet="${VAST_MIN_INET:-200}"
label="${VAST_LABEL:-orbis-gpu}"

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

echo "searching offers: ${num_gpus}x ${gpu_name} <= \$${max_dph}/hr, >=${disk_gb}GB disk..."
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
echo "$offers" | jq -r '.offers[:5][] | "  id=\(.id)  \(.num_gpus)x\(.gpu_name)  \(.gpu_ram//0)MB  $\(.dph_total|.*1000|round/1000)/hr  \(.geolocation // "?")  rel=\(.reliability|.*1000|round/1000)"'

offer_id="$(echo "$offers" | jq -r '.offers[0].id')"
offer_dph="$(echo "$offers" | jq -r '.offers[0].dph_total')"
echo "renting offer ${offer_id} at \$${offer_dph}/hr"

# Under ssh_* runtypes Vast replaces PID 1 with sshd — image CMD never runs.
# onstart is the only bootstrap hook. Smoke train + generate, then stay up for SSH.
onstart="$(cat <<'ONSTART'
set -eu
mkdir -p /var/log /workspace
exec >/var/log/orbis-smoke.log 2>&1
echo "[orbis] onstart $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi || true
cd /workspace
if [ ! -f scripts/train_all.py ]; then
  echo "[orbis] workspace missing project — cloning"
  rm -rf /tmp/orbis-src
  git clone --depth 1 https://github.com/zaitrarrio/orbis.git /tmp/orbis-src
  cd /tmp/orbis-src
  uv sync --frozen --group dev || uv sync --group dev
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
echo "[orbis] pytest"
uv run pytest -q
echo "[orbis] smoke train"
uv run python scripts/train_all.py /workspace/orbis-smoke.pt 0.1
echo "[orbis] generate"
uv run orbis generate --ckpt /workspace/orbis-smoke.pt \
  --prompt "a red circle moving right" --chunks 4 --out /workspace/out.gif
echo "[orbis] live switch"
uv run orbis live --ckpt /workspace/orbis-smoke.pt \
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
  --arg image "${GHCR_IMAGE}:${GHCR_TAG}" \
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

# Persist ids into .env for later SSH / destroy (never commit .env)
if [[ -f .env ]]; then
  grep -v '^VAST_INSTANCE_ID=' .env | grep -v '^VAST_INSTANCE_IP=' | grep -v '^VAST_SSH_PORT=' >.env.tmp || true
  {
    cat .env.tmp
    echo "VAST_INSTANCE_ID=${instance_id}"
  } >.env
  rm -f .env.tmp
fi

echo "waiting for instance to start (image pull may take several minutes)..."
for _ in $(seq 1 90); do
  detail="$(curl -sS --max-time 30 "${api}/instances/${instance_id}/" "${auth[@]}" \
    | tr -d '\000-\010\013\014\016-\037' || true)"
  status="$(echo "$detail" | jq -r '.instances.actual_status // empty' 2>/dev/null || true)"
  ip="$(echo "$detail" | jq -r '.instances.public_ipaddr // empty' 2>/dev/null || true)"
  ssh_port="$(echo "$detail" | jq -r '.instances.ports."22/tcp"[0].HostPort // empty' 2>/dev/null || true)"
  if [[ "$status" == "running" && -n "$ip" ]]; then
    ip="${ip%/}"
    echo "instance running: ${ip}  ssh_port=${ssh_port:-?}"
    if [[ -f .env ]]; then
      grep -v '^VAST_INSTANCE_IP=' .env | grep -v '^VAST_SSH_PORT=' >.env.tmp || true
      {
        cat .env.tmp
        echo "VAST_INSTANCE_IP=${ip}"
        echo "VAST_SSH_PORT=${ssh_port:-22}"
      } >.env
      rm -f .env.tmp
    fi
    ssh_key="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
    echo "  ssh:     ssh -i ${ssh_key} -p ${ssh_port:-22} root@${ip}"
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
