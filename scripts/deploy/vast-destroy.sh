#!/usr/bin/env bash
# Destroy a Vast.ai instance. Stopping is not enough — destroy to stop billing.
# Usage: bash scripts/deploy/vast-destroy.sh <instance_id>
#        bash scripts/deploy/vast-destroy.sh --all-labelled
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_cmd curl
require_cmd jq
require_env VAST_API_KEY

api="https://console.vast.ai/api/v0"
api_list="https://console.vast.ai/api/v1"
auth=(-H "Authorization: Bearer ${VAST_API_KEY}" -H "Content-Type: application/json")
label="${VAST_LABEL:-orbis-gpu}"

destroy() {
  local id="$1"
  local resp
  resp="$(curl -sS --max-time 60 -X DELETE "${api}/instances/${id}/" "${auth[@]}")"
  if [[ "$(echo "$resp" | jq -r '.success // false')" == "true" ]]; then
    echo "destroyed instance ${id}"
  else
    echo "error: failed to destroy ${id}: $(echo "$resp" | jq -r '.msg // .')" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--all-labelled" ]]; then
  ids="$(curl -sS --max-time 60 "${api_list}/instances/" "${auth[@]}" \
    | jq -r --arg l "$label" '.instances[] | select(.label == $l) | .id')"
  if [[ -z "$ids" ]]; then
    echo "no instances labelled '${label}'"
    exit 0
  fi
  for id in $ids; do destroy "$id"; done
  exit 0
fi

if [[ -z "${1:-}" ]]; then
  echo "usage: $0 <instance_id> | --all-labelled" >&2
  echo >&2
  echo "current instances:" >&2
  curl -sS --max-time 60 "${api_list}/instances/" "${auth[@]}" \
    | jq -r '.instances[]? | "  id=\(.id)  \(.label // "-")  \(.actual_status // "?")  $\(.dph_total // 0)/hr"' >&2
  exit 1
fi

destroy "$1"
