#!/usr/bin/env bash
# Shared helpers for Orbis deploy scripts (pattern from strobe/scripts/deploy).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

if [[ -f .env ]]; then
  _preset_env="$(export -p)"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  eval "$_preset_env"
  unset _preset_env
fi

: "${IMAGE:=orbis}"
: "${GHCR_REGISTRY:=ghcr.io}"
: "${GHCR_OWNER:=${GITHUB_OWNER:-zaitrarrio}}"
: "${GHCR_IMAGE:=${GHCR_REGISTRY}/${GHCR_OWNER}/${IMAGE}}"
: "${GHCR_TAG:=cuda128}"

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "error: required command not found: $cmd" >&2
    exit 1
  }
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: set $name in .env or the environment" >&2
    exit 1
  fi
}
