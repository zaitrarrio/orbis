#!/usr/bin/env bash
# Poll smoke log on a running Vast instance; fail if smoke did not complete.
# Requires: VAST_INSTANCE_IP, VAST_SSH_PORT (from vast-create / .env), SSH key.
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_cmd ssh
require_env VAST_INSTANCE_IP

ip="${VAST_INSTANCE_IP%/}"
port="${VAST_SSH_PORT:-22}"
key="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

echo "waiting for SSH ${ip}:${port} ..."
for _ in $(seq 1 60); do
  if ssh "${ssh_opts[@]}" "root@${ip}" 'true' 2>/dev/null; then
    break
  fi
  sleep 5
done

echo "polling /var/log/orbis-smoke.log for smoke OK ..."
for _ in $(seq 1 120); do
  if ssh "${ssh_opts[@]}" "root@${ip}" 'test -f /workspace/.orbis_smoke_ok'; then
    echo "SMOKE PASS"
    ssh "${ssh_opts[@]}" "root@${ip}" 'tail -n 40 /var/log/orbis-smoke.log; ls -la /workspace/out.gif /workspace/live.gif /workspace/orbis-smoke.pt'
    exit 0
  fi
  # show last lines while waiting
  ssh "${ssh_opts[@]}" "root@${ip}" 'tail -n 5 /var/log/orbis-smoke.log 2>/dev/null || echo "(log not ready)"' || true
  sleep 15
done

echo "SMOKE FAIL — timed out waiting for /workspace/.orbis_smoke_ok" >&2
ssh "${ssh_opts[@]}" "root@${ip}" 'tail -n 80 /var/log/orbis-smoke.log' || true
exit 1
