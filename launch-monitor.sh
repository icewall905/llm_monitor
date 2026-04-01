#!/usr/bin/env bash
set -euo pipefail

MONITOR_APP="${MONITOR_APP:-nvtop}"

case "$MONITOR_APP" in
  nvtop)
    exec nvtop
    ;;
  btop)
    exec btop --utf-force
    ;;
  *)
    echo "Unsupported MONITOR_APP='$MONITOR_APP'. Use 'nvtop' or 'btop'."
    exit 1
    ;;
esac
