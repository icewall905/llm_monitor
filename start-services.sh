#!/usr/bin/env bash

TTYD_INTERNAL_PORT="${TTYD_INTERNAL_PORT:-7681}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

cleanup() {
  kill "$TTYD_PID" "$DASH_PID" "$NGINX_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

/usr/local/bin/ttyd -p "${TTYD_INTERNAL_PORT}" -W /usr/local/bin/launch-monitor.sh &
TTYD_PID=$!

python3 /usr/local/bin/dashboard-server.py &
DASH_PID=$!

nginx -g 'daemon off;' &
NGINX_PID=$!

wait "$TTYD_PID" "$DASH_PID" "$NGINX_PID"
