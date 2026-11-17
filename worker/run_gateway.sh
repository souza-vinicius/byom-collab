#!/usr/bin/env bash
# Runs the gateway in a tmux window, using the venv created by bootstrap.
set -euo pipefail

BYOM_DIR=/content/byom

echo $$ > "$BYOM_DIR/gateway.pid"
exec "$BYOM_DIR/venv/bin/python" "$BYOM_DIR/gateway.py"
