#!/usr/bin/env bash
# Runs vLLM in a tmux window. Writes the PID to the file the gateway uses to
# stop the server when the worker needs to be torn down.
set -euo pipefail

BYOM_DIR=/content/byom

# FlashInfer invokes `ninja` by name; the binary is installed in this venv.
export PATH="$BYOM_DIR/venv/bin:$PATH"

# `exec` makes the shell become the vLLM process, so the PID written here is
# the server's own.
echo $$ > "$BYOM_DIR/vllm.pid"
exec bash "$BYOM_DIR/vllm.cmd"
