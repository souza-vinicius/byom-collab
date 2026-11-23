#!/usr/bin/env bash
# Prepare the Colab runtime and start vLLM and the gateway in a tmux session.
#
# Runs as a kernel cell because the published CLI does not accept
# `colab ssh -- CMD`:
#   colab exec -s <name>  (code that calls this script)
#
# Returns as soon as the processes start; model readiness is verified later,
# through the tunnel.
set -euo pipefail

BYOM_DIR=/content/byom
TARBALL=/content/byom-worker.tar.gz
VENV="$BYOM_DIR/venv"
LOGS="$BYOM_DIR/logs"
SESSION_FILE="$BYOM_DIR/session.json"
TMUX_SESSION=byom

log() { printf '[byom-bootstrap] %s\n' "$*"; }
die() { printf '[byom-bootstrap] error: %s\n' "$*" >&2; exit 1; }

# The bundle is uploaded to /content, which always exists, and extracted here
# because the content API does not create intermediate directories.
mkdir -p "$BYOM_DIR"
if [ -f "$TARBALL" ]; then
  log "Extracting $TARBALL to $BYOM_DIR"
  tar xzf "$TARBALL" -C "$BYOM_DIR"
fi

[ -f "$SESSION_FILE" ] || die "session.json not found at $SESSION_FILE"
mkdir -p "$LOGS"

# 1. uv --------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv is unavailable after installation"

# 2. venv ------------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  log "Creating virtual environment at $VENV"
  uv venv --python 3.12 "$VENV"
fi

# 3. dependencies ---------------------------------------------------------
VLLM_SPEC="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["vllm_spec"])' "$SESSION_FILE")"
RUNTIME="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("runtime","vllm"))' "$SESSION_FILE")"
log "Runtime: $RUNTIME"
if [ "$RUNTIME" = "vllm" ]; then
  log "Installing $VLLM_SPEC and gateway dependencies"
  uv pip install --python "$VENV/bin/python" "$VLLM_SPEC"
fi
uv pip install --python "$VENV/bin/python" -r "$BYOM_DIR/requirements.txt"

# 4. runtime-specific binaries and weights --------------------------------
python3 "$BYOM_DIR/prepare_runtime.py" "$SESSION_FILE"

# 5. server command -------------------------------------------------------
python3 "$BYOM_DIR/build_vllm_command.py" "$SESSION_FILE" > "$BYOM_DIR/runtime.cmd"
log "Wrote server command to $BYOM_DIR/runtime.cmd"

# 6. tmux processes -------------------------------------------------------
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  log "Stopping previous tmux session"
  tmux kill-session -t "$TMUX_SESSION"
fi

if [ "$RUNTIME" = "vllm" ]; then
  RUN_SERVER="$BYOM_DIR/run_vllm.sh"
  LOG_NAME=vllm
else
  RUN_SERVER="$BYOM_DIR/run_runtime.sh"
  LOG_NAME="$RUNTIME"
fi

tmux new-session -d -s "$TMUX_SESSION" -n "$LOG_NAME" \
  "bash $RUN_SERVER > $LOGS/$LOG_NAME.log 2>&1"
tmux new-window -t "$TMUX_SESSION" -n gateway \
  "bash $BYOM_DIR/run_gateway.sh > $LOGS/gateway.log 2>&1"

log "$RUNTIME and gateway started; logs are in $LOGS"
log "ready"
