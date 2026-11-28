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
VLLM_OMNI_SPEC="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print((d.get("worker") or {}).get("vllm_omni_spec",""))' "$SESSION_FILE")"
RUNTIME="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("runtime","vllm"))' "$SESSION_FILE")"
log "Runtime: $RUNTIME"
if [ "$RUNTIME" = "tts" ]; then
  [ -n "$VLLM_OMNI_SPEC" ] || die "runtime tts requires [worker].vllm_omni_spec"
  log "Installing $VLLM_SPEC and $VLLM_OMNI_SPEC"
  uv pip install --python "$VENV/bin/python" "$VLLM_SPEC" "$VLLM_OMNI_SPEC"
elif [ "$RUNTIME" = "vllm" ]; then
  log "Installing $VLLM_SPEC and gateway dependencies"
  uv pip install --python "$VENV/bin/python" "$VLLM_SPEC"
fi
# `requirements.lock` traz versoes exatas com hashes (fonte: requirements.txt);
# sem ele o install cairia nas faixas e um release novo poderia quebrar a sessao.
if [ -f "$BYOM_DIR/requirements.lock" ]; then
  log "Installing gateway dependencies from requirements.lock (pinned, hashes)"
  uv pip install --python "$VENV/bin/python" --require-hashes -r "$BYOM_DIR/requirements.lock"
else
  log "WARNING: requirements.lock is missing; falling back to requirements.txt ranges"
  uv pip install --python "$VENV/bin/python" -r "$BYOM_DIR/requirements.txt"
fi

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

if [ "$RUNTIME" = "vllm" ] || [ "$RUNTIME" = "tts" ]; then
  RUN_SERVER="$BYOM_DIR/run_vllm.sh"
  LOG_NAME="$RUNTIME"
else
  RUN_SERVER="$BYOM_DIR/run_runtime.sh"
  LOG_NAME="$RUNTIME"
fi

SERVER_LOG="$LOGS/$LOG_NAME.log"

# O tmux encerra a sessão quando o único processo dela sai, e o cliente então
# imprime só "server exited unexpectedly". A causa real (biblioteca ausente,
# argumento inválido, arquivo de modelo) fica no log do runtime, e é a única
# informação que resolve a falha: mostre-a antes de sair.
dump_log() {
  [ -s "$1" ] || return 0
  log "--- tail of $1 ---"
  tail -n 25 "$1" || true
}

rm -f "$BYOM_DIR/vllm.pid"
start_failed=""
if ! tmux new-session -d -s "$TMUX_SESSION" -n "$LOG_NAME" \
    "bash $RUN_SERVER > $SERVER_LOG 2>&1"; then
  start_failed=1
elif ! tmux new-window -t "$TMUX_SESSION" -n gateway \
    "bash $BYOM_DIR/run_gateway.sh > $LOGS/gateway.log 2>&1"; then
  start_failed=1
fi
if [ -n "$start_failed" ]; then
  log "error: tmux could not start the $RUNTIME and gateway windows"
  dump_log "$SERVER_LOG"
  dump_log "$LOGS/gateway.log"
  exit 1
fi

# `run_runtime.sh` publica o PID antes do exec: um processo vivo prova que o
# runtime passou do carregamento do binário. Esperar o servidor ficar pronto
# aqui seria errado (minutos); a prontidão é verificada pelo túnel.
for _ in $(seq 1 20); do
  if [ -s "$BYOM_DIR/vllm.pid" ] && kill -0 "$(cat "$BYOM_DIR/vllm.pid")" 2>/dev/null; then
    break
  fi
  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  log "error: runtime $RUNTIME exited during startup"
  dump_log "$SERVER_LOG"
  dump_log "$LOGS/gateway.log"
  exit 1
fi

log "$RUNTIME and gateway started; logs are in $LOGS"
log "ready"
