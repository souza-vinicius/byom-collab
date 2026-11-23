#!/usr/bin/env bash
# Executa o servidor do runtime (llama-server ou sd-server) em uma janela tmux.
# Escreve vllm.pid para manter o contrato de desligamento do gateway.
set -euo pipefail

BYOM_DIR=/content/byom
export PATH="$BYOM_DIR/venv/bin:$PATH"

echo $$ > "$BYOM_DIR/vllm.pid"
exec bash "$BYOM_DIR/runtime.cmd"
