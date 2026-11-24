#!/usr/bin/env bash
# Executa o servidor do runtime (llama-server ou sd-server) em uma janela tmux.
# Escreve vllm.pid para manter o contrato de desligamento do gateway.
set -euo pipefail

BYOM_DIR=/content/byom
export PATH="$BYOM_DIR/venv/bin:$PATH"

# Os binários CUDA do llama.cpp declaram libcudart.so.12 e libcublas.so.12. A
# imagem do Colab põe CUDA 13 no ldconfig e mantém o stack 12 dentro dos pacotes
# `nvidia-*-cu12` do Python do kernel — fora do caminho do loader do processo
# que roda por SSH. Sem esses diretórios o servidor nem abre ("error while
# loading shared libraries"). As bibliotecas do driver (libcuda.so.1) já vêm em
# LD_LIBRARY_PATH pelo sshd da VM (/usr/lib64-nvidia).
for cuda_lib in /usr/local/lib/python3*/dist-packages/nvidia/*/lib; do
  if [ -d "$cuda_lib" ]; then
    export LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done

echo $$ > "$BYOM_DIR/vllm.pid"
exec bash "$BYOM_DIR/runtime.cmd"
