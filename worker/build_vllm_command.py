#!/usr/bin/env python3
"""Monte a linha de comando do runtime a partir do payload da sessão.

O gateway continua na frente e escuta em 127.0.0.1; vLLM, llama-server e
sd-server escutam apenas em loopback na porta `vllm_port`.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

MANIFEST = os.environ.get("BYOM_RUNTIME_MANIFEST") or "/content/byom/runtime_manifest.json"
MODEL_DIR = "/content/byom/models"

#: Papéis de imagem e a flag do sd-server de cada um. Confirmado na build
#: fixada (`master-1106-532d2c1`):
#: https://github.com/leejet/stable-diffusion.cpp/blob/532d2c1/docs/models/qwen_image_2.1.md
#: (consultado em 2026-11-24), tabela "Model files".
IMAGE_ROLE_FLAGS = {
    "diffusion": "--diffusion-model",
    "vae": "--vae",
    "text_encoder": "--llm",
    # Confirmado na mesma tabela ("Model files") da build fixada: para edição de
    # imagem com text encoder GGUF, `mmproj-...-F16.gguf` vai em `--llm_vision`.
    "vision": "--llm_vision",
}

#: Papéis obrigatórios. `vision` é opcional: sem ele o servidor sobe e gera
#: imagem; apenas `/v1/images/edits` é recusado pelo gateway.
REQUIRED_IMAGE_ROLES = ("diffusion", "vae", "text_encoder")


def build_args(cfg: dict) -> list[str]:
    """Comando vLLM (mantém a interface usada pelos testes existentes)."""
    args = [
        "/content/byom/venv/bin/vllm",
        "serve",
        cfg["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg["vllm_port"]),
        "--served-model-name",
        cfg["served_model_name"],
        "--max-model-len",
        str(cfg["max_model_len"]),
        "--kv-cache-dtype",
        cfg.get("kv_cache_dtype", "fp8"),
        "--gpu-memory-utilization",
        str(cfg["gpu_memory_utilization"]),
        "--enable-prefix-caching",
    ]
    if cfg.get("enable_auto_tool_choice", True):
        args.append("--enable-auto-tool-choice")
    if cfg.get("tool_call_parser"):
        args.extend(["--tool-call-parser", cfg["tool_call_parser"]])
    if cfg.get("reasoning_parser"):
        args.extend(["--reasoning-parser", cfg["reasoning_parser"]])
    if cfg.get("vllm_language_model_only", True):
        args.append("--language-model-only")
    if cfg.get("vllm_enforce_eager", False):
        args.append("--enforce-eager")
    return args


def _manifest() -> dict:
    try:
        raw = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _resolved(key: str, default: str) -> str:
    value = _manifest().get(key)
    return str(value) if value else default


def build_llama_args(cfg: dict) -> list[str]:
    """Comando llama-server para GGUF, com/mmproj, Jinja e decodificação DSpark."""
    worker = cfg.get("worker") or {}
    sampling = worker.get("sampling") or {}
    args = [
        _resolved("server", "/content/byom/llama.cpp/llama-server"),
        "--model",
        _resolved("main", str(worker.get("main_pattern", cfg["model"]))),
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg["vllm_port"]),
        "--ctx-size",
        str(cfg["max_model_len"]),
        "--alias",
        cfg["served_model_name"],
    ]
    if worker.get("llama_jinja"):
        args.append("--jinja")
    mmproj = worker.get("mmproj_pattern")
    if mmproj:
        args.extend(["--mmproj", _resolved("mmproj", str(mmproj))])
    if worker.get("dspark"):
        drafter = worker.get("drafter_pattern")
        if not drafter:
            raise ValueError("dspark requires drafter_pattern in the profile")
        args.extend(
            [
                "--model-draft",
                _resolved("drafter", str(drafter)),
                "--spec-type",
                "draft-dspark",
                "--spec-draft-n-max",
                "3",
                "--n-gpu-layers-draft",
                "99",
            ]
        )
    for key, flag in (
        ("temperature", "--temp"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
        ("min_p", "--min-p"),
    ):
        if key in sampling:
            args.extend([flag, str(sampling[key])])
    if worker.get("chat_template_kwargs"):
        args.extend(["--chat-template-kwargs", str(worker["chat_template_kwargs"])])
    return args


def _unsupported(reason: str) -> RuntimeError:
    """Falha explícita: melhor recusar do que subir um servidor sem o arquivo."""
    return RuntimeError(f"runtime de imagem não suportado nesta versão: {reason}")


def build_image_args(cfg: dict) -> list[str]:
    """Comando sd-server montado a partir do manifest, nunca de um palpite.

    Cada papel de `[worker.image_roles]` vira uma flag confirmada na
    documentação da build fixada (`--diffusion-model`, `--vae`, `--llm`).
    """
    manifest = _manifest()
    server = manifest.get("server")
    if not isinstance(server, str) or not server:
        raise _unsupported(
            "o manifest não registra o binário sd-server; rode prepare_runtime.py"
        )
    if not Path(server).is_file():
        raise _unsupported(f"o binário sd-server não existe em {server}")
    image = manifest.get("image")
    if not isinstance(image, dict) or not image:
        raise _unsupported(
            "o manifest não registra image; declare [worker] image_roles "
            "(diffusion, vae, text_encoder) no perfil"
        )
    missing = [role for role in REQUIRED_IMAGE_ROLES if not image.get(role)]
    if missing:
        raise _unsupported(f"o manifest não registra o papel {', '.join(missing)}")
    args = [server, "--model-dir", MODEL_DIR]
    for role, flag in IMAGE_ROLE_FLAGS.items():
        if image.get(role):
            args.extend([flag, str(image[role])])
    args.extend(["--listen-ip", "127.0.0.1", "--listen-port", str(cfg["vllm_port"])])
    return args


def build_command(cfg: dict) -> list[str]:
    runtime = cfg.get("runtime", "vllm")
    if runtime == "vllm":
        return build_args(cfg)
    if runtime == "llama-cpp":
        return build_llama_args(cfg)
    if runtime == "image":
        return build_image_args(cfg)
    raise ValueError(f"Unsupported runtime: {runtime!r}")


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        command = build_command(cfg)
    except (RuntimeError, ValueError) as exc:
        # Falha legível: o bootstrap mostra isto em vez de subir sem o arquivo.
        print(f"[byom-runtime] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print("exec " + shlex.join(command))


if __name__ == "__main__":
    main()
