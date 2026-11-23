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

MANIFEST = "/content/byom/runtime_manifest.json"
MODEL_DIR = "/content/byom/models"


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
    """Comando llama-server para GGUF, com/mmproj e decodificação DSpark."""
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


def build_image_args(cfg: dict) -> list[str]:
    """Comando sd-server: expõe /v1/images/generations, não /v1/chat/completions."""
    binary = os.environ.get("BYOM_SD_SERVER", "/content/byom/sdcpp/sd-server")
    return [
        binary,
        "--model-dir",
        MODEL_DIR,
        "--listen-ip",
        "127.0.0.1",
        "--listen-port",
        str(cfg["vllm_port"]),
    ]


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
    print("exec " + shlex.join(build_command(cfg)))


if __name__ == "__main__":
    main()
