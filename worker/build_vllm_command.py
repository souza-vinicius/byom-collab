#!/usr/bin/env python3
"""Build the vLLM command line from the session payload."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path


def build_args(cfg: dict) -> list[str]:
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


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print("exec " + shlex.join(build_args(cfg)))


if __name__ == "__main__":
    main()
