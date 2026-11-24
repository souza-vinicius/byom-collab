#!/usr/bin/env python3
"""argv do llama-server por perfil — sonda que imprime, não teste de CI.

Uso:

    uv run python scripts/llama_argv.py                       # todos os llama-cpp
    uv run python scripts/llama_argv.py ternary-bonsai-25-27b-ptq1-mtp

Monta o argv com o MESMO código que o worker usa em produção
(`worker/build_vllm_command.py`) a partir do TOML embarcado. Sem
`runtime_manifest.json` os caminhos caem nos padrões do perfil — é o que
interessa aqui: a lista de flags. Nada é baixado, nada sobe.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

from byom import catalog, config

ROOT = Path(__file__).resolve().parent.parent


def _build_module():
    spec = importlib.util.spec_from_file_location(
        "byom_build_argv", ROOT / "worker" / "build_vllm_command.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - só em instalação quebrada
        raise SystemExit("worker/build_vllm_command.py não encontrado")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def argv_do_perfil(profile: config.Profile, build_module) -> list[str]:
    """argv de um perfil, com o sampling do catálogo quando o TOML não declara."""
    worker = dict(profile.worker or {})
    if not worker.get("sampling") and profile.catalog_model:
        worker["sampling"] = dict(catalog.model_by_id(profile.catalog_model).sampling)
    return build_module.build_llama_args(
        {
            "runtime": profile.runtime,
            "model": profile.settings["model"],
            "served_model_name": profile.settings["served_model_name"],
            "max_model_len": int(profile.settings["max_model_len"]),
            "vllm_port": 8000,
            "worker": worker,
        }
    )


def main(argv: list[str] | None = None) -> int:
    pedidos = list(argv if argv is not None else sys.argv[1:])
    build_module = _build_module()
    perfis = [
        perfil
        for perfil in config.available_profiles()
        if perfil.runtime == "llama-cpp" and (not pedidos or perfil.id in pedidos)
    ]
    if not perfis:
        print("nenhum perfil llama-cpp casa com o pedido", file=sys.stderr)
        return 1
    for perfil in sorted(perfis, key=lambda item: item.id):
        gpu = perfil.settings.get("gpu", "?")
        print(f"# {perfil.id} — ctx {int(perfil.settings['max_model_len'])}, gpu {gpu}")
        print("  " + shlex.join(argv_do_perfil(perfil, build_module)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
