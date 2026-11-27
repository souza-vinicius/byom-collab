#!/usr/bin/env python3
"""Monte a linha de comando do runtime a partir do payload da sessão.

O gateway continua na frente e escuta em 127.0.0.1; vLLM, llama-server,
sd-server e o control do Strata escutam apenas em loopback na porta
`vllm_port`.
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


#: Tipos de spec cuja cabeça de rascunho vem EMBUTIDA no arquivo principal, em
#: vez de um `-md` separado. O `-mtp.gguf` do Bonsai 2.5 é carregado como
#: modelo (`-m`), não como drafter: "Load an MTP file instead of the plain one,
#: not alongside it" (docs.prismml.com/models/bonsai-2-5-27b.md, 2026-11-24).
#: Os demais tipos (`draft-simple`, `draft-eagle3`, `draft-dflash`,
#: `draft-dspark`) precisam de `-md <arquivo>`.
EMBEDDED_DRAFTER_SPEC_TYPES = frozenset({"draft-mtp"})


def _spec(worker: dict) -> dict | None:
    """Tabela de spec efetiva do `[worker]`: `[worker.spec]` ou o `dspark` legado.

    `dspark = true` equivale a `spec.type = "draft-dspark"` com `n_max = 3`, o
    argv que o perfil DSpark já produzia. Sem spec nenhuma, devolve None.
    """
    spec = worker.get("spec")
    if spec is not None:
        if not isinstance(spec, dict):
            raise ValueError("[worker.spec] must be a table")
        if not isinstance(spec.get("type"), str) or not spec["type"].strip():
            raise ValueError("[worker.spec] requires a non-empty type (e.g. draft-mtp)")
        return dict(spec)
    if worker.get("dspark"):
        if not worker.get("drafter_pattern"):
            raise ValueError("dspark requires drafter_pattern in the profile")
        return {"type": "draft-dspark", "n_max": 3}
    return None


def build_llama_args(cfg: dict) -> list[str]:
    """Comando llama-server para GGUF, com/mmproj, Jinja, attention e spec."""
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
    if worker.get("flash_attn"):
        # `-fa on` é a forma dos docs do publicador e a que o binário aceita:
        # `-fa, --flash-attn [on|off|auto]` no `--help` da build fixada.
        args.extend(["-fa", "on"])
    mmproj = worker.get("mmproj_pattern")
    if mmproj:
        args.extend(["--mmproj", _resolved("mmproj", str(mmproj))])
    spec = _spec(worker)
    if spec is not None:
        tipo = str(spec["type"])
        drafter = spec.get("drafter_pattern") or worker.get("drafter_pattern")
        if not drafter and tipo not in EMBEDDED_DRAFTER_SPEC_TYPES:
            raise ValueError(
                f"spec type {tipo!r} needs drafter_pattern (a separate draft file "
                "passed as --model-draft); only draft-mtp embeds the head"
            )
        if drafter:
            args.extend(["--model-draft", _resolved("drafter", str(drafter))])
        args.extend(["--spec-type", tipo])
        for key, flag in (
            ("n_min", "--spec-draft-n-min"),
            ("n_max", "--spec-draft-n-max"),
            ("p_min", "--spec-draft-p-min"),
        ):
            if key in spec:
                args.extend([flag, str(spec[key])])
        if drafter:
            args.extend(["--n-gpu-layers-draft", "99"])
    for key, flag in (
        ("temperature", "--temp"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
        ("min_p", "--min-p"),
        # Penas de presença/repetição: os modelos pequenos da mesma família
        # pedem `--presence-penalty 1.0 --repeat-penalty 1.0 --repeat-last-n 64`
        # (docs.prismml.com/run/llamacpp.md, 2026-11-24); o 27B não usa nenhuma.
        ("presence_penalty", "--presence-penalty"),
        ("repeat_penalty", "--repeat-penalty"),
        ("repeat_last_n", "--repeat-last-n"),
    ):
        if key in sampling:
            args.extend([flag, str(sampling[key])])
    if worker.get("chat_template_kwargs"):
        args.extend(["--chat-template-kwargs", str(worker["chat_template_kwargs"])])
    return args


def _unsupported(reason: str) -> RuntimeError:
    """Falha explícita: melhor recusar do que subir um servidor sem o arquivo."""
    return RuntimeError(f"runtime de imagem não suportado nesta versão: {reason}")


def _unsupported_strata(reason: str) -> RuntimeError:
    """Mesma regra do sd-server: sem manifest completo, o argv não é adivinhado."""
    return RuntimeError(f"runtime strata não pode subir: {reason}")


#: Valores de `--kv` do engine `strata-inkling` 0.1.70 que o catálogo modela (1
#: byte e 2 bytes por elemento). O binário aceita `auto|fp16|fp8|k8v8|k8v4` e
#: recusa `int8` com "unknown --kv value 'int8'" — por isso o perfil declara
#: `fp8` e não `int8`.
STRATA_KV_VALUES = ("fp16", "fp8")


def strata_control_root(manifest: dict) -> str:
    """Diretório do control: o cwd que faz `-m serve.server` achar `serve`."""
    raiz = manifest.get("control_root")
    if isinstance(raiz, str) and raiz:
        return raiz
    server = manifest.get("server")
    if isinstance(server, str) and server:
        # <control>/python/bin/python3 -> <control>
        return str(Path(server).parent.parent.parent)
    return ""


def build_strata_args(cfg: dict, manifest: dict | None = None) -> list[str]:
    """Servidor HTTP do control + run config JSON do engine `strata-inkling`.

    O argv do engine não é a linha de comando: o control lê um JSON
    (`--config`) com `exe`, `args`, `cwd`, `model_name`, `log` e `arch`, e sobe o
    binário com `--serve`. Escrever esse JSON aqui é o equivalente do
    `runtime_manifest` do llama.cpp — o pack, o contexto e o KV vêm do perfil,
    nunca de um palpite.
    """
    dados = manifest if manifest is not None else _manifest()
    server = dados.get("server")
    engine = dados.get("engine")
    pack = dados.get("pack")
    for nome, valor in (("server", server), ("engine", engine), ("pack", pack)):
        if not isinstance(valor, str) or not valor:
            raise _unsupported_strata(
                f"o manifest não registra {nome}; rode prepare_runtime.py no worker"
            )
    for nome, valor in (("server", server), ("engine", engine), ("pack", pack)):
        if not Path(str(valor)).is_file():
            raise _unsupported_strata(f"{nome} não existe no worker: {valor}")

    kv = str(cfg.get("kv_cache_dtype") or "fp8").strip().lower()
    if kv not in STRATA_KV_VALUES:
        raise ValueError(
            f"kv_cache_dtype {kv!r} não é um valor de --kv do engine do Strata "
            f"({', '.join(STRATA_KV_VALUES)}); o binário 0.1.70 recusa 'int8'"
        )
    worker = cfg.get("worker") or {}
    extra = worker.get("run_config_extra") or []
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ValueError("[worker].run_config_extra must be a list of strings")

    args = [
        "--native",
        str(pack),
        "--max-context",
        str(cfg["max_model_len"]),
        "--kv",
        kv,
        # Um pedido por vez: é o que o gateway já serializa (`DEFAULT_MAX_CONCURRENT_JOBS`)
        # e o que evita cobrar VRAM de slots que ninguém usa.
        "--parallel",
        "1",
    ]
    args.extend(str(item) for item in extra)

    base = Path(MANIFEST).parent
    model_name = str(cfg.get("served_model_name") or cfg.get("model") or "model")
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in model_name)[:64]
    run_config = base / f"strata-{slug}.json"
    tokenizer = base / "strata-tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    payload = {
        "exe": str(engine),
        "args": args,
        "cwd": str(Path(str(engine)).parent),
        "model_name": model_name,
        "log": str(base / "strata-engine.log"),
        # `arch` é o que faz o serve tratar o engine como Inkling (`serving`,
        # dialect de esforço, template) — `serve/inkling_kind.py`, ARCH.
        "arch": "inkling2_loop",
        # Sem esta chave o control usaria `<control>/pack/full/tokenizer`, que não
        # existe no tarball; com ela o próprio servidor gera o `vocab.json` do
        # manifest do `.strata` na primeira subida.
        "tokenizer": str(tokenizer),
    }
    run_config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return [
        str(server),
        "-m",
        "serve.server",
        "--engine",
        "strata",
        "--config",
        str(run_config),
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg["vllm_port"]),
        "--offline",
    ]


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
    if runtime == "strata":
        return build_strata_args(cfg)
    raise ValueError(f"Unsupported runtime: {runtime!r}")


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        command = build_command(cfg)
    except (RuntimeError, ValueError) as exc:
        # Falha legível: o bootstrap mostra isto em vez de subir sem o arquivo.
        print(f"[byom-runtime] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if cfg.get("runtime") == "strata":
        # `python -m serve.server` resolve `serve` e `strata_control` pelo cwd (o
        # server importa `strata_control.user_args` na primeira linha de código).
        # O `runtime.cmd` é executado por `bash`, então um `cd` antes do `exec` é
        # o jeito de fixar o diretório sem depender de onde o tmux abriu a janela.
        raiz = strata_control_root(_manifest())
        if raiz:
            print(f"cd {shlex.quote(raiz)}")
    print("exec " + shlex.join(command))


if __name__ == "__main__":
    main()
