"""Estado lateral que permite ao `down` desfazer a configuração de harness.

`up` grava `model` (OpenCode) e `provider`/`model` (dsh) apontando para o proxy
local. Quando a sessão termina, esses valores precisam voltar ao que eram —
senão o agente fica apontando para `127.0.0.1:4000` morto.

O registro mora em `<byom_home>/harness_state.json` (0600): são só ponteiros de
configuração, sem nonce nem credencial.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from . import config, dsh, opencode

log = logging.getLogger("byom.harness")

FILENAME = "harness_state.json"


def state_path() -> Path:
    return config.config_dir() / FILENAME


def load() -> dict[str, Any] | None:
    path = state_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None
    return raw if isinstance(raw, dict) else None


def save(dados: dict[str, Any]) -> Path:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config.atomic_write_text(path, json.dumps(dados, indent=2) + "\n", mode=0o600)
    with contextlib.suppress(OSError):  # FS sem permissões POSIX
        path.chmod(0o600)
    return path


def clear() -> None:
    try:
        state_path().unlink(missing_ok=True)
    except OSError:  # pragma: no cover - FS somente leitura
        log.warning("Could not remove %s", state_path())


def capture(
    opencode_path: Path | None = None, dsh_path: Path | None = None
) -> dict[str, Any]:
    """Lê os valores ATUAIS dos harnesses antes de o `up` sobrescrevê-los."""
    dados: dict[str, Any] = {}
    try:
        op = opencode.resolve_config_path(opencode_path)
    except opencode.OpenCodeConfigError:
        op = None
    if op is not None and op.exists():
        try:
            atual = json.loads(op.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            atual = None
        if isinstance(atual, dict):
            dados["opencode"] = {"path": str(op), "model": atual.get("model")}
    ds = dsh.resolve_config_path(dsh_path)
    if ds.exists():
        padrao = dsh.default_model_values(ds)
        if padrao is not None:
            dados["dsh"] = {"path": str(ds), **padrao}
    return dados


def restore() -> tuple[list[str], str]:
    """Devolve aos harnesses os valores anteriores; apaga o registro.

    Retorna (harnesses restaurados, motivo de eventual falha). Nunca levanta: o
    `down` não pode falhar por causa da configuração de terceiros.
    """
    dados = load()
    if not dados:
        return [], ""
    restaurados: list[str] = []
    falhas: list[str] = []
    op = dados.get("opencode")
    if isinstance(op, dict) and op.get("path"):
        try:
            opencode.restore_default_model(Path(str(op["path"])), op.get("model"))
        except (opencode.OpenCodeConfigError, OSError) as exc:
            falhas.append(f"OpenCode: {exc}")
        else:
            restaurados.append("OpenCode")
    ds = dados.get("dsh")
    if isinstance(ds, dict) and ds.get("path"):
        try:
            dsh.restore_default_model(
                Path(str(ds["path"])), ds.get("provider"), ds.get("model")
            )
        except (dsh.DshConfigError, OSError) as exc:
            falhas.append(f"dsh: {exc}")
        else:
            restaurados.append("dsh")
    if not falhas:
        clear()
    return restaurados, "; ".join(falhas)
