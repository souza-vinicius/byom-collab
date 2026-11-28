"""Local Open WebUI frontend wired to the byom proxy."""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import socket
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import config, llm_chat

OPEN_WEBUI_VERSION = "0.13.5"
OPEN_WEBUI_PYTHON = "3.12"
DEFAULT_PORT = 8080
DATA_DIRNAME = "webui"
SECRET_FILENAME = "webui-secret"
HEALTH_PATH = "/health"
READY_TIMEOUT_S = 180.0
POLL_INTERVAL_S = 0.5
STOP_TIMEOUT_S = 10.0


class WebUIError(ValueError):
    """Raised when the local frontend cannot be planned or started."""


@dataclass(frozen=True)
class WebUIPlan:
    argv: tuple[str, ...]
    env: dict[str, str]
    url: str
    proxy_url: str
    data_dir: Path
    port: int
    overrides: tuple[str, ...] = ()


def _validate_port(port: int) -> int:
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise WebUIError(f"Porta inválida: {port!r}; use um valor entre 1 e 65535")
    return port


def build_command(*, port: int) -> list[str]:
    _validate_port(port)
    return [
        "uv",
        "tool",
        "run",
        "--python",
        OPEN_WEBUI_PYTHON,
        "--from",
        f"open-webui=={OPEN_WEBUI_VERSION}",
        "open-webui",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


#: Variáveis que pertencem ao isolamento do byom: nunca aceitam override,
#: senão o frontend passa a ler/escrever fora de `byom_home()`.
ISOLATION_KEYS = frozenset({"DATA_DIR", "WEBUI_SECRET_KEY", "WEBUI_URL"})


def _env_defaults(
    *,
    proxy_port: int,
    port: int,
    data_dir: Path,
    secret_key: str,
    served_model_name: str,
) -> dict[str, str]:
    """Valores que o byom fornece ao Open WebUI (fonte: docs.openwebui.com, 2026-11-28)."""
    _validate_port(proxy_port)
    _validate_port(port)
    base = f"http://127.0.0.1:{proxy_port}/v1"
    defaults = {
        "DATA_DIR": str(data_dir),
        "WEBUI_SECRET_KEY": secret_key,
        "WEBUI_AUTH": "false",
        "WEBUI_URL": f"http://127.0.0.1:{port}",
        "OPENAI_API_BASE_URL": base,
        "OPENAI_API_BASE_URLS": base,
        "OPENAI_API_KEY": "byom",
        "OPENAI_API_KEYS": "byom",
        "ENABLE_OPENAI_API": "true",
        "ENABLE_OLLAMA_API": "false",
        # Sem isto, um URL salvo no webui.db pelo painel admin vence as variáveis
        # de ambiente nas próximas execuções (persistência ligada por default).
        "ENABLE_PERSISTENT_CONFIG": "false",
        "ENABLE_VERSION_UPDATE_CHECK": "false",
    }
    # `DEFAULT_MODELS` pré-seleciona IDs existentes em /v1/models; o gateway
    # publica `cfg.served_model_name` (alias do runtime, ou id sintético no TTS).
    modelo = served_model_name.strip()
    if modelo:
        defaults["DEFAULT_MODELS"] = modelo
    return defaults


def build_env(
    *,
    proxy_port: int,
    port: int,
    data_dir: Path,
    secret_key: str,
    served_model_name: str = "",
) -> dict[str, str]:
    """Completa o ambiente do frontend sem sobrescrever escolhas do usuário."""
    env = {**os.environ}
    for name in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    defaults = _env_defaults(
        proxy_port=proxy_port,
        port=port,
        data_dir=data_dir,
        secret_key=secret_key,
        served_model_name=served_model_name,
    )
    for name, value in defaults.items():
        if name in ISOLATION_KEYS:
            env[name] = value
        else:
            env.setdefault(name, value)
    return env


def _secret(path: Path) -> str:
    if path.exists():
        try:
            value = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            value = ""
        if value:
            return value
    value = secrets.token_urlsafe(32)
    config.atomic_write_text(path, value + "\n", mode=0o600)
    return value


def plan(*, port: int = DEFAULT_PORT, proxy_port: int | None = None) -> WebUIPlan:
    """Resolve the command, environment and data directory for the frontend."""
    _validate_port(port)
    if shutil.which("uv") is None:
        raise WebUIError("uv não encontrado no PATH; instale o uv para iniciar o frontend")
    cfg = config.load_config()
    effective_proxy = _validate_port(
        proxy_port if proxy_port is not None else cfg.proxy_port
    )
    home = llm_chat.ensure_managed_home()
    target = home / DATA_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        with contextlib.suppress(OSError):
            target.chmod(0o700)
    url = f"http://127.0.0.1:{port}"
    secret_key = _secret(home / SECRET_FILENAME)
    defaults = _env_defaults(
        proxy_port=effective_proxy,
        port=port,
        data_dir=target,
        secret_key=secret_key,
        served_model_name=cfg.served_model_name,
    )
    overrides = tuple(
        name
        for name in defaults
        if name in os.environ and name not in ISOLATION_KEYS
    )
    return WebUIPlan(
        argv=tuple(build_command(port=port)),
        env=build_env(
            proxy_port=effective_proxy,
            port=port,
            data_dir=target,
            secret_key=secret_key,
            served_model_name=cfg.served_model_name,
        ),
        url=url,
        proxy_url=f"http://127.0.0.1:{effective_proxy}/v1",
        data_dir=target,
        port=port,
        overrides=overrides,
    )


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_for_health(url: str, proc: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    health = f"{url}{HEALTH_PATH}"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            response = httpx.get(health, timeout=1.0)
        except httpx.HTTPError:
            pass
        else:
            if response.status_code == 200:
                return True
        time.sleep(POLL_INTERVAL_S)
    return False


def run(plan: WebUIPlan, *, open_browser: bool = True) -> int:
    """Start Open WebUI in the foreground and return its exit code."""
    if not port_available(plan.port):
        raise WebUIError(
            f"Porta {plan.port} já está em uso; escolha outra com `byom web --port`"
        )
    try:
        proc = subprocess.Popen(list(plan.argv), env=plan.env)
    except OSError as exc:
        raise WebUIError(f"Não foi possível iniciar o Open WebUI: {exc}") from exc
    try:
        if open_browser and _wait_for_health(plan.url, proc, READY_TIMEOUT_S):
            webbrowser.open(plan.url)
        return proc.wait()
    except KeyboardInterrupt:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            proc.wait()
        return 130
