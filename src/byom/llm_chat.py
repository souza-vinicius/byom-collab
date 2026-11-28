"""Isolated `llm chat` setup, execution, and safe uninstall under `byom_home()`."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from . import config

MARKER_FILENAME = ".byom-managed"
PREFS_FILENAME = "llm-chat.json"
EXTRA_MODELS_FILENAME = "extra-openai-models.yaml"
DEFAULT_MODEL_FILENAME = "default_model.txt"
KEYS_FILENAME = "keys.json"
LOGS_OFF_FILENAME = "logs-off"
LOGS_DB_FILENAME = "logs.db"
DEFAULT_MODEL_ID = "byom"


class LlmChatError(ValueError):
    """Raised when `llm chat` setup, execution, or uninstall preconditions fail."""


@dataclass(frozen=True)
class SetupResult:
    home: Path
    api_base: str
    model_name: str
    model_id: str
    extra_models_path: Path
    prefs_path: Path
    marker_path: Path


@dataclass(frozen=True)
class UninstallTarget:
    home: Path
    marker_path: Path
    entries: tuple[Path, ...]


@dataclass(frozen=True)
class ChatResult:
    returncode: int
    stderr: str = ""

    @property
    def aborted(self) -> bool:
        # click prints "Aborted!" to stderr and exits 1 on Ctrl+C/Ctrl+D; that
        # string is the signal of a normal user cancellation, not a chat failure.
        return self.returncode == 1 and "Aborted!" in self.stderr


def validate_safe_byom_dir(path: Path) -> Path:
    """Reject empty, relative `.`/`..`, symlink, filesystem root, and user home paths."""
    raw = str(path).strip()
    if not raw or raw in {".", ".."}:
        raise LlmChatError(f"Caminho inválido para o diretório do byom: {str(path)!r}")
    if not path.is_absolute():
        # Relativo ao CWD no `rmtree` apagaria o diretório errado.
        raise LlmChatError(
            f"Recusando operar sobre caminho relativo: {raw!r}; use um caminho absoluto"
        )
    if path.is_symlink():
        raise LlmChatError(f"Recusando operar sobre symlink no diretório do byom: {path}")
    resolved = path.resolve()
    home_resolved = Path.home().resolve()
    if resolved == Path(resolved.anchor) or resolved.parent == resolved:
        raise LlmChatError(f"Recusando operar sobre a raiz do sistema: {resolved}")
    if resolved == home_resolved:
        raise LlmChatError(f"Recusando operar sobre o diretório home do usuário: {resolved}")
    if path.exists() and not path.is_dir():
        raise LlmChatError(f"Caminho do byom existe e não é um diretório: {path}")
    return path


def _resolve_safe_home() -> Path:
    try:
        home = config.byom_home()
    except config.ConfigError as exc:
        raise LlmChatError(str(exc)) from exc
    return validate_safe_byom_dir(home)


def ensure_marker(home: Path) -> Path:
    """Write the `.byom-managed` marker that gates `byom uninstall`."""
    marker_path = home / MARKER_FILENAME
    payload = json.dumps({"managed_by": "byom", "version": 1}, indent=2) + "\n"
    config.atomic_write_text(marker_path, payload, mode=0o600)
    return marker_path


def ensure_managed_home() -> Path:
    """Create `byom_home()`, tighten permissions and ensure the marker."""
    home = _resolve_safe_home()
    home.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        with contextlib.suppress(OSError):
            home.chmod(0o700)
    ensure_marker(home)
    return home


def format_extra_openai_models_yaml(
    *,
    model_id: str,
    model_name: str,
    api_base: str,
    api_key_name: str | None = None,
) -> str:
    """Render an `extra-openai-models.yaml` list entry using only the Python stdlib."""
    lines = [
        f"- model_id: {json.dumps(model_id)}",
        f"  model_name: {json.dumps(model_name)}",
        f"  api_base: {json.dumps(api_base)}",
    ]
    if api_key_name:
        lines.append(f"  api_key_name: {json.dumps(api_key_name)}")
    return "\n".join(lines) + "\n"


def _load_prefs(prefs_path: Path) -> dict[str, object]:
    if not prefs_path.exists():
        return {}
    try:
        raw = json.loads(prefs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise LlmChatError(f"Não foi possível ler preferências em {prefs_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise LlmChatError(
            f"Arquivo de preferências inválido em {prefs_path}: esperado objeto JSON"
        )
    return raw


def _validate_url(url: str) -> str:
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LlmChatError(
            f"URL inválida: {url!r}. Informe uma URL http:// ou https:// (ex.: http://127.0.0.1:4000/v1)."
        )
    return cleaned


def _write_keys(keys_path: Path, key_name: str, api_key: str) -> None:
    keys: dict[str, str] = {}
    if keys_path.exists():
        try:
            loaded = json.loads(keys_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise LlmChatError(f"Não foi possível ler {keys_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise LlmChatError(f"Formato inválido em {keys_path}: esperado objeto JSON")
        keys = {str(k): str(v) for k, v in loaded.items()}
    keys[key_name] = api_key
    config.atomic_write_text(keys_path, json.dumps(keys, indent=2) + "\n", mode=0o600)


def run_setup(
    *,
    url: str | None = None,
    model_name: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
) -> SetupResult:
    """Idempotently create `byom_home()`, marker, preferences, and `llm` OpenAI YAML."""
    home = ensure_managed_home()
    prefs_path = home / PREFS_FILENAME
    prev = _load_prefs(prefs_path)

    try:
        cfg = config.load_config()
    except config.ConfigError:
        cfg = config.Config()

    prev_url = str(prev.get("api_base") or "")
    prev_model_name = str(prev.get("model_name") or "")
    prev_model_id = str(prev.get("model_id") or "")

    effective_url = _validate_url(url or prev_url or f"http://127.0.0.1:{cfg.proxy_port}/v1")
    effective_model_name = (model_name or prev_model_name or cfg.served_model_name).strip()
    if not effective_model_name:
        raise LlmChatError("O nome do modelo remoto não pode ser vazio.")
    effective_model_id = (model_id or prev_model_id or DEFAULT_MODEL_ID).strip()
    if not effective_model_id:
        raise LlmChatError("O ID local do modelo no llm não pode ser vazio.")

    keys_path = home / KEYS_FILENAME
    api_key_name: str | None = None
    if api_key is not None:
        cleaned_key = api_key.strip()
        if not cleaned_key:
            raise LlmChatError("A chave de API informada não pode ser vazia.")
        api_key_name = effective_model_id
        _write_keys(keys_path, api_key_name, cleaned_key)
    elif prev.get("api_key_name") and keys_path.exists():
        api_key_name = str(prev["api_key_name"])

    marker_path = home / MARKER_FILENAME
    extra_models_path = home / EXTRA_MODELS_FILENAME
    default_model_path = home / DEFAULT_MODEL_FILENAME

    prefs_payload = (
        json.dumps(
            {
                "api_base": effective_url,
                "model_name": effective_model_name,
                "model_id": effective_model_id,
                "api_key_name": api_key_name,
                "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    yaml_payload = format_extra_openai_models_yaml(
        model_id=effective_model_id,
        model_name=effective_model_name,
        api_base=effective_url,
        api_key_name=api_key_name,
    )

    config.atomic_write_text(prefs_path, prefs_payload, mode=0o600)
    config.atomic_write_text(extra_models_path, yaml_payload, mode=0o600)
    config.atomic_write_text(default_model_path, f"{effective_model_id}\n", mode=0o600)

    return SetupResult(
        home=home,
        api_base=effective_url,
        model_name=effective_model_name,
        model_id=effective_model_id,
        extra_models_path=extra_models_path,
        prefs_path=prefs_path,
        marker_path=marker_path,
    )


def run_chat(
    *,
    system: str | None = None,
    no_history: bool = False,
    continue_last: bool = False,
    model_id: str | None = None,
) -> ChatResult:
    """Execute `[sys.executable, '-m', 'llm', 'chat', ...]` isolated with `LLM_USER_PATH`."""
    home = _resolve_safe_home()
    marker_path = home / MARKER_FILENAME
    prefs_path = home / PREFS_FILENAME
    extra_models_path = home / EXTRA_MODELS_FILENAME

    if not marker_path.is_file() or not prefs_path.is_file() or not extra_models_path.is_file():
        raise LlmChatError(
            "Setup do chat não encontrado. Execute `byom setup` antes de iniciar o `byom chat`."
        )

    prefs = _load_prefs(prefs_path)
    effective_model_id = (model_id or str(prefs.get("model_id") or "") or DEFAULT_MODEL_ID).strip()
    if not effective_model_id:
        raise LlmChatError("Nenhum modelo configurado; execute `byom setup` novamente.")

    cmd = [sys.executable, "-m", "llm", "chat", "-m", effective_model_id]
    if system is not None:
        cmd.extend(["--system", system])
    if continue_last and not no_history:
        cmd.append("--continue")

    env = {**os.environ, "LLM_USER_PATH": str(home)}
    logs_off_path = home / LOGS_OFF_FILENAME
    had_logs_off = logs_off_path.exists()
    if no_history and not had_logs_off:
        config.atomic_write_text(logs_off_path, "", mode=0o600)

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            check=False,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        if no_history and not had_logs_off:
            with contextlib.suppress(OSError):
                logs_off_path.unlink(missing_ok=True)
    return ChatResult(proc.returncode, proc.stderr or "")


def inspect_uninstall_target() -> UninstallTarget:
    """Validate safety preconditions and list everything inside `byom_home()`."""
    home = _resolve_safe_home()
    if not home.exists():
        raise LlmChatError(f"Diretório do byom não encontrado: {home}")
    if not home.is_dir():
        raise LlmChatError(f"Caminho não é um diretório: {home}")

    marker_path = home / MARKER_FILENAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise LlmChatError(
            f"Recusando remover {home}: arquivo-marcador {MARKER_FILENAME!r} não encontrado. "
            "Execute `byom setup` primeiro ou remova manualmente."
        )

    entries = tuple(sorted(home.iterdir(), key=lambda entry: entry.name))
    return UninstallTarget(home=home, marker_path=marker_path, entries=entries)


def execute_uninstall(target: UninstallTarget) -> None:
    """Re-check safety preconditions and remove only `target.home`."""
    validate_safe_byom_dir(target.home)
    if (
        target.marker_path.is_symlink()
        or not target.marker_path.is_file()
        or target.marker_path.parent.resolve() != target.home.resolve()
    ):
        raise LlmChatError(
            f"Recusando remover {target.home}: marcador inválido ({target.marker_path})."
        )
    shutil.rmtree(target.home)
