"""Local byom configuration: `config.toml` and `state.json`.

`state.json` records the association between the created session, its Colab
backend endpoint, the nonce used to validate the worker's identity, and local
process IDs. The nonce is an identity secret: it is not a user credential, but
anyone who reads it and controls the endpoint can pass the proxy's identity
check. Files are written with mode 0600.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import sys
import tempfile
import tomllib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any

from . import catalog

try:  # pragma: no cover - platform-dependent
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

#: Header que carrega o nonce da sessão nos endpoints internos do gateway.
#: O forward do byom é `ssh -L 127.0.0.1:<porta>:127.0.0.1:<porta>`, então todo
#: cliente do túnel aparece como loopback: o header é o que prova identidade.
NONCE_HEADER = "X-BYOM-Nonce"

#: O valor padrão NÃO é literal aqui: vem da propriedade do modelo no catálogo.
#: O modelo padrão sem perfil é o Qwen3.8 AWQ (xhigh conforme a página).
DEFAULTS: dict[str, Any] = {
    "model": "philbert440/Qwen3.8-27B-W4A16-AWQ",
    "max_model_len": 16384,
    "reasoning_effort": catalog.model_by_id("qwen3.8-27b-awq").reasoning_effort,
    "proxy_port": 4000,
    "tunnel_port": 4001,
    "gateway_port": 8001,
    "vllm_port": 8000,
    "idle_timeout_min": 20,
    "max_job_min": 40,
    "bootstrap_timeout_min": 30,
    "watchdog_timeout_s": 60,
    "gpu": "L4",
    "gpu_high_mem": False,
    "profile": "",
    "runtime": "vllm",
    "served_model_name": "qwen3.8-27b",
    "gpu_memory_utilization": 0.92,
    "kv_cache_dtype": "fp8",
    "enable_auto_tool_choice": True,
    "tool_call_parser": "qwen3_coder",
    "reasoning_parser": "qwen3",
    "ssh_key": "~/.ssh/byom_ed25519",
    # `vllm_spec` is a package specification. The spike (H6) records the
    # resolved value; do not pin a version before that.
    "vllm_spec": "vllm",
    "vllm_enforce_eager": False,
    "vllm_language_model_only": True,
}

#: Valores documentados pelas páginas conferidas em 2026-11-23. A origem de
#: cada valor por modelo está em catalog.ModelEntry.reasoning_source.
REASONING_EFFORTS = frozenset(catalog.REASONING_EFFORTS)
PORT_FIELDS = frozenset({"proxy_port", "tunnel_port", "gateway_port", "vllm_port"})
TIMEOUT_FIELDS = frozenset(
    {"idle_timeout_min", "max_job_min", "bootstrap_timeout_min", "watchdog_timeout_s"}
)
BOOL_FIELDS = frozenset(
    {"gpu_high_mem", "enable_auto_tool_choice", "vllm_enforce_eager", "vllm_language_model_only"}
)
STRING_FIELDS = frozenset(
    {
        "model",
        "gpu",
        "profile",
        "runtime",
        "served_model_name",
        "kv_cache_dtype",
        "ssh_key",
        "vllm_spec",
    }
)


class ConfigError(ValueError):
    """A configuration value failed boundary validation."""


class StateFileError(ValueError):
    """The state file exists but is unreadable or invalid.

    It is deliberately distinct from "no state file": `up` must not treat a
    corrupted state as a missing session and allocate another VM.
    """


def byom_home(*, platform: str | None = None) -> Path:
    """Return the exclusive `byom` directory for config, state, and `llm` data.

    Precedence: `BYOM_HOME`, then `BYOM_CONFIG_DIR` (used by existing tests),
    then the OS-standard per-user location named `byom` via the standard library.
    """
    if "BYOM_HOME" in os.environ:
        raw = os.environ["BYOM_HOME"]
        if not raw.strip():
            raise ConfigError("BYOM_HOME cannot be empty")
        return Path(raw).expanduser()
    legacy_override = os.environ.get("BYOM_CONFIG_DIR", "").strip()
    if legacy_override:
        return Path(legacy_override).expanduser()
    plat = platform or sys.platform
    if plat == "darwin":
        return Path.home() / "Library" / "Application Support" / "byom"
    if plat.startswith("win"):
        appdata = os.environ.get("APPDATA", "").strip()
        base = Path(appdata).expanduser() if appdata else Path.home() / "AppData" / "Roaming"
        return base / "byom"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "byom"


def config_dir() -> Path:
    """Return the config directory (delegates to `byom_home`)."""
    return byom_home()


def config_path() -> Path:
    return config_dir() / "config.toml"


def state_path() -> Path:
    return config_dir() / "state.json"


def colab_sessions_path() -> Path:
    """Return the Colab CLI session file used only by byom.

    It isolates byom sessions from the user's sessions: the Colab CLI resolves
    `-s` and `--proxy-mode` against this file.
    """
    return config_dir() / "colab-sessions.json"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_config(cfg: Config) -> None:
    """Check types and ranges at the TOML boundary; raise `ConfigError`."""
    values = asdict(cfg)
    for key in sorted(PORT_FIELDS):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            raise ConfigError(f"Invalid {key}: {value!r}; expected a port between 1 and 65535")
    for key in sorted(TIMEOUT_FIELDS):
        value = values[key]
        if not _is_number(value) or value <= 0:
            raise ConfigError(f"Invalid {key}: {value!r}; expected a positive timeout")
    value = values["max_model_len"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"Invalid max_model_len: {value!r}; expected a positive integer")
    value = values["gpu_memory_utilization"]
    if not _is_number(value) or not 0 < value <= 1:
        raise ConfigError(
            f"Invalid gpu_memory_utilization: {value!r}; expected a number in (0, 1]"
        )
    value = values["runtime"]
    if not isinstance(value, str) or value not in catalog.RUNTIMES:
        choices = ", ".join(catalog.RUNTIMES)
        raise ConfigError(f"Invalid runtime: {value!r}; expected one of {choices}")
    value = values["reasoning_effort"]
    # None é legítimo: a página do modelo (Gemma 4) não documenta o campo e o
    # gateway não deve injetar nada nesse caso.
    if value is not None and (not isinstance(value, str) or value not in REASONING_EFFORTS):
        choices = ", ".join(sorted(REASONING_EFFORTS))
        raise ConfigError(f"Invalid reasoning_effort: {value!r}; expected one of {choices} or none")
    for key in sorted(BOOL_FIELDS):
        if not isinstance(values[key], bool):
            raise ConfigError(f"Invalid {key}: {values[key]!r}; expected a boolean")
    for key in sorted(STRING_FIELDS):
        if not isinstance(values[key], str):
            raise ConfigError(f"Invalid {key}: {values[key]!r}; expected a string")
    for key in ("tool_call_parser", "reasoning_parser"):
        value = values[key]
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"Invalid {key}: {value!r}; expected a string or null")


@dataclass
class Config:
    model: str = DEFAULTS["model"]
    max_model_len: int = DEFAULTS["max_model_len"]
    reasoning_effort: str | None = DEFAULTS["reasoning_effort"]
    proxy_port: int = DEFAULTS["proxy_port"]
    tunnel_port: int = DEFAULTS["tunnel_port"]
    gateway_port: int = DEFAULTS["gateway_port"]
    vllm_port: int = DEFAULTS["vllm_port"]
    idle_timeout_min: int = DEFAULTS["idle_timeout_min"]
    max_job_min: int = DEFAULTS["max_job_min"]
    bootstrap_timeout_min: int = DEFAULTS["bootstrap_timeout_min"]
    watchdog_timeout_s: int = DEFAULTS["watchdog_timeout_s"]
    gpu: str = DEFAULTS["gpu"]
    gpu_high_mem: bool = DEFAULTS["gpu_high_mem"]
    profile: str = DEFAULTS["profile"]
    runtime: str = DEFAULTS["runtime"]
    served_model_name: str = DEFAULTS["served_model_name"]
    gpu_memory_utilization: float = DEFAULTS["gpu_memory_utilization"]
    kv_cache_dtype: str = DEFAULTS["kv_cache_dtype"]
    enable_auto_tool_choice: bool = DEFAULTS["enable_auto_tool_choice"]
    tool_call_parser: str | None = DEFAULTS["tool_call_parser"]
    reasoning_parser: str | None = DEFAULTS["reasoning_parser"]
    ssh_key: str = DEFAULTS["ssh_key"]
    vllm_spec: str = DEFAULTS["vllm_spec"]
    vllm_enforce_eager: bool = DEFAULTS["vllm_enforce_eager"]
    vllm_language_model_only: bool = DEFAULTS["vllm_language_model_only"]

    @property
    def ssh_key_path(self) -> Path:
        return Path(self.ssh_key).expanduser()


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    description: str
    opencode_output_limit: int
    settings: dict[str, Any]
    runtime: str = "vllm"
    catalog_model: str = ""
    catalog_variant: str = ""
    worker: dict[str, Any] | None = None

    @property
    def opencode_model(self) -> str:
        return f"byom/{self.id}"


#: Chaves aceitas na tabela [worker] de um perfil. Valores são strings ou
#: listas de strings; nada é interpretado como código.
WORKER_FIELDS = frozenset(
    {
        "binary_url",
        "binary_sha256",
        "binary_dir",
        "binary_asset",
        "model_repo",
        "model_patterns",
        "image_roles",
        "main_pattern",
        "mmproj_pattern",
        "drafter_pattern",
        "dspark",
        "llama_jinja",
        "chat_template_kwargs",
        "sampling",
    }
)

#: Papéis de imagem aceitos em `[worker.image_roles]`, cada um no formato
#: `repo::pattern`. A flag do sd-server de cada papel está em
#: `worker/build_vllm_command.py` e foi conferida na documentação da build
#: fixada (`master-1106-532d2c1`): `--diffusion-model`, `--vae`, `--llm` e, para
#: edição de imagem, `--llm_vision` (projetor do Qwen3-VL).
#: `vision` é OPCIONAL: sem ele a geração funciona, a edição é recusada.
IMAGE_ROLES = ("diffusion", "vae", "text_encoder", "vision")

#: Papéis exigidos para gerar imagem. `vision` só é exigido em `/v1/images/edits`.
REQUIRED_IMAGE_ROLES = ("diffusion", "vae", "text_encoder")

PROFILE_FIELDS = frozenset(
    {
        "model",
        "served_model_name",
        "max_model_len",
        "reasoning_effort",
        "gpu",
        "gpu_high_mem",
        "vllm_spec",
        "vllm_enforce_eager",
        "vllm_language_model_only",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "enable_auto_tool_choice",
        "tool_call_parser",
        "reasoning_parser",
    }
)


def available_profiles() -> list[Profile]:
    """Returns validated model profiles packaged with the `byom` module."""
    profile_dir = resources.files("byom").joinpath("profiles")
    profiles: list[Profile] = []
    for path in sorted(profile_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or not path.name.endswith(".toml"):
            continue
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        settings = raw.get("config")
        if not isinstance(settings, dict):
            raise ValueError(f"profile {path.name} must define [config]")
        unknown = set(settings) - PROFILE_FIELDS
        if unknown:
            raise ValueError(f"Unknown fields in {path.name}: {', '.join(sorted(unknown))}")
        try:
            validate_config(Config(**settings))
        except ConfigError as exc:
            raise ValueError(f"Invalid values in {path.name}: {exc}") from exc
        output_limit = raw.get("opencode_output_limit", 8192)
        if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit <= 0:
            raise ValueError(f"profile {path.name} must define a positive opencode_output_limit")
        runtime = str(raw.get("runtime", "vllm"))
        if runtime not in catalog.RUNTIMES:
            raise ValueError(f"Invalid runtime in {path.name}: {runtime!r}")
        catalog_model = str(raw.get("catalog_model") or "")
        catalog_variant = str(raw.get("catalog_variant") or "")
        if bool(catalog_model) != bool(catalog_variant):
            raise ValueError(f"{path.name} must declare catalog_model and catalog_variant together")
        if catalog_model:
            try:
                variant = catalog.variant_by_id(catalog_model, catalog_variant)
                entry = catalog.model_by_id(catalog_model)
            except ValueError as exc:
                raise ValueError(f"Invalid catalog reference in {path.name}: {exc}") from exc
            if entry.runtime != runtime:
                raise ValueError(f"Runtime mismatch in {path.name}: {runtime} != {entry.runtime}")
            if not variant.available:
                raise ValueError(f"{path.name} references an unavailable catalog variant")
            limite = catalog.context_limit(catalog_model, catalog_variant)
            if limite and settings["max_model_len"] > limite:
                raise ValueError(
                    f"Invalid values in {path.name}: max_model_len "
                    f"{settings['max_model_len']} acima do máximo servido por "
                    f"{catalog_model}:{catalog_variant} ({limite} tokens)"
                )
        worker = raw.get("worker", {})
        if not isinstance(worker, dict):
            raise ValueError(f"profile {path.name}: [worker] must be a table")
        if runtime != "vllm" and "kv_cache_dtype" in settings:
            # O llama-server não recebe `--cache-type-k/-v`; declarar fp8 ali
            # fazia o orçamento cobrar metade do KV que o runtime não usa.
            raise ValueError(
                f"{path.name} declares kv_cache_dtype, but runtime {runtime!r} always "
                "uses the f16 KV cache; remove the field or use runtime 'vllm'"
            )
        unknown_worker = set(worker) - WORKER_FIELDS
        if unknown_worker:
            raise ValueError(
                f"Unknown [worker] fields in {path.name}: {', '.join(sorted(unknown_worker))}"
            )
        for key, value in worker.items():
            if key == "sampling":
                if not isinstance(value, dict) or not all(
                    isinstance(k, str) and isinstance(v, int | float) and not isinstance(v, bool)
                    for k, v in value.items()
                ):
                    raise ValueError(f"Invalid [worker].sampling in {path.name}: {value!r}")
                continue
            if key == "image_roles":
                if not isinstance(value, dict) or not value:
                    raise ValueError(
                        f"profile {path.name}: [worker.image_roles] must be a non-empty table"
                    )
                unknown_roles = set(value) - set(IMAGE_ROLES)
                if unknown_roles:
                    raise ValueError(
                        f"Unknown [worker.image_roles] in {path.name}: "
                        f"{', '.join(sorted(unknown_roles))}; expected {', '.join(IMAGE_ROLES)}"
                    )
                for role, spec in value.items():
                    if not isinstance(spec, str) or "::" not in spec:
                        raise ValueError(
                            f"Invalid [worker].image_roles.{role} in {path.name}: {spec!r}; "
                            "expected 'repo::pattern'"
                        )
                continue
            if not isinstance(value, str | bool | int | list):
                raise ValueError(f"Invalid [worker].{key} in {path.name}: {value!r}")
            if isinstance(value, list) and not all(isinstance(item, str) for item in value):
                raise ValueError(f"Invalid [worker].{key} in {path.name}: {value!r}")
        profiles.append(
            Profile(
                id=path.name.removesuffix(".toml"),
                name=str(raw.get("name") or path.stem),
                description=str(raw.get("description") or ""),
                opencode_output_limit=output_limit,
                settings=settings,
                runtime=runtime,
                catalog_model=catalog_model,
                catalog_variant=catalog_variant,
                worker=worker,
            )
        )
    return profiles


def load_profile(profile_id: str) -> Profile:
    """Load a profile by stable ID; path input is not accepted."""
    if not profile_id or Path(profile_id).name != profile_id:
        raise ValueError(f"Invalid profile ID: {profile_id!r}")
    for profile in available_profiles():
        if profile.id == profile_id:
            return profile
    names = ", ".join(profile.id for profile in available_profiles()) or "none"
    raise ValueError(f"Unknown profile {profile_id!r}; available profiles: {names}")


def apply_profile(cfg: Config, profile_id: str) -> Profile:
    """Aplica o perfil; reasoning_effort vem do catálogo quando omitido."""
    profile = load_profile(profile_id)
    for key, value in profile.settings.items():
        setattr(cfg, key, value)
    cfg.profile = profile.id
    cfg.runtime = profile.runtime
    # Herança: o valor padrão é propriedade do MODELO no catálogo. Um perfil
    # pode sobrescrever explicitamente em [config]; o cliente pode sobrescrever
    # por requisição (chat_template_kwargs.reasoning_effort no gateway).
    if "reasoning_effort" not in profile.settings and profile.catalog_model:
        cfg.reasoning_effort = catalog.model_by_id(profile.catalog_model).reasoning_effort
    return profile


def active_catalog_ref(cfg: Config) -> tuple[str, str]:
    """Return (model_id, variant_id) for the active config, defaulting to AWQ/vLLM."""
    if cfg.profile:
        profile = load_profile(cfg.profile)
        if profile.catalog_model:
            return profile.catalog_model, profile.catalog_variant
    return "qwen3.8-27b-awq", "q4-awq"


def check_compatibility(cfg: Config) -> catalog.CompatibilityResult:
    """Check the declared GPU against the active variant before any allocation."""
    model_id, variant_id = active_catalog_ref(cfg)
    gpu = catalog.resolve_gpu(cfg.gpu, cfg.gpu_high_mem)
    return catalog.check_fit(
        model_id, variant_id, gpu.id, cfg.max_model_len, kv_cache_dtype=cfg.kv_cache_dtype
    )


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        # JSON strings are valid basic TOML strings (quotes and escapes match).
        return json.dumps(value)
    raise TypeError(f"Unsupported value in config.toml: {value!r}")


def dumps_config(cfg: Config) -> str:
    lines = []
    for key, value in asdict(cfg).items():
        # TOML não tem null. Ausência de reasoning_effort = herdar do catálogo.
        if value is None:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def validate_context_limit(cfg: Config, model_id: str, variant_id: str) -> None:
    """Recusa `max_model_len` acima do contexto que a variante declara servir."""
    limite = catalog.context_limit(model_id, variant_id)
    if limite and cfg.max_model_len > limite:
        raise ConfigError(
            f"Invalid max_model_len: {cfg.max_model_len}; o máximo servido por "
            f"{model_id}:{variant_id} é {limite} tokens"
        )


def load_config() -> Config:
    path = config_path()
    if not path.exists():
        return Config()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        # Sem esta tradução o erro de parse escapa como TOMLDecodeError, que não
        # é ConfigError: a TUI cairia no handler genérico e fecharia o menu.
        raise ConfigError(
            f"config.toml não pode ser interpretado ({path}): {exc}. "
            "Corrija a sintaxe ou mova o arquivo para fora do diretório do byom."
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"config.toml não pode ser lido ({path}): {exc}") from exc
    known = {f.name for f in fields(Config)}
    values = {k: v for k, v in raw.items() if k in known}
    cfg = Config(**values)
    if "reasoning_effort" not in values and cfg.profile:
        # Perfil ativo sem valor no arquivo: herda a propriedade do modelo.
        try:
            profile = load_profile(cfg.profile)
        except ValueError:
            pass  # perfil inválido é tratado por validate_config/outras camadas
        else:
            if profile.catalog_model:
                cfg.reasoning_effort = catalog.model_by_id(profile.catalog_model).reasoning_effort
    validate_config(cfg)
    model_id, variant_id = active_catalog_ref(cfg)
    validate_context_limit(cfg, model_id, variant_id)
    return cfg


def save_config(cfg: Config) -> Path:
    path = config_path()
    _atomic_write(path, dumps_config(cfg))
    return path


@dataclass
class State:
    session: str
    endpoint: str
    nonce: str
    created_at: str
    daemon_pid: int | None = None
    watchdog_pid: int | None = None
    #: Campo 22 de `/proc/<pid>/stat` no instante em que o PID foi registrado.
    #: Sem ele, um PID reciclado depois de um reboot seria tratado como byom.
    daemon_start_ticks: int | None = None
    watchdog_start_ticks: int | None = None


def process_start_ticks(pid: int) -> int | None:
    """Instante de início do processo (campo 22 de `/proc/<pid>/stat`).

    Devolve None quando o processo não existe mais ou quando a plataforma não
    tem `/proc`; nesse caso o chamador deve tratar o PID como morto.
    """
    if pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # O campo 2 (`comm`) pode conter espaços e parênteses: o split começa depois
    # do último `)`, então `fields[0]` é o campo 3 e o campo 22 é `fields[19]`.
    _, _, rest = raw.rpartition(")")
    fields = rest.split()
    if len(fields) < 20:
        return None
    try:
        value = int(fields[19])
    except ValueError:
        return None
    return value if value > 0 else None


def process_cmdline(pid: int) -> str:
    """Linha de comando do processo, usada só quando não há start time."""
    if pid <= 0:
        return ""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def validate_state(state: State) -> None:
    """Check state types before the values reach `os.kill` or a VM allocation."""
    for key in ("session", "endpoint", "nonce", "created_at"):
        value = getattr(state, key)
        if not isinstance(value, str) or (key in {"session", "nonce"} and not value):
            raise StateFileError(f"Invalid {key} in state.json: {value!r}; expected a string")
    for key in ("daemon_pid", "watchdog_pid", "daemon_start_ticks", "watchdog_start_ticks"):
        value = getattr(state, key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise StateFileError(
                f"Invalid {key} in state.json: {value!r}; expected a positive integer"
            )


def load_state() -> State | None:
    """Return None only when the file is absent; raise on unreadable/invalid state."""
    path = state_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise StateFileError(
            f"state.json exists but cannot be read or parsed ({exc}); "
            "a VM may still be allocated. Stop it in Colab, then remove the file."
        ) from exc
    if not isinstance(raw, dict):
        raise StateFileError("state.json must contain a JSON object")
    known = {f.name for f in fields(State)}
    filtered = {k: v for k, v in raw.items() if k in known}
    if "session" not in filtered or "nonce" not in filtered:
        raise StateFileError("state.json is missing required fields (session, nonce)")
    filtered.setdefault("endpoint", "")
    filtered.setdefault("created_at", "")
    try:
        state = State(**filtered)
    except TypeError as exc:  # pragma: no cover - defensivo: as chaves já foram filtradas
        raise StateFileError(f"state.json inválido: {exc}") from exc
    validate_state(state)
    return state


def _save_state_unlocked(state: State) -> Path:
    validate_state(state)
    path = state_path()
    _atomic_write(path, json.dumps(asdict(state), indent=2) + "\n")
    return path


def save_state(state: State, *, expected_session: str | None = None) -> Path:
    """Atomically persist state, serializing writers from all local processes.

    `expected_session` é o dono declarado da escrita: se o `state.json` atual
    pertence a outra sessão (um `up` que acabou de registrar outra VM), nada é
    gravado e `StateFileError` explica o motivo. Sem isso, um `down` antigo
    apagava o registro da sessão nova.
    """
    with state_lock():
        _assert_owner(expected_session)
        return _save_state_unlocked(state)


def _assert_owner(expected_session: str | None) -> None:
    """Confere o dono do `state.json` atual; deve rodar dentro de `state_lock`."""
    if expected_session is None:
        return
    current = load_state()
    if current is not None and current.session != expected_session:
        raise StateFileError(
            f"state.json pertence à sessão {current.session!r}; "
            f"a escrita da sessão {expected_session!r} foi recusada"
        )


def update_state(*, expected_session: str | None = None, **changes: Any) -> State:
    """Re-read state under the lock, apply `changes`, then persist atomically.

    Processes that only own one PID (the daemon, the watchdog supervisor) must
    use this instead of `save_state` so they do not erase each other's PIDs.
    """
    unknown = set(changes) - {f.name for f in fields(State)}
    if unknown:
        raise ValueError(f"Unknown state fields: {', '.join(sorted(unknown))}")
    with state_lock():
        current = load_state()
        if current is None:
            raise StateFileError("No active session is registered in state.json")
        _assert_owner(expected_session)
        for key, value in changes.items():
            setattr(current, key, value)
        _save_state_unlocked(current)
        return current


def clear_state(*, expected_session: str | None = None) -> None:
    """Remove `state.json` apenas quando a sessão é a esperada (se informada)."""
    with state_lock():
        _assert_owner(expected_session)
        with contextlib.suppress(FileNotFoundError):
            state_path().unlink()



def nonce_matches(payload: Any, state: State) -> bool:
    """True when a gateway payload carries this session's identity nonce."""
    if not isinstance(payload, dict):
        return False
    nonce = payload.get("nonce")
    return isinstance(nonce, str) and bool(nonce) and nonce == state.nonce


@contextlib.contextmanager
def state_lock() -> Iterator[None]:
    """Cross-process advisory lock around state read-modify-write cycles."""
    if fcntl is None:  # pragma: no cover - non-POSIX platforms
        yield
        return
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def up_lock_path() -> Path:
    """Arquivo do lock da transação de `up` (exclusivo do byom)."""
    return config_dir() / "up.lock"


@contextlib.contextmanager
def up_lock(holder: str = "up") -> Iterator[None]:
    """Lock exclusivo e não bloqueante de uma transação `byom up`/`byom down`.

    Dois `up` simultâneos alocariam duas GPUs, e um `down` lento (até ~11 min
    com os timeouts do Colab) apagaria o state de um `up` que acabou de criar
    outra sessão: o segundo processo é recusado com `ConfigError` em vez de
    esperar. `flock` é liberado pelo kernel quando o processo morre, então uma
    execução interrompida não deixa lock preso.
    """
    if fcntl is None:  # pragma: no cover - non-POSIX platforms
        yield
        return
    path = up_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    detalhe = (
        "já existe um up em andamento"
        if holder == "up"
        else "um `up`/`down` está em andamento"
    )
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ConfigError(
                f"{detalhe} (lock {path} ocupado); "
                "aguarde a outra execução ou encerre-a antes de tentar de novo"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)



def atomic_write_text(
    path: Path, text: str, *, mode: int | None = None, newline: str | None = None
) -> None:
    """Write text via temp file + `os.replace`, preserving the target mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.resolve()
    if mode is None:
        mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o600
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _atomic_write(path: Path, text: str) -> None:
    """Write the file atomically to avoid leaving partial state."""
    atomic_write_text(path, text)


async def sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    """Sleep, or return immediately when the caller is asked to stop."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)
