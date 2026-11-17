"""Local byom configuration: `config.toml` and `state.json`.

No secrets are stored in these files. `state.json` only records the association
between the created session, its Colab backend endpoint, the nonce used to
validate the worker's identity, and local process IDs.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import tomllib
from dataclasses import asdict, dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any

#: Default values for `config.toml`. See PRD 2.3.
DEFAULTS: dict[str, Any] = {
    "model": "philbert440/Qwen3.8-27B-W4A16-AWQ",
    "max_model_len": 16384,
    "reasoning_effort": "medium",
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


def config_dir() -> Path:
    """Return the config directory; tests can override it with `BYOM_CONFIG_DIR`."""
    override = os.environ.get("BYOM_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "byom"


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


@dataclass
class Config:
    model: str = DEFAULTS["model"]
    max_model_len: int = DEFAULTS["max_model_len"]
    reasoning_effort: str = DEFAULTS["reasoning_effort"]
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

    @property
    def opencode_model(self) -> str:
        return f"byom/{self.id}"


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
        Config(**settings)
        output_limit = raw.get("opencode_output_limit", 8192)
        if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit <= 0:
            raise ValueError(f"profile {path.name} must define a positive opencode_output_limit")
        profiles.append(
            Profile(
                id=path.name.removesuffix(".toml"),
                name=str(raw.get("name") or path.stem),
                description=str(raw.get("description") or ""),
                opencode_output_limit=output_limit,
                settings=settings,
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
    """Apply model/runtime settings while preserving local connection options."""
    profile = load_profile(profile_id)
    for key, value in profile.settings.items():
        setattr(cfg, key, value)
    cfg.profile = profile.id
    return profile


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
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def load_config() -> Config:
    path = config_path()
    if not path.exists():
        return Config()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})


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


def load_state() -> State | None:
    path = state_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    known = {f.name for f in fields(State)}
    filtered = {k: v for k, v in raw.items() if k in known}
    if "session" not in filtered or "nonce" not in filtered:
        return None
    filtered.setdefault("endpoint", "")
    filtered.setdefault("created_at", "")
    return State(**filtered)


def save_state(state: State) -> Path:
    path = state_path()
    _atomic_write(path, json.dumps(asdict(state), indent=2) + "\n")
    return path


def clear_state() -> None:
    with contextlib.suppress(FileNotFoundError):
        state_path().unlink()


def _atomic_write(path: Path, text: str) -> None:
    """Write the file atomically to avoid leaving partial state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
