"""Merge BYOM settings into the global OpenCode V1 config."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config

PLUGIN_NAME = "opencode-models-discovery"
# Pinned on 23/11/2026: `latest` made `up` non-reproducible. npm lists 2.2.1
# as latest and its README documents a separate OpenCode v1 adapter, but V1
# plugin execution under V2 was not verified here (see README).
PLUGIN_SPEC = f"{PLUGIN_NAME}@2.2.1"
PROVIDER_ID = "byom"


class OpenCodeConfigError(Exception):
    """Safe error raised while locating, reading, or updating OpenCode config."""


@dataclass(frozen=True)
class SyncResult:
    path: Path
    model: str
    changed: bool


def resolve_config_path(explicit_path: Path | None = None) -> Path:
    """Resolve the primary global OpenCode config file without editing project configs."""
    if explicit_path is not None:
        return explicit_path.expanduser()

    override = os.environ.get("OPENCODE_CONFIG")
    if override:
        return Path(override).expanduser()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        config_home = Path(xdg_config_home).expanduser()
        if not config_home.is_absolute():
            raise OpenCodeConfigError("XDG_CONFIG_HOME must be an absolute path")
    else:
        config_home = Path.home() / ".config"

    opencode_dir = config_home / "opencode"
    for name in ("opencode.jsonc", "opencode.json"):
        candidate = opencode_dir / name
        if candidate.exists():
            return candidate
    return opencode_dir / "opencode.json"


def sync_config(cfg: config.Config, explicit_path: Path | None = None) -> SyncResult:
    """Merge BYOM profiles and endpoint while preserving the rest of the config."""
    if os.environ.get("OPENCODE_CONFIG_CONTENT"):
        raise OpenCodeConfigError(
            "OPENCODE_CONFIG_CONTENT is set and may override the global config; "
            "unset it or use a config without this override before running `byom up`"
        )

    path = resolve_config_path(explicit_path)
    original = _read_config(path)
    updated = _merge_config(original, cfg)
    changed = updated != original
    if changed:
        _atomic_write(path, json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
    return SyncResult(path=path, model=updated["model"], changed=changed)


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"$schema": "https://opencode.ai/config.json"}
    if not path.is_file():
        raise OpenCodeConfigError(f"OpenCode config path is not a file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenCodeConfigError(
            f"Could not update {path}: the file must contain valid JSON "
            "(JSONC with comments or trailing commas is not rewritten)"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise OpenCodeConfigError(f"Could not read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise OpenCodeConfigError(f"The root of {path} must be a JSON object")
    return raw


def _merge_config(raw: dict[str, Any], cfg: config.Config) -> dict[str, Any]:
    result = deepcopy(raw)
    try:
        profile = config.load_profile(cfg.profile) if cfg.profile else None
    except ValueError as exc:
        raise OpenCodeConfigError(f"Invalid BYOM profile: {exc}") from exc
    profile_aliases = profile_aliases_from_config()
    default_model = profile.opencode_model if profile else f"{PROVIDER_ID}/{cfg.served_model_name}"
    if profile and default_model.removeprefix(f"{PROVIDER_ID}/") not in profile_aliases:
        raise OpenCodeConfigError(
            f"Profile {profile.id!r} references an alias missing from the OpenCode template"
        )

    plugin_list = result.setdefault("plugin", [])
    if not isinstance(plugin_list, list):
        raise OpenCodeConfigError("OpenCode's `plugin` field must be a list")
    if not any(_is_discovery_plugin(entry) for entry in plugin_list):
        plugin_list.append(PLUGIN_SPEC)

    providers = result.setdefault("provider", {})
    if not isinstance(providers, dict):
        raise OpenCodeConfigError("OpenCode's `provider` field must be an object")
    provider = providers.setdefault(PROVIDER_ID, {})
    if not isinstance(provider, dict):
        raise OpenCodeConfigError("`provider.byom` must be an object")
    provider["npm"] = "@ai-sdk/openai-compatible"
    provider.setdefault("name", "BYOM Colab")

    options = provider.setdefault("options", {})
    if not isinstance(options, dict):
        raise OpenCodeConfigError("`provider.byom.options` must be an object")
    options["baseURL"] = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    discovery = options.setdefault("modelsDiscovery", {})
    if not isinstance(discovery, dict):
        raise OpenCodeConfigError("`provider.byom.options.modelsDiscovery` must be an object")
    discovery["enabled"] = True
    discovery["modelInfoFormat"] = "vllm"

    models = provider.setdefault("models", {})
    if not isinstance(models, dict):
        raise OpenCodeConfigError("`provider.byom.models` must be an object")
    for model_id, template in profile_aliases.items():
        _merge_model(models, model_id, template)

    base_model = models.setdefault(cfg.served_model_name, {})
    if not isinstance(base_model, dict):
        raise OpenCodeConfigError(
            f"`provider.byom.models.{cfg.served_model_name}` must be an object"
        )
    base_model.setdefault("name", "Qwen3.8 27B (BYOM)")
    base_model["tool_call"] = cfg.enable_auto_tool_choice
    base_limit = base_model.setdefault("limit", {})
    if not isinstance(base_limit, dict):
        raise OpenCodeConfigError(
            f"`provider.byom.models.{cfg.served_model_name}.limit` must be an object"
        )
    base_limit["context"] = cfg.max_model_len
    base_limit.setdefault("output", 8192)

    result["model"] = default_model
    return result


def profile_aliases_from_config() -> dict[str, dict[str, Any]]:
    """Build OpenCode model aliases from the canonical BYOM TOML profiles."""
    try:
        profiles = config.available_profiles()
        return {
            profile.id: {
                "id": profile.settings["served_model_name"],
                "name": profile.name,
                "limit": {
                    "context": profile.settings["max_model_len"],
                    "output": profile.opencode_output_limit,
                },
                "tool_call": profile.settings["enable_auto_tool_choice"],
            }
            for profile in profiles
        }
    except (KeyError, OSError, ValueError) as exc:
        raise OpenCodeConfigError("Could not generate OpenCode aliases from profiles") from exc


def _is_discovery_plugin(entry: Any) -> bool:
    spec = entry[0] if isinstance(entry, list) and entry else entry
    if not isinstance(spec, str):
        return False
    package = spec.rsplit("/", 1)[-1]
    return re.fullmatch(rf"{re.escape(PLUGIN_NAME)}(?:@[^/]+)?", package) is not None


def _merge_model(models: dict[str, Any], model_id: str, template: dict[str, Any]) -> None:
    model = models.setdefault(model_id, {})
    if not isinstance(model, dict):
        raise OpenCodeConfigError(f"`provider.byom.models.{model_id}` must be an object")
    for key in ("id", "name", "tool_call"):
        if key in template:
            model[key] = template[key]
    limit = model.setdefault("limit", {})
    if not isinstance(limit, dict):
        raise OpenCodeConfigError(f"`provider.byom.models.{model_id}.limit` must be an object")
    for key in ("context", "output"):
        if key in template.get("limit", {}):
            limit[key] = template["limit"][key]


def _atomic_write(path: Path, contents: str) -> None:
    try:
        config.atomic_write_text(path, contents)
    except OSError as exc:
        raise OpenCodeConfigError(f"Could not write {path}: {exc}") from exc
