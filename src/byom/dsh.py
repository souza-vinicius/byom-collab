"""Merge BYOM settings into the dsh (DeepSeek harness) profile patch.

The file (``~/.dsh/profiles/desktop/cordis.patch.yml``) is edited as text so comments,
secrets and unrelated entries survive. Only two things are written:

* ``providers.byom`` inside the ``llm-pi-ai`` entry (``apiKeyEnv`` when absent, ``api``,
  ``baseURL`` and the BYOM model; other keys and other models of that provider are kept);
* ``provider``/``model`` of the ``agent-default-model`` entry.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import config

PROVIDER_ID = "byom"
API_KEY_ENV = "BYOM_API_KEY"  # any value works: the local proxy does not check it
LLM_ENTRY = ("llm-pi-ai", "@deepseek-ai/dsh-llm-pi-ai")
DEFAULT_MODEL_ENTRY = ("agent-default-model", "@deepseek-ai/dsh-agent-default-model")
_MANAGED_KEYS = ("api", "baseURL", "models")


class DshConfigError(Exception):
    """Safe error raised while locating, reading, or updating the dsh config."""


@dataclass(frozen=True)
class SyncResult:
    path: Path
    model: str
    changed: bool


def default_config_path() -> Path:
    return Path.home() / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml"


def resolve_config_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser()
    return default_config_path()


def is_installed() -> bool:
    return default_config_path().is_file()


def sync_config(cfg: config.Config, explicit_path: Path | None = None) -> SyncResult:
    """Register the BYOM provider/model in dsh and make it the default model."""
    path = resolve_config_path(explicit_path)
    original = _read(path)
    updated = _merge(original, cfg)
    changed = updated != original
    if changed:
        _atomic_write(path, updated)
    return SyncResult(path=path, model=f"{PROVIDER_ID}/{cfg.served_model_name}", changed=changed)


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    if not path.is_file():
        raise DshConfigError(f"dsh config path is not a file: {path}")
    try:
        text = path.read_bytes().decode("utf-8")  # no newline translation
    except (OSError, UnicodeError) as exc:
        raise DshConfigError(f"Could not read {path}: {exc}") from exc
    if re.search(r"^\t", text, re.MULTILINE):
        raise DshConfigError(f"{path} uses tab indentation, which this tool does not rewrite")
    stripped = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if stripped and not stripped[0].startswith("- "):
        raise DshConfigError(f"{path} must be a top-level YAML list of patch entries")
    return text


# --- tiny line-oriented helpers -------------------------------------------------------
def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_content(line: str) -> bool:
    return bool(line.strip()) and not line.lstrip().startswith("#")


def _block_end(lines: list[str], start: int, limit: int, indent: int) -> int:
    """End (exclusive) of the block opened at ``start`` whose children are > ``indent``."""
    end = start + 1
    for j in range(start + 1, limit):
        if not _is_content(lines[j]):
            continue
        if _indent(lines[j]) <= indent:
            break
        end = j + 1
    return end


def _find_key(lines: list[str], start: int, end: int, indent: int, key: str) -> int | None:
    pattern = re.compile(rf"^ {{{indent}}}{re.escape(key)}:(\s|$)")
    for i in range(start, end):
        if pattern.match(lines[i]):
            return i
    return None


def _find_entry(lines: list[str], entry_id: str) -> tuple[int, int] | None:
    pattern = re.compile(rf"""^-\s+id:\s*["']?{re.escape(entry_id)}["']?\s*(#.*)?$""")
    for i, line in enumerate(lines):
        if pattern.match(line):
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("- "):
                    end = j
                    break
            while end > i + 1 and not _is_content(lines[end - 1]):
                end -= 1
            return i, end
    return None


def _ensure_entry(lines: list[str], entry: tuple[str, str]) -> tuple[int, int]:
    found = _find_entry(lines, entry[0])
    if found:
        return found
    if lines and lines[-1].strip():
        lines.append("")
    lines += [f"- id: {entry[0]}", f'  name: "{entry[1]}"', "  config:"]
    return _find_entry(lines, entry[0])  # type: ignore[return-value]


def _ensure_key(
    lines: list[str], entry: tuple[str, str], parent: tuple[str, ...]
) -> tuple[int, int]:
    """Locate (creating when missing) the mapping at ``parent`` below the entry.

    ``parent`` is the key chain under the entry (e.g. ("config", "providers")); returns
    the (line, end) of the last key. Keys are expected at indent 2, 4, ... in order.
    """
    s, e = _ensure_entry(lines, entry)
    line = s
    end = e
    for depth, key in enumerate(parent):
        ind = 2 + 2 * depth
        idx = _find_key(lines, line + 1, end, ind, key)
        if idx is None:
            idx = end
            lines.insert(idx, f"{' ' * ind}{key}:")
            end += 1
        else:
            text = lines[idx]
            value = text.split(":", 1)[1].split("#", 1)[0].strip()
            if value == "{}":
                lines[idx] = f"{' ' * ind}{key}:"
            elif value:
                raise DshConfigError(f"`{key}` in {entry[0]} must be a block mapping")
        line = idx
        end = _block_end(lines, idx, end, ind)
    return line, end


def _scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _merge(text: str, cfg: config.Config) -> str:
    lines = text.splitlines()
    try:
        profile = config.load_profile(cfg.profile) if cfg.profile else None
    except ValueError as exc:
        raise DshConfigError(f"Invalid BYOM profile: {exc}") from exc
    model_id = cfg.served_model_name
    model_name = profile.name if profile else "Qwen3.8 27B (BYOM)"

    _merge_provider(lines, cfg, model_id, model_name)
    _set_default_model(lines, model_id)
    # Keep the file's own line endings and final-newline convention.
    newline = "\r\n" if "\r\n" in text else "\n"
    ends_with_newline = text.endswith("\n") or not text
    return newline.join(lines) + (newline if ends_with_newline else "")


def _merge_provider(lines: list[str], cfg: config.Config, model_id: str, model_name: str) -> None:
    prov_line, prov_end = _ensure_key(lines, LLM_ENTRY, ("config", "providers"))
    byom = _find_key(lines, prov_line + 1, prov_end, 6, PROVIDER_ID)
    preserved_children: list[str] = []
    other_models: list[str] = []
    if byom is not None:
        value = lines[byom].split(":", 1)[1].split("#", 1)[0].strip()
        if value and value != "{}":
            raise DshConfigError("`providers.byom` must be a block mapping")
        byom_end = _block_end(lines, byom, prov_end, 6)
        # Fail safely on layouts `byom up` does not write: children at another
        # indent would be dropped, and `- ` items at child level would be
        # re-nested into the mapping, corrupting the file.
        child_indent = next(
            (_indent(lines[j]) for j in range(byom + 1, byom_end) if _is_content(lines[j])), 8
        )
        if child_indent != 8 or any(
            _indent(lines[j]) == 8 and lines[j].lstrip().startswith("- ")
            for j in range(byom + 1, byom_end)
            if _is_content(lines[j])
        ):
            raise DshConfigError(
                "`providers.byom` does not match the layout written by `byom up` "
                "(child keys at 8 spaces, list items deeper); fix or delete the block"
            )
        children = [
            i for i in range(byom + 1, byom_end) if _is_content(lines[i]) and _indent(lines[i]) == 8
        ]
        for n, i in enumerate(children):
            stop = children[n + 1] if n + 1 < len(children) else byom_end
            key = lines[i].strip().split(":", 1)[0]
            chunk = lines[i:stop]
            if key == "models":
                other_models = _other_models(chunk, model_id)
            elif key not in _MANAGED_KEYS:
                preserved_children += chunk
        del lines[byom:byom_end]
        insert_at = byom
    else:
        insert_at = prov_end

    if not any(ln.strip().startswith("apiKeyEnv:") for ln in preserved_children):
        preserved_children.insert(0, f"        apiKeyEnv: {API_KEY_ENV}")
    # User-owned keys first, so re-running is a no-op regardless of who added them.
    block = [
        f"      {PROVIDER_ID}:",
        *preserved_children,
        "        api: openai-completions",
        f"        baseURL: http://127.0.0.1:{cfg.proxy_port}/v1",
        "        models:",
        f"          - id: {_scalar(model_id)}",
        f"            name: {_scalar(model_name)}",
        f"            contextWindow: {cfg.max_model_len}",
        *other_models,
    ]
    lines[insert_at:insert_at] = block


def _other_models(chunk: list[str], model_id: str) -> list[str]:
    """Model list items of ``chunk`` (a `models:` block) that are not ours."""
    first = chunk[0].split(":", 1)[1].split("#", 1)[0].strip()
    if first and first != "[]":
        raise DshConfigError("`providers.byom.models` must be a block list")
    # The first `- ` line in the chunk is necessarily a top-level item; items at
    # another indent would be silently dropped by the `^ {10}- ` scan below.
    first_item = next((i for i in range(1, len(chunk)) if re.match(r"^ +- ", chunk[i])), None)
    if first_item is not None and _indent(chunk[first_item]) != 10:
        raise DshConfigError(
            "`providers.byom.models` does not match the layout written by `byom up` "
            "(items at 10 spaces); fix or delete the list"
        )
    starts = [i for i, ln in enumerate(chunk) if re.match(r"^ {10}- ", ln)]
    kept: list[str] = []
    for n, i in enumerate(starts):
        stop = starts[n + 1] if n + 1 < len(starts) else len(chunk)
        item = chunk[i:stop]
        match = re.match(r"""^ {10}- id:\s*["']?([^"'\s#]+)["']?""", item[0])
        if match and match.group(1) == model_id:
            continue
        kept += item
    return kept


def _set_default_model(lines: list[str], model_id: str) -> None:
    cfg_line, cfg_end = _ensure_key(lines, DEFAULT_MODEL_ENTRY, ("config",))
    for key, value in (("provider", PROVIDER_ID), ("model", model_id)):
        idx = _find_key(lines, cfg_line + 1, cfg_end, 4, key)
        new = f"    {key}: {_scalar(value)}"
        if idx is None:
            lines.insert(cfg_end, new)
            cfg_end += 1
        else:
            lines[idx] = new


def _atomic_write(path: Path, contents: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path.resolve()
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        fd, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
                os.fchmod(output.fileno(), mode)
                output.write(contents)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
    except OSError as exc:
        raise DshConfigError(f"Could not write {path}: {exc}") from exc
