"""Guarantee that `byom up` never touches configuration it does not own.

OpenCode: only `model`, the discovery plugin entry and `provider.byom.*` may change.
dsh: only `providers.byom` of `llm-pi-ai` and `provider`/`model` of
`agent-default-model` may change. Nothing may ever be removed.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path
from typing import Any

import pytest

from byom import config, dsh, opencode

PROFILES = ["qwen3.8-27b-32k", "qwen3.8-27b-256k"]


def _cfg(profile: str) -> config.Config:
    cfg = config.Config()
    config.apply_profile(cfg, profile)
    return cfg


# ----------------------------------------------------------------------------------
# OpenCode
# ----------------------------------------------------------------------------------
def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten dicts to {slash/path: leaf}; lists and scalars are leaves."""
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, child in value.items():
            out.update(_leaves(child, f"{prefix}/{key}" if prefix else key))
        return out
    return {prefix: value}


OPENCODE_ALLOWED = [
    r"model",
    r"provider/byom/npm",
    r"provider/byom/name",
    r"provider/byom/options/baseURL",
    r"provider/byom/options/modelsDiscovery/(enabled|modelInfoFormat)",
    r"provider/byom/models/[^/]+/(id|name|tool_call)",
    r"provider/byom/models/[^/]+/limit/(context|output)",
    # Empty containers created for the first time (e.g. `provider: {}`).
    r"provider",
    r"provider/byom",
    r"provider/byom/options",
    r"provider/byom/options/modelsDiscovery",
    r"provider/byom/models",
    r"provider/byom/models/[^/]+",
    r"provider/byom/models/[^/]+/limit",
]


def _assert_opencode_scope(before: dict[str, Any], after: dict[str, Any]) -> None:
    old, new = _leaves(before), _leaves(after)
    removed = sorted(set(old) - set(new))
    # An empty `{}` leaf may legitimately turn into a populated container.
    removed = [p for p in removed if old[p] != {} or p not in {k.rsplit("/", 1)[0] for k in new}]
    removed = [p for p in removed if not any(n.startswith(p + "/") for n in new)]
    assert not removed, f"OpenCode config entries were removed: {removed}"

    for path in sorted(set(new) - set(old) | {p for p in old if p in new and old[p] != new[p]}):
        if path == "plugin":
            assert new["plugin"][: len(old.get("plugin", []))] == old.get("plugin", []), (
                "existing plugins were reordered or changed"
            )
            continue
        assert any(re.fullmatch(rx, path) for rx in OPENCODE_ALLOWED), (
            f"unexpected OpenCode change at {path!r}: {old.get(path)!r} -> {new.get(path)!r}"
        )


def _opencode_fixture(with_byom: bool) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": "mimo/mimo-v2.5",
        "small_model": "mimo/small",
        "theme": "tokyonight",
        "autoupdate": False,
        "instructions": ["AGENTS.md", "docs/*.md"],
        "keybinds": {"leader": "ctrl+x"},
        "plugin": ["existing-plugin@1.0.0", ["other-plugin", {"opt": 1}]],
        "provider": {
            "mimo": {
                "name": "MiMo",
                "npm": "@ai-sdk/openai-compatible",
                "options": {"apiKey": "mimo-secret", "baseURL": "https://mimo.invalid/v1"},
                "models": {"mimo-v2.5": {"name": "MiMo", "limit": {"context": 1, "output": 2}}},
            },
            "anthropic": {"options": {"apiKey": "anthropic-secret"}},
        },
        "mcp": {"docs": {"type": "local", "command": ["docs-mcp"], "environment": {"T": "x"}}},
        "permission": {"bash": {"*": "ask", "git *": "allow"}},
        "agent": {"build": {"model": "mimo/mimo-v2.5", "tools": {"write": True}}},
        "unknown_future_key": {"nested": [1, 2, {"a": None}]},
    }
    if with_byom:
        cfg["provider"]["byom"] = {
            "name": "My own name",
            "options": {
                "apiKey": "byom-secret",
                "headers": {"X-Custom": "1"},
                "modelsDiscovery": {"timeoutMs": 6000, "extra": True},
            },
            "models": {
                "user-model": {"name": "Mine", "limit": {"context": 10}},
                "qwen3.8-27b": {"customOption": "keep", "limit": {"output": 4096}},
            },
            "customProviderKey": ["x"],
        }
    return cfg


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("with_byom", [False, True])
def test_opencode_only_touches_its_own_scope(tmp_path: Path, profile: str, with_byom: bool):
    path = tmp_path / "opencode.json"
    before = _opencode_fixture(with_byom)
    path.write_text(json.dumps(before, indent=2), encoding="utf-8")

    opencode.sync_config(_cfg(profile), path)
    after = json.loads(path.read_text(encoding="utf-8"))

    _assert_opencode_scope(before, after)
    # Secrets and user-owned entries are byte-for-byte equal.
    assert after["provider"]["mimo"] == before["provider"]["mimo"]
    assert after["provider"]["anthropic"] == before["provider"]["anthropic"]
    for key in ("mcp", "permission", "agent", "keybinds", "instructions", "unknown_future_key"):
        assert after[key] == before[key]
    if with_byom:
        byom_before, byom_after = before["provider"]["byom"], after["provider"]["byom"]
        assert byom_after["options"]["apiKey"] == "byom-secret"
        assert byom_after["options"]["headers"] == {"X-Custom": "1"}
        assert byom_after["options"]["modelsDiscovery"]["timeoutMs"] == 6000
        assert byom_after["options"]["modelsDiscovery"]["extra"] is True
        assert byom_after["customProviderKey"] == byom_before["customProviderKey"]
        assert byom_after["name"] == "My own name"
        assert byom_after["models"]["user-model"] == byom_before["models"]["user-model"]
        assert byom_after["models"]["qwen3.8-27b"]["customOption"] == "keep"


def test_opencode_scope_checker_detects_violations():
    """The checker itself must fail on out-of-scope edits and on removals."""
    before = {"model": "a", "theme": "dark", "provider": {"x": {"k": 1}}}
    with pytest.raises(AssertionError, match="unexpected"):
        _assert_opencode_scope(before, {**before, "theme": "light"})
    with pytest.raises(AssertionError, match="removed"):
        _assert_opencode_scope(before, {"model": "a", "provider": {"x": {"k": 1}}})
    with pytest.raises(AssertionError, match="unexpected"):
        _assert_opencode_scope(before, {**before, "provider": {"x": {"k": 2}}})


# ----------------------------------------------------------------------------------
# dsh
# ----------------------------------------------------------------------------------
DSH_FIXTURE = """\
# Your patch layer for this dsh profile
- id: ui-theme   # trailing comment
  name: "@deepseek-ai/dsh-client-ui-theme"
  config:
    preference: dark
    fontSize: 16

# provider settings
- id: "llm-pi-ai"
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      anthropic:
        apiKeyEnv: ANTHROPIC_SECRET_SENTINEL
      openai-codex: {}
      amd:
        apiKeyEnv: AMD_SECRET_SENTINEL
        api: openai-completions
        baseURL: https://amd.invalid/v1
        models:
          - id: Qwen3.8-27B
            name: Qwen3.8-27B
            contextWindow: 262144
      # keep this comment
      minimax:
        models:
          - id: MiniMax-M3
            name: MiniMax-M3
            input:
              - text
        apiKeyEnv: MINIMAX_SECRET_SENTINEL
@@BYOM@@      opencode:
        apiKeyEnv: OPENCODE_SECRET_SENTINEL
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: amd
    model: DeepSeek-V4.1-Flash
    reasoningEffort: high
- id: subagent-model-selection-settings
  name: "@deepseek-ai/dsh-tool-subagent/model-selection-settings"
  config:
    enabled: true
    allowedModels:
      - provider: anthropic
        model: claude-haiku-5-5
      - provider: amd
        model: DeepSeek-V4.1-Flash
- id: web
  name: "@deepseek-ai/dsh-web"
  config:
    port: !!js 1 + 1
"""

EXISTING_BYOM = """\
      byom:
        apiKeyEnv: MY_OWN_ENV
        headers:
          X-Custom: "1"
        api: stale
        baseURL: http://stale.invalid/v1
        models:
          - id: user-model
            name: Mine
            contextWindow: 10
          - id: qwen3.8-27b
            name: Stale
            contextWindow: 1
"""


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _strip_managed(text: str) -> list[str]:
    """Remove the only regions dsh sync may write, then return the remaining lines."""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    in_default_entry = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("- "):
            in_default_entry = bool(re.match(r"^- id:\s*[\"']?agent-default-model", line))
        if line == "      byom:":  # whole provider block belongs to byom
            i += 1
            while i < len(lines) and lines[i].strip() and _indent(lines[i]) > 6:
                i += 1
            continue
        if in_default_entry and re.match(r"^    (provider|model):", line):
            i += 1
            continue
        out.append(line)
        i += 1
    return out


def _fixture(with_byom: bool) -> str:
    return DSH_FIXTURE.replace("@@BYOM@@", EXISTING_BYOM if with_byom else "")


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("with_byom", [False, True])
def test_dsh_only_touches_its_own_scope(tmp_path: Path, profile: str, with_byom: bool):
    path = tmp_path / "cordis.patch.yml"
    before = _fixture(with_byom)
    path.write_text(before, encoding="utf-8")
    path.chmod(0o600)

    dsh.sync_config(_cfg(profile), path)
    after = path.read_text(encoding="utf-8")

    # Everything outside the managed regions is identical, line for line.
    assert _strip_managed(after) == _strip_managed(before)
    # No secret or user value disappeared.
    for sentinel in (
        "ANTHROPIC_SECRET_SENTINEL",
        "AMD_SECRET_SENTINEL",
        "MINIMAX_SECRET_SENTINEL",
        "OPENCODE_SECRET_SENTINEL",
        "# keep this comment",
        "# trailing comment",
        "!!js 1 + 1",
        "reasoningEffort: high",
    ):
        assert sentinel in after
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    if with_byom:
        assert "        apiKeyEnv: MY_OWN_ENV" in after
        assert '        headers:\n          X-Custom: "1"' in after
        user_model = (
            "          - id: user-model\n            name: Mine\n            contextWindow: 10"
        )
        assert user_model in after
        for stale in ("api: stale", "stale.invalid", "name: Stale", "contextWindow: 1\n"):
            assert stale not in after
        assert "BYOM_API_KEY" not in after
    else:
        assert "        apiKeyEnv: BYOM_API_KEY" in after

    # Re-running is a byte-for-byte no-op.
    assert dsh.sync_config(_cfg(profile), path).changed is False
    assert path.read_text(encoding="utf-8") == after


def test_dsh_scope_checker_detects_violations():
    """`_strip_managed` must expose edits outside the managed regions."""
    before = _fixture(False)
    assert _strip_managed(before.replace("fontSize: 16", "fontSize: 17")) != _strip_managed(before)
    assert _strip_managed(before.replace("AMD_SECRET_SENTINEL", "x")) != _strip_managed(before)
    assert _strip_managed(before.replace("reasoningEffort: high\n", "")) != _strip_managed(before)


def test_dsh_preserves_crlf_and_missing_final_newline(tmp_path: Path):
    path = tmp_path / "cordis.patch.yml"
    crlf = _fixture(False).replace("\n", "\r\n").rstrip("\r\n")
    path.write_bytes(crlf.encode())
    dsh.sync_config(_cfg("qwen3.8-27b-256k"), path)
    raw = path.read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b"")  # no bare LF introduced
    assert not raw.endswith(b"\n")  # no final newline added
    assert _strip_managed(raw.decode()) == _strip_managed(crlf)


def test_dsh_failure_leaves_file_untouched(tmp_path: Path):
    path = tmp_path / "cordis.patch.yml"
    bad = "key: value\n"
    path.write_text(bad)
    with pytest.raises(dsh.DshConfigError):
        dsh.sync_config(_cfg("qwen3.8-27b-256k"), path)
    assert path.read_text() == bad
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "home"]
    assert leftovers == ["cordis.patch.yml"]  # no temp files


def test_opencode_failure_leaves_file_untouched(tmp_path: Path):
    path = tmp_path / "opencode.json"
    bad = json.dumps({"provider": {"byom": "not-an-object"}, "model": "keep/me"})
    path.write_text(bad)
    with pytest.raises(opencode.OpenCodeConfigError):
        opencode.sync_config(_cfg("qwen3.8-27b-256k"), path)
    assert path.read_text() == bad
    assert [p.name for p in tmp_path.iterdir() if p.name != "home"] == ["opencode.json"]
