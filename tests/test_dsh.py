from __future__ import annotations

import stat
from pathlib import Path

import pytest

from byom import config, dsh

SAMPLE = """\
# Your patch layer
- id: ui-theme
  name: "@deepseek-ai/dsh-client-ui-theme"
  config:
    preference: dark
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      anthropic:
        apiKeyEnv: SECRET_ENV_SENTINEL
      amd:
        apiKeyEnv: AMD_KEY
        api: openai-completions
        baseURL: https://amd.invalid/v1
        models:
          - id: Qwen3.8-27B
            name: Qwen3.8-27B
            contextWindow: 262144
      minimax:
        models:
          - id: MiniMax-M3
            name: MiniMax-M3
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: amd
    model: DeepSeek-V4.1-Flash
- id: web
  name: "@deepseek-ai/dsh-web"
  config:
    port: 1
"""


def _cfg(profile: str = "qwen3.8-27b-256k") -> config.Config:
    cfg = config.Config()
    config.apply_profile(cfg, profile)
    return cfg


def _only_added(original: str, updated: str) -> None:
    """Every original line survives, in order (replaced default-model lines excepted)."""
    replaced = {"provider: amd", "model: DeepSeek-V4.1-Flash"}
    keep = [ln for ln in original.splitlines() if ln.strip() not in replaced]
    it = iter(updated.splitlines())
    for line in keep:
        assert any(line == other for other in it), f"lost line: {line!r}"


def test_adds_provider_and_default_without_touching_the_rest(tmp_path: Path):
    path = tmp_path / "cordis.patch.yml"
    path.write_text(SAMPLE)
    path.chmod(0o600)
    cfg = _cfg()
    result = dsh.sync_config(cfg, path)
    text = path.read_text()
    assert result.changed
    assert result.model == "byom/qwen3.8-27b"
    _only_added(SAMPLE, text)
    assert "      byom:" in text
    assert "        api: openai-completions" in text
    assert f"        baseURL: http://127.0.0.1:{cfg.proxy_port}/v1" in text
    assert '          - id: "qwen3.8-27b"' in text
    assert "            contextWindow: 262144" in text
    byom_block = text.split("      byom:")[1].split("- id: agent")[0]
    assert "        apiKeyEnv: BYOM_API_KEY" in byom_block
    assert '    provider: "byom"' in text
    assert '    model: "qwen3.8-27b"' in text
    assert "provider: amd" not in text
    assert "SECRET_ENV_SENTINEL" in text
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_is_idempotent_and_updates_context_on_profile_switch(tmp_path: Path):
    path = tmp_path / "cordis.patch.yml"
    path.write_text(SAMPLE)
    dsh.sync_config(_cfg(), path)
    assert dsh.sync_config(_cfg(), path).changed is False
    dsh.sync_config(_cfg("qwen3.8-27b-32k"), path)
    text = path.read_text()
    assert text.count("      byom:") == 1
    byom_block = text.split("      byom:")[1].split("      minimax:")[0]
    assert "contextWindow: 32768" in byom_block
    assert "262144" not in byom_block
    assert text.count('- id: "qwen3.8-27b"') == 1


def test_keeps_user_keys_and_other_models_in_byom_provider(tmp_path: Path):
    path = tmp_path / "cordis.patch.yml"
    path.write_text(
        SAMPLE.replace(
            "      minimax:",
            "      byom:\n        apiKeyEnv: MY_ENV\n        api: old\n        models:\n"
            "          - id: user-model\n            name: Mine\n"
            "          - id: qwen3.8-27b\n            name: Old\n      minimax:",
        )
    )
    dsh.sync_config(_cfg(), path)
    text = path.read_text()
    assert "        apiKeyEnv: MY_ENV" in text
    assert "BYOM_API_KEY" not in text
    assert "          - id: user-model\n            name: Mine" in text
    assert "api: old" not in text and "name: Old" not in text
    assert text.count("      byom:") == 1
    assert "      minimax:" in text


def test_creates_file_and_missing_entries(tmp_path: Path):
    path = tmp_path / "new" / "cordis.patch.yml"
    dsh.sync_config(_cfg(), path)
    text = path.read_text()
    assert "- id: llm-pi-ai" in text and "- id: agent-default-model" in text
    assert text.index("- id: llm-pi-ai") < text.index("      byom:")


def test_adds_providers_key_when_missing(tmp_path: Path):
    path = tmp_path / "cordis.patch.yml"
    path.write_text("- id: llm-pi-ai\n  name: x\n  config:\n    other: 1\n- id: web\n  name: w\n")
    dsh.sync_config(_cfg(), path)
    text = path.read_text()
    assert "    other: 1\n    providers:\n      byom:" in text
    assert "- id: web\n  name: w" in text


def test_rejects_non_list_and_tabs(tmp_path: Path):
    path = tmp_path / "cordis.patch.yml"
    path.write_text("key: value\n")
    with pytest.raises(dsh.DshConfigError):
        dsh.sync_config(_cfg(), path)
    path.write_text("- id: x\n\tname: y\n")
    with pytest.raises(dsh.DshConfigError):
        dsh.sync_config(_cfg(), path)


def test_default_path_and_is_installed(isolated_home: Path):
    assert dsh.is_installed() is False
    target = isolated_home / ".dsh" / "profiles" / "desktop"
    target.mkdir(parents=True)
    (target / "cordis.patch.yml").write_text("[]\n")
    assert dsh.is_installed() is True


NONCANONICAL_INDENT = """\
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      byom:
          apiKeyEnv: MY_ENV
          headers:
            X-Custom: "1"
          models:
            - id: user-model
              name: Mine
              contextWindow: 10
"""


def test_rejects_noncanonical_byom_indent_without_writing(tmp_path: Path):
    """Exposes bug: byom children are only recognized at exactly 8 spaces.

    A differently indented (but valid YAML) `byom:` block makes
    `_merge_provider` see zero children, silently dropping the user's
    `apiKeyEnv`, `headers`, and custom models when the block is rebuilt.
    Sync must fail safely instead of writing a lossy file.
    """
    path = tmp_path / "cordis.patch.yml"
    path.write_text(NONCANONICAL_INDENT)
    with pytest.raises(dsh.DshConfigError):
        dsh.sync_config(_cfg(), path)
    assert path.read_text() == NONCANONICAL_INDENT


SAME_INDENT_MODELS = """\
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      byom:
        apiKeyEnv: MY_ENV
        models:
        - id: user-model
          name: Mine
          contextWindow: 10
"""


def test_rejects_same_indent_model_items_without_writing(tmp_path: Path):
    """Exposes bug: `models:` items at the same indent as the key are valid
    YAML, but `_merge_provider` re-nests them as direct children of `byom:`,
    mixing a sequence into a mapping and writing a file dsh cannot load.
    Sync must fail safely instead of writing a corrupted file.
    """
    path = tmp_path / "cordis.patch.yml"
    path.write_text(SAME_INDENT_MODELS)
    with pytest.raises(dsh.DshConfigError):
        dsh.sync_config(_cfg(), path)
    assert path.read_text() == SAME_INDENT_MODELS
