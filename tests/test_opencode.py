from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from byom import cli, config, opencode


@pytest.mark.parametrize(
    ("profile_id", "context", "model_id"),
    [
        ("qwen3.8-27b-32k", 32768, "byom/qwen3.8-27b-32k"),
        ("qwen3.8-27b-256k", 262144, "byom/qwen3.8-27b-256k"),
    ],
)
def test_sync_merges_active_profile_without_replacing_other_settings(
    tmp_path: Path, profile_id: str, context: int, model_id: str
):
    path = tmp_path / "opencode.json"
    initial = {
        "$schema": "https://opencode.ai/config.json",
        "model": "mimo/mimo-v2.5",
        "plugin": ["existing-plugin@1.0.0"],
        "provider": {
            "mimo": {
                "name": "MiMo",
                "options": {"apiKey": "mimo-secret-sentinel", "baseURL": "https://mimo.invalid/v1"},
            },
            "byom": {
                "name": "BYOM Colab",
                "options": {
                    "apiKey": "byom-secret-sentinel",
                    "baseURL": "http://127.0.0.1:4000/v1",
                    "modelsDiscovery": {"timeoutMs": 6000},
                },
                "models": {
                    "qwen3.8-27b": {
                        "name": "Existing BYOM name",
                        "limit": {"output": 4096},
                        "customOption": "preserve",
                    },
                    "user-model": {"name": "User model"},
                },
            },
        },
        "mcp": {"docs": {"type": "local", "command": ["docs-mcp"]}},
        "permission": {"bash": {"*": "ask"}},
    }
    path.write_text(json.dumps(initial, indent=2), encoding="utf-8")
    cfg = config.Config()
    config.apply_profile(cfg, profile_id)

    first = opencode.sync_config(cfg, path)
    written = path.read_text(encoding="utf-8")
    second = opencode.sync_config(cfg, path)
    result = json.loads(written)

    assert first.changed is True
    assert second.changed is False
    assert path.read_text(encoding="utf-8") == written
    assert result["model"] == model_id
    assert opencode.PLUGIN_SPEC in result["plugin"]
    assert result["provider"]["mimo"] == initial["provider"]["mimo"]
    assert result["mcp"] == initial["mcp"]
    assert result["permission"] == initial["permission"]

    byom = result["provider"]["byom"]
    assert byom["options"]["baseURL"] == "http://127.0.0.1:4000/v1"
    assert byom["options"]["modelsDiscovery"] == {
        "timeoutMs": 6000,
        "enabled": True,
        "modelInfoFormat": "vllm",
    }
    assert byom["options"]["apiKey"] == "byom-secret-sentinel"
    assert byom["models"]["qwen3.8-27b"]["limit"] == {
        "output": 4096,
        "context": context,
    }
    assert byom["models"]["qwen3.8-27b"]["customOption"] == "preserve"
    assert byom["models"]["user-model"] == {"name": "User model"}
    assert byom["models"]["qwen3.8-27b-32k"]["limit"]["context"] == 32768
    assert byom["models"]["qwen3.8-27b-256k"]["limit"]["context"] == 262144


def test_sync_resolves_xdg_and_explicit_config_paths(tmp_path: Path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    assert opencode.resolve_config_path() == xdg / "opencode" / "opencode.json"

    custom = tmp_path / "custom" / "opencode.json"
    monkeypatch.setenv("OPENCODE_CONFIG", str(custom))
    assert opencode.resolve_config_path() == custom
    assert opencode.resolve_config_path(tmp_path / "explicit.json") == tmp_path / "explicit.json"


def test_sync_keeps_existing_plugin_version_and_file_permissions(tmp_path: Path):
    path = tmp_path / "opencode.json"
    path.write_text(
        json.dumps({"plugin": [["opencode-models-discovery@1.5.5", {"setting": True}]]}),
        encoding="utf-8",
    )
    path.chmod(0o640)

    opencode.sync_config(config.Config(), path)

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["plugin"] == [["opencode-models-discovery@1.5.5", {"setting": True}]]
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_sync_creates_private_config_when_missing(tmp_path: Path):
    path = tmp_path / "new" / "opencode.json"
    result = opencode.sync_config(config.Config(), path)

    assert result.changed is True
    assert result.path == path
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["provider"]["byom"]["options"]["baseURL"] == "http://127.0.0.1:4000/v1"


def test_sync_rejects_jsonc_without_modifying_it(tmp_path: Path):
    path = tmp_path / "opencode.jsonc"
    original = '{\n  // keep this comment\n  "model": "mimo/model",\n}\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(opencode.OpenCodeConfigError, match="JSONC"):
        opencode.sync_config(config.Config(), path)

    assert path.read_text(encoding="utf-8") == original


def test_sync_rejects_non_utf8_config_without_modifying_it(tmp_path: Path):
    path = tmp_path / "opencode.json"
    original = b"\xff\xfe"
    path.write_bytes(original)

    with pytest.raises(opencode.OpenCodeConfigError, match="Could not read"):
        opencode.sync_config(config.Config(), path)

    assert path.read_bytes() == original


def test_sync_fails_when_inline_opencode_config_can_override_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"model":"mimo/model"}')
    path = tmp_path / "opencode.json"

    with pytest.raises(opencode.OpenCodeConfigError, match="OPENCODE_CONFIG_CONTENT"):
        opencode.sync_config(config.Config(), path)

    assert not path.exists()


def test_up_does_not_allocate_gpu_if_opencode_sync_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    cfg = config.Config()
    config.apply_profile(cfg, "qwen3.8-27b-256k")
    config.save_config(cfg)
    allocated = False

    def fail_sync(*args, **kwargs):
        raise opencode.OpenCodeConfigError("invalid OpenCode config")

    def track_allocation(*args, **kwargs):
        nonlocal allocated
        allocated = True

    monkeypatch.setattr(cli.opencode, "sync_config", fail_sync)
    monkeypatch.setattr(cli.colab, "new", track_allocation)
    # Nunca consultar o Colab real: uma sessão ativa na conta quebraria o teste.
    monkeypatch.setattr(cli.colab, "unregistered_sessions", lambda known=None: [])
    result = CliRunner().invoke(cli.app, ["up", "--opencode-config", str(tmp_path / "bad.json")])

    assert result.exit_code == 1
    assert "before allocating the GPU" in result.output
    assert allocated is False


# -- 5.7: `.json` tem precedência e `.jsonc` vira erro explicativo ------------
def test_json_tem_precedencia_sobre_jsonc(isolated_home):
    pasta = isolated_home / ".config" / "opencode"
    pasta.mkdir(parents=True)
    (pasta / "opencode.json").write_text('{"model": "antigo"}', encoding="utf-8")
    (pasta / "opencode.jsonc").write_text('{"model": "comentado"}', encoding="utf-8")

    assert opencode.resolve_config_path() == pasta / "opencode.json"


def test_apenas_jsonc_recusa_com_caminho_e_como_apontar_outro(isolated_home):
    pasta = isolated_home / ".config" / "opencode"
    pasta.mkdir(parents=True)
    jsonc = pasta / "opencode.jsonc"
    original = '{\n  // mantém\n  "model": "x"\n}\n'
    jsonc.write_text(original, encoding="utf-8")

    with pytest.raises(opencode.OpenCodeConfigError) as excinfo:
        opencode.resolve_config_path()

    mensagem = str(excinfo.value)
    assert str(jsonc) in mensagem
    assert "--opencode-config" in mensagem and "OPENCODE_CONFIG" in mensagem
    assert jsonc.read_text(encoding="utf-8") == original


def test_jsonc_explicito_tambem_e_recusado_sem_escrever(tmp_path):
    jsonc = tmp_path / "config.jsonc"
    original = '{\n  // mantém\n  "model": "x"\n}\n'
    jsonc.write_text(original, encoding="utf-8")

    with pytest.raises(opencode.OpenCodeConfigError, match="JSONC"):
        opencode.sync_config(config.Config(), jsonc)

    assert jsonc.read_text(encoding="utf-8") == original


def test_override_por_ambiente_jsonc_e_recusado(tmp_path, monkeypatch):
    jsonc = tmp_path / "custom.jsonc"
    jsonc.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENCODE_CONFIG", str(jsonc))

    with pytest.raises(opencode.OpenCodeConfigError, match="JSONC"):
        opencode.resolve_config_path()


def test_sem_nenhum_arquivo_o_alvo_e_json(tmp_path, monkeypatch, isolated_home):
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    assert opencode.resolve_config_path() == (
        isolated_home / ".config" / "opencode" / "opencode.json"
    )
