"""`up` guarda e `down` devolve o modelo padrão que OpenCode/dsh tinham."""

from __future__ import annotations

import json
import stat
from pathlib import Path

from typer.testing import CliRunner

from byom import cli, colab, config, harness_state, opencode

runner = CliRunner()

OPENCODE_ANTES = {"$schema": "https://opencode.ai/config.json", "model": "antigo/modelo"}
DSH_ANTES = """\
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      anthropic:
        apiKeyEnv: SECRET_ENV
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: anthropic
    model: claude-x
"""


def _cfg() -> config.Config:
    cfg = config.Config()
    config.apply_profile(cfg, "qwen3.8-27b-32k")
    return cfg


def _harnesses(tmp_path: Path) -> tuple[Path, Path]:
    opencode_path = tmp_path / "opencode" / "opencode.json"
    opencode_path.parent.mkdir(parents=True, exist_ok=True)
    opencode_path.write_text(json.dumps(OPENCODE_ANTES), encoding="utf-8")
    dsh_path = tmp_path / "dsh" / "cordis.patch.yml"
    dsh_path.parent.mkdir(parents=True, exist_ok=True)
    dsh_path.write_text(DSH_ANTES, encoding="utf-8")
    return opencode_path, dsh_path


def test_up_guarda_e_down_restaura_exatamente_o_valor_anterior(tmp_path):
    cfg = _cfg()
    opencode_path, dsh_path = _harnesses(tmp_path)

    cli._sync_external_configs(cfg, opencode_path, dsh_path)
    depois_do_up = json.loads(opencode_path.read_text(encoding="utf-8"))
    assert depois_do_up["model"] == "byom/qwen3.8-27b-32k"
    assert 'provider: "byom"' in dsh_path.read_text(encoding="utf-8")

    registro = harness_state.state_path()
    assert registro.exists()
    assert stat.S_IMODE(registro.stat().st_mode) == 0o600

    restaurados, falha = harness_state.restore()

    assert falha == ""
    assert set(restaurados) == {"OpenCode", "dsh"}
    restaurado = json.loads(opencode_path.read_text(encoding="utf-8"))
    # O ponteiro padrão volta ao valor anterior; o provedor `byom` e o plugin
    # adicionados pelo `up` ficam (apontam para um proxy parado, sem efeito).
    assert restaurado["model"] == "antigo/modelo"
    assert restaurado["$schema"] == OPENCODE_ANTES["$schema"]
    assert "byom" in restaurado["provider"]
    final = dsh_path.read_text(encoding="utf-8")
    assert "provider: anthropic" in final
    assert "model: claude-x" in final
    assert "byom" not in final.split("agent-default-model")[1]
    assert not registro.exists()


def test_sem_registro_o_down_avisa_e_nao_altera_nada(tmp_path, monkeypatch):
    opencode_path, dsh_path = _harnesses(tmp_path)
    antes_opencode = opencode_path.read_text(encoding="utf-8")
    antes_dsh = dsh_path.read_text(encoding="utf-8")
    monkeypatch.setattr(opencode, "resolve_config_path", lambda explicit=None: opencode_path)

    restaurados, falha = harness_state.restore()

    assert (restaurados, falha) == ([], "")
    assert opencode_path.read_text(encoding="utf-8") == antes_opencode
    assert dsh_path.read_text(encoding="utf-8") == antes_dsh


def test_down_da_cli_restaura_quando_a_sessao_e_encerrada(tmp_path, monkeypatch):
    opencode_path, dsh_path = _harnesses(tmp_path)
    cfg = _cfg()
    config.save_config(cfg)
    config.save_state(
        config.State(session="byom-x", endpoint="m-s-x", nonce="n" * 32, created_at="t")
    )
    cli._sync_external_configs(cfg, opencode_path, dsh_path)
    monkeypatch.setattr(colab, "stop", lambda name: None)
    monkeypatch.setattr(colab, "is_allocated", lambda name, endpoint="": False)
    monkeypatch.setattr(cli.remote, "Remote", lambda **kwargs: type(
        "R", (), {"stop": lambda self: None}
    )())

    resultado = runner.invoke(cli.app, ["down"])

    assert resultado.exit_code == 0, resultado.output
    assert "Harness config restored: OpenCode, dsh" in resultado.output
    assert json.loads(opencode_path.read_text(encoding="utf-8"))["model"] == "antigo/modelo"
    assert "provider: anthropic" in dsh_path.read_text(encoding="utf-8")


def test_status_avisa_quando_nao_ha_registro(tmp_path, monkeypatch):
    cfg = _cfg()
    config.save_config(cfg)
    monkeypatch.setattr(colab, "orphan_assignments", list)

    resultado = runner.invoke(cli.app, ["status"])

    assert resultado.exit_code == 0
    assert "no harness record to restore" in resultado.output


def test_restore_e_idempotente_e_nao_quebra_arquivo_com_jsonc(tmp_path):
    """Registro apontando para um .jsonc não derruba o `down`."""
    jsonc = tmp_path / "opencode.jsonc"
    jsonc.write_text('{\n  // comentário\n  "model": "x"\n}\n', encoding="utf-8")
    original = jsonc.read_text(encoding="utf-8")
    harness_state.save({"opencode": {"path": str(jsonc), "model": "antigo/modelo"}})

    restaurados, falha = harness_state.restore()

    assert restaurados == []
    assert "JSONC" in falha
    assert jsonc.read_text(encoding="utf-8") == original
    # Falhou: o registro fica para uma segunda tentativa.
    assert harness_state.state_path().exists()


def test_capture_ignora_arquivos_ausentes(tmp_path):
    assert harness_state.capture(tmp_path / "nao_existe.json", tmp_path / "nao.yml") == {}


def test_restore_remove_chave_que_nao_existia(tmp_path):
    opencode_path = tmp_path / "opencode.json"
    opencode_path.write_text(json.dumps({"plugin": []}), encoding="utf-8")
    harness_state.save({"opencode": {"path": str(opencode_path), "model": None}})

    restaurados, _ = harness_state.restore()

    assert restaurados == ["OpenCode"]
    assert "model" not in json.loads(opencode_path.read_text(encoding="utf-8"))
