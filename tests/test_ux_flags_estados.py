"""Flags da CLI e estados da sessão Colab, exercitados apenas com mocks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from byom import cli, colab, config, llm_chat

runner = CliRunner()


def test_setup_model_id_persiste_no_prefs_e_no_yaml(tmp_path, monkeypatch):
    byom_dir = tmp_path / "setup-model-id"
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))

    res = runner.invoke(
        cli.app,
        ["setup", "--url", "http://127.0.0.1:4000/v1", "--model-id", "meu-modelo"],
    )

    assert res.exit_code == 0
    prefs = json.loads((byom_dir / llm_chat.PREFS_FILENAME).read_text("utf-8"))
    assert prefs["model_id"] == "meu-modelo"
    assert (byom_dir / llm_chat.DEFAULT_MODEL_FILENAME).read_text("utf-8").strip() == "meu-modelo"


def test_chat_flag_model_chega_no_argv_do_llm(tmp_path, monkeypatch):
    byom_dir = tmp_path / "chat-model-flag"
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))
    assert runner.invoke(cli.app, ["setup"]).exit_code == 0
    chamadas: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        chamadas.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(llm_chat.subprocess, "run", fake_run)

    res = runner.invoke(cli.app, ["chat", "-m", "outro"])

    assert res.exit_code == 0
    assert chamadas == [[sys.executable, "-m", "llm", "chat", "-m", "outro"]]


def test_up_encaminha_flags_de_config_externa(tmp_path, monkeypatch):
    config.save_config(config.Config())
    recebido: list[tuple[bool, Path | None, Path | None]] = []
    monkeypatch.setattr(
        cli,
        "_run_up",
        lambda cfg, force, opencode_config, dsh_config: recebido.append(
            (force, opencode_config, dsh_config)
        ),
    )

    res = runner.invoke(
        cli.app,
        [
            "up",
            "--force",
            "--opencode-config",
            str(tmp_path / "opencode.json"),
            "--dsh-config",
            str(tmp_path / "dsh.yml"),
        ],
    )

    assert res.exit_code == 0
    assert recebido == [
        (True, tmp_path / "opencode.json", tmp_path / "dsh.yml")
    ]


def test_wait_ready_estado_ready_retorna_payload(monkeypatch):
    payload = {"worker": {"state": "ready"}}
    monkeypatch.setattr(cli, "_daemon_status", lambda cfg, expected_session=None: payload)
    assert cli._wait_ready(config.Config(), 0.01) is payload


def test_wait_ready_daemon_desaparecido_diz_o_que_fazer(monkeypatch):
    monkeypatch.setattr(cli, "_pid_is_byom", lambda *args: False)
    state = config.State(
        session="byom-x",
        endpoint="",
        nonce="n" * 32,
        created_at="now",
        daemon_pid=4242,
        daemon_start_ticks=1,
    )

    with pytest.raises(TimeoutError, match="não existe mais"):
        cli._wait_ready(config.Config(), 0.01, state=state)


def test_wait_ready_desconexao_e_cota_sao_acionaveis(monkeypatch):
    monkeypatch.setattr(cli, "READY_POLL_S", 0)

    monkeypatch.setattr(cli, "_daemon_status", lambda cfg, expected_session=None: None)
    with pytest.raises(TimeoutError, match="parou de responder"):
        cli._wait_ready(config.Config(), 0.01)

    monkeypatch.setattr(
        cli,
        "_daemon_status",
        lambda cfg, expected_session=None: {
            "worker": {"state": "unavailable", "error": "GPU quota exceeded"},
            "tunnel": {"state": "failed", "last_error": "tunnel closed"},
        },
    )
    with pytest.raises(TimeoutError, match="GPU quota exceeded"):
        cli._wait_ready(config.Config(), 0.01)


def test_orphan_line_ensina_a_parar_e_nomeia_sem_registro():
    line = colab.SessionLine(
        name="byom-1234abcd",
        endpoint="m-s-abc",
        hardware="L4",
        shape="standard",
        variant="GPU",
    )
    descricao = cli._orphan_line(line)
    assert "byom-1234abcd" in descricao
    assert "colab stop -s byom-1234abcd" in descricao

    sem_nome = colab.SessionLine(
        name=None, endpoint="m-s-xyz", hardware="T4", shape="standard", variant="GPU"
    )
    assert "[?]" in cli._orphan_line(sem_nome)
