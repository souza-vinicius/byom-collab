"""Testes da integração `llm chat` (`setup`, `chat`, `uninstall` e `byom_home()`)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from byom import cli, config, llm_chat

runner = CliRunner()


def test_byom_home_precedencia_e_plataformas(tmp_path, monkeypatch):
    monkeypatch.delenv("BYOM_HOME", raising=False)
    monkeypatch.delenv("BYOM_CONFIG_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)

    fake_home = tmp_path / "usuario"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    assert config.byom_home(platform="linux") == fake_home / ".config" / "byom"
    assert config.byom_home(platform="darwin") == (
        fake_home / "Library" / "Application Support" / "byom"
    )
    assert config.byom_home(platform="win32") == fake_home / "AppData" / "Roaming" / "byom"

    xdg_dir = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
    assert config.byom_home(platform="linux") == xdg_dir / "byom"

    appdata_dir = tmp_path / "appdata"
    monkeypatch.setenv("APPDATA", str(appdata_dir))
    assert config.byom_home(platform="win32") == appdata_dir / "byom"

    cfg_dir = tmp_path / "legacy-cfg"
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(cfg_dir))
    assert config.byom_home(platform="linux") == cfg_dir

    custom_home = tmp_path / "custom-byom-home"
    monkeypatch.setenv("BYOM_HOME", str(custom_home))
    assert config.byom_home(platform="linux") == custom_home
    assert config.config_dir() == custom_home

    monkeypatch.setenv("BYOM_HOME", "   ")
    with pytest.raises(config.ConfigError, match="BYOM_HOME"):
        config.byom_home()


def test_cli_help_exibe_setup_chat_e_uninstall():
    res = runner.invoke(cli.app, ["--help"])
    assert res.exit_code == 0
    for command in ("setup", "chat", "uninstall"):
        assert command in res.output


def test_setup_cria_estrutura_yaml_permissoes_e_eh_idempotente(tmp_path, monkeypatch):
    byom_dir = tmp_path / "isolated-byom"
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))

    res1 = runner.invoke(
        cli.app,
        [
            "setup",
            "--url",
            "https://tunel-1.trycloudflare.com/v1",
            "--model",
            "qwen3.8-27b",
            "--api-key",
            "segredo-local",
        ],
    )
    assert res1.exit_code == 0
    assert str(byom_dir) in res1.output

    expected_files = [
        byom_dir / llm_chat.MARKER_FILENAME,
        byom_dir / llm_chat.PREFS_FILENAME,
        byom_dir / llm_chat.EXTRA_MODELS_FILENAME,
        byom_dir / llm_chat.DEFAULT_MODEL_FILENAME,
        byom_dir / llm_chat.KEYS_FILENAME,
    ]
    for path in expected_files:
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    parsed_yaml = yaml.safe_load((byom_dir / llm_chat.EXTRA_MODELS_FILENAME).read_text("utf-8"))
    assert parsed_yaml == [
        {
            "model_id": "byom",
            "model_name": "qwen3.8-27b",
            "api_base": "https://tunel-1.trycloudflare.com/v1",
            "api_key_name": "byom",
        }
    ]

    res2 = runner.invoke(
        cli.app,
        ["setup", "--url", "https://tunel-2.trycloudflare.com/v1"],
    )
    assert res2.exit_code == 0

    parsed_yaml2 = yaml.safe_load((byom_dir / llm_chat.EXTRA_MODELS_FILENAME).read_text("utf-8"))
    assert parsed_yaml2 == [
        {
            "model_id": "byom",
            "model_name": "qwen3.8-27b",
            "api_base": "https://tunel-2.trycloudflare.com/v1",
            "api_key_name": "byom",
        }
    ]
    prefs = json.loads((byom_dir / llm_chat.PREFS_FILENAME).read_text("utf-8"))
    assert prefs["api_base"] == "https://tunel-2.trycloudflare.com/v1"
    assert prefs["model_name"] == "qwen3.8-27b"


def test_chat_falha_sem_setup_previo(tmp_path, monkeypatch):
    byom_dir = tmp_path / "sem-setup"
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))

    res = runner.invoke(cli.app, ["chat"])
    assert res.exit_code != 0
    assert "byom setup" in res.output


def test_chat_executa_subprocesso_isolado_com_system_e_no_history(tmp_path, monkeypatch):
    byom_dir = tmp_path / "chat-byom"
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))
    monkeypatch.delenv("LLM_USER_PATH", raising=False)

    assert runner.invoke(cli.app, ["setup"]).exit_code == 0

    chamadas: list[dict[str, object]] = []

    def fake_run(cmd, env=None, check=False, stderr=None, text=False, encoding=None, errors=None):
        logs_off_during = (byom_dir / llm_chat.LOGS_OFF_FILENAME).exists()
        chamadas.append(
            {
                "cmd": cmd,
                "llm_user_path": env.get("LLM_USER_PATH") if env else None,
                "logs_off_during": logs_off_during,
                "check": check,
                "stderr": stderr,
                "text": text,
                "encoding": encoding,
                "errors": errors,
            }
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(llm_chat.subprocess, "run", fake_run)

    res_normal = runner.invoke(cli.app, ["chat", "--system", "Responda em PT-BR", "--continue"])
    assert res_normal.exit_code == 0
    assert os.environ.get("LLM_USER_PATH") is None
    assert len(chamadas) == 1
    assert chamadas[0]["cmd"] == [
        sys.executable,
        "-m",
        "llm",
        "chat",
        "-m",
        "byom",
        "--system",
        "Responda em PT-BR",
        "--continue",
    ]
    assert chamadas[0]["llm_user_path"] == str(byom_dir)
    assert chamadas[0]["logs_off_during"] is False

    res_no_hist = runner.invoke(cli.app, ["chat", "--no-history"])
    assert res_no_hist.exit_code == 0
    assert len(chamadas) == 2
    assert chamadas[1]["cmd"] == [sys.executable, "-m", "llm", "chat", "-m", "byom"]
    assert chamadas[1]["llm_user_path"] == str(byom_dir)
    assert chamadas[1]["logs_off_during"] is True
    assert not (byom_dir / llm_chat.LOGS_OFF_FILENAME).exists()
    assert chamadas[0]["stderr"] is subprocess.PIPE
    assert chamadas[0]["text"] is True
    assert chamadas[0]["encoding"] == "utf-8"
    assert chamadas[0]["errors"] == "replace"


def test_chat_abortado_por_ctrl_c_ou_ctrl_d_encerra_sem_erro(tmp_path, monkeypatch):
    byom_dir = tmp_path / "chat-abortado"
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))
    assert runner.invoke(cli.app, ["setup"]).exit_code == 0

    monkeypatch.setattr(
        llm_chat.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="Aborted!\n"),
    )

    res = runner.invoke(cli.app, ["chat", "--no-history"])
    assert res.exit_code == 0
    assert "Aborted!" not in res.output
    assert "Error" not in res.output


def test_chat_com_falha_real_mostra_stderr_e_sai_com_o_codigo(tmp_path, monkeypatch):
    byom_dir = tmp_path / "chat-falha"
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))
    assert runner.invoke(cli.app, ["setup"]).exit_code == 0

    monkeypatch.setattr(
        llm_chat.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="Traceback: boom\n"),
    )

    res = runner.invoke(cli.app, ["chat", "--no-history"])
    assert res.exit_code == 1
    assert "Traceback: boom" in res.output


def test_uninstall_confirmacao_yes_e_protecao_de_caminhos(tmp_path, monkeypatch):
    byom_dir = tmp_path / "uninstall-byom"
    outside_file = tmp_path / "arquivo-externo.txt"
    outside_file.write_text("preservado", encoding="utf-8")

    monkeypatch.setenv("BYOM_HOME", str(byom_dir))
    assert runner.invoke(cli.app, ["setup"]).exit_code == 0
    (byom_dir / "logs.db").write_text("sqlite-placeholder", encoding="utf-8")

    res_cancel = runner.invoke(cli.app, ["uninstall"], input="n\n")
    assert res_cancel.exit_code == 0
    assert str(byom_dir) in res_cancel.output
    assert "logs.db" in res_cancel.output
    assert byom_dir.exists()

    res_yes = runner.invoke(cli.app, ["uninstall", "--yes"])
    assert res_yes.exit_code == 0
    assert not byom_dir.exists()
    assert "pip uninstall byom" in res_yes.output
    assert outside_file.read_text(encoding="utf-8") == "preservado"


def test_uninstall_recusa_sem_marcador_e_caminhos_perigosos(tmp_path, monkeypatch):
    sem_marcador = tmp_path / "sem-marcador"
    sem_marcador.mkdir()
    (sem_marcador / "qualquer.txt").write_text("nao apagar", encoding="utf-8")
    monkeypatch.setenv("BYOM_HOME", str(sem_marcador))
    res_sem_marcador = runner.invoke(cli.app, ["uninstall", "--yes"])
    assert res_sem_marcador.exit_code != 0
    assert sem_marcador.exists()

    alvo = tmp_path / "alvo"
    alvo.mkdir()
    (alvo / llm_chat.MARKER_FILENAME).write_text("ok", encoding="utf-8")
    link = tmp_path / "symlink-byom"
    link.symlink_to(alvo)
    monkeypatch.setenv("BYOM_HOME", str(link))
    res_symlink = runner.invoke(cli.app, ["uninstall", "--yes"])
    assert res_symlink.exit_code != 0
    assert link.is_symlink()
    assert alvo.exists()

    home_dir = Path.home()
    (home_dir / llm_chat.MARKER_FILENAME).write_text("nao apagar home", encoding="utf-8")
    for perigoso in (str(home_dir), "/", "", "."):
        monkeypatch.setenv("BYOM_HOME", perigoso)
        res = runner.invoke(cli.app, ["uninstall", "--yes"])
        assert res.exit_code != 0
    assert home_dir.exists()


def test_integracao_real_llm_cria_logs_db_apenas_em_byom_home(tmp_path, monkeypatch):
    """Confirma com o pacote `llm` real que `logs.db` e config ficam só em `BYOM_HOME`."""
    global_xdg = tmp_path / "global-xdg"
    global_xdg.mkdir()
    byom_dir = tmp_path / "real-byom-home"

    monkeypatch.setenv("XDG_CONFIG_HOME", str(global_xdg))
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))

    assert (
        runner.invoke(
            cli.app,
            ["setup", "--url", "http://127.0.0.1:49999/v1", "--model", "qwen3.8-27b"],
        ).exit_code
        == 0
    )

    env = {**os.environ, "LLM_USER_PATH": str(byom_dir)}
    proc = subprocess.run(
        [sys.executable, "-m", "llm", "models", "--query", "byom"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "byom" in proc.stdout

    proc_logs = subprocess.run(
        [sys.executable, "-m", "llm", "logs", "status"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(byom_dir / "logs.db") in proc_logs.stdout
    assert list(global_xdg.iterdir()) == []
