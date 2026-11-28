"""Testes do frontend Open WebUI (`byom web`)."""

from __future__ import annotations

import socket
import stat

import pytest
from typer.testing import CliRunner

from byom import cli, config, llm_chat, webui

runner = CliRunner()


@pytest.fixture
def byom_home(tmp_path, monkeypatch):
    home = tmp_path / "webui-byom"
    monkeypatch.setenv("BYOM_HOME", str(home))
    monkeypatch.setattr(webui.shutil, "which", lambda name: "/usr/bin/uv")
    return home


class FakeProcess:
    def __init__(self, *, returncode: int = 0):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return None

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def test_plan_monta_comando_env_e_dados(byom_home):
    plan = webui.plan(port=8099, proxy_port=4100)

    assert plan.argv == (
        "uv",
        "tool",
        "run",
        "--python",
        "3.12",
        "--from",
        f"open-webui=={webui.OPEN_WEBUI_VERSION}",
        "open-webui",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8099",
    )
    assert plan.url == "http://127.0.0.1:8099"
    assert plan.proxy_url == "http://127.0.0.1:4100/v1"
    assert plan.data_dir == byom_home / webui.DATA_DIRNAME

    env = plan.env
    assert env["DATA_DIR"] == str(byom_home / webui.DATA_DIRNAME)
    assert env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:4100/v1"
    assert env["OPENAI_API_BASE_URLS"] == "http://127.0.0.1:4100/v1"
    assert env["OPENAI_API_KEYS"] == "byom"
    assert env["WEBUI_AUTH"] == "false"
    assert env["ENABLE_OLLAMA_API"] == "false"
    assert env["WEBUI_URL"] == "http://127.0.0.1:8099"
    assert env["WEBUI_SECRET_KEY"]

    assert (byom_home / llm_chat.MARKER_FILENAME).is_file()
    assert plan.data_dir.is_dir()
    secret = byom_home / webui.SECRET_FILENAME
    assert secret.is_file()
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600

    again = webui.plan(port=8099, proxy_port=4100)
    assert again.env["WEBUI_SECRET_KEY"] == env["WEBUI_SECRET_KEY"]


def test_plan_rejeita_portas_invalidas(byom_home):
    with pytest.raises(webui.WebUIError, match="Porta inválida"):
        webui.plan(port=0, proxy_port=4000)
    with pytest.raises(webui.WebUIError, match="Porta inválida"):
        webui.plan(port=70000, proxy_port=4000)
    with pytest.raises(webui.WebUIError, match="Porta inválida"):
        webui.plan(port=8080, proxy_port=0)


def test_plan_falha_sem_uv(byom_home, monkeypatch):
    monkeypatch.setattr(webui.shutil, "which", lambda name: None)
    with pytest.raises(webui.WebUIError, match="uv"):
        webui.plan(port=8080, proxy_port=4000)


def test_plan_remove_variaveis_do_venv_do_byom(byom_home, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/venv")
    monkeypatch.setenv("PYTHONHOME", "/tmp/python")
    monkeypatch.setenv("PYTHONPATH", "/tmp/src")

    plan = webui.plan(port=8080, proxy_port=4000)

    assert "VIRTUAL_ENV" not in plan.env
    assert "PYTHONHOME" not in plan.env
    assert "PYTHONPATH" not in plan.env


def test_plan_injeta_modelo_padrao_do_perfil_ativo(byom_home, monkeypatch):
    monkeypatch.delenv("DEFAULT_MODELS", raising=False)
    cfg = config.load_config()

    plan = webui.plan(port=8099, proxy_port=4000)

    assert plan.env["DEFAULT_MODELS"] == cfg.served_model_name


def test_plan_preserva_overrides_do_usuario(byom_home, monkeypatch):
    valores = {
        "OPENAI_API_BASE_URLS": "http://outro-provedor:8000/v1",
        "OPENAI_API_KEYS": "chave-do-usuario",
        "WEBUI_AUTH": "true",
        "DEFAULT_MODELS": "outro-modelo",
    }
    for nome, valor in valores.items():
        monkeypatch.setenv(nome, valor)

    plan = webui.plan(port=8099, proxy_port=4000)

    for nome, valor in valores.items():
        assert plan.env[nome] == valor
    assert set(valores) <= set(plan.overrides)
    assert "DATA_DIR" not in plan.overrides
    # O que é do byom continua presente quando o usuário não escolheu outro valor.
    assert plan.env["DATA_DIR"] == str(byom_home / webui.DATA_DIRNAME)
    assert plan.env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:4000/v1"


def test_cli_web_mostra_modelo_padrao_e_overrides(byom_home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEYS", "chave-do-usuario")
    planos: list[webui.WebUIPlan] = []

    def falso_run(plan, *, open_browser=True):
        planos.append(plan)
        return 0

    monkeypatch.setattr(webui, "run", falso_run)
    res = runner.invoke(cli.app, ["web", "--port", "8099", "--no-browser"])

    assert res.exit_code == 0
    assert "Padrão:   " in res.output
    assert "Ambiente: OPENAI_API_KEYS (valores preservados)" in res.output
    assert "OPENAI_API_BASE_URLS" in res.output
    assert planos[0].env["OPENAI_API_KEYS"] == "chave-do-usuario"


def test_run_recusa_porta_ocupada(byom_home):
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        port = holder.getsockname()[1]
        plan = webui.plan(port=port, proxy_port=4000)
        with pytest.raises(webui.WebUIError, match="já está em uso"):
            webui.run(plan, open_browser=False)
    finally:
        holder.close()


def test_run_abre_navegador_quando_health_responde(byom_home, monkeypatch):
    plan = webui.plan(port=8099, proxy_port=4000)
    proc = FakeProcess()
    opened: list[str] = []

    monkeypatch.setattr(webui.subprocess, "Popen", lambda argv, env: proc)
    monkeypatch.setattr(webui, "port_available", lambda port: True)
    monkeypatch.setattr(webui, "_wait_for_health", lambda url, process, timeout: True)
    monkeypatch.setattr(webui.webbrowser, "open", opened.append)

    assert webui.run(plan, open_browser=True) == 0
    assert opened == ["http://127.0.0.1:8099"]


def test_run_sem_navegador_nao_abre_e_encerra_no_ctrl_c(byom_home, monkeypatch):
    plan = webui.plan(port=8099, proxy_port=4000)
    proc = FakeProcess()
    opened: list[str] = []

    monkeypatch.setattr(webui.subprocess, "Popen", lambda argv, env: proc)
    monkeypatch.setattr(webui, "port_available", lambda port: True)
    monkeypatch.setattr(webui.webbrowser, "open", opened.append)

    assert webui.run(plan, open_browser=False) == 0
    assert opened == []

    def interrupted_wait(timeout=None):
        if timeout is None:
            raise KeyboardInterrupt
        return 0

    proc.wait = interrupted_wait
    assert webui.run(plan, open_browser=False) == 130
    assert proc.terminated is True


def test_cli_web_imprime_endpoint_e_respeita_no_browser(byom_home, monkeypatch):
    chamadas: list[bool] = []

    def fake_run(plan, *, open_browser=True):
        chamadas.append(open_browser)
        return 0

    monkeypatch.setattr(webui, "run", fake_run)
    res = runner.invoke(cli.app, ["web", "--port", "8099", "--no-browser"])

    assert res.exit_code == 0
    assert chamadas == [False]
    assert "Frontend: http://127.0.0.1:8099" in res.output
    assert "Modelo:   http://127.0.0.1:4000/v1" in res.output
    assert "Ctrl+C" in res.output


def test_cli_web_falha_com_porta_invalida(byom_home):
    res = runner.invoke(cli.app, ["web", "--port", "0"])
    assert res.exit_code != 0
    assert "Porta inválida" in res.output


def test_cli_help_exibe_web():
    res = runner.invoke(cli.app, ["--help"])
    assert res.exit_code == 0
    assert "web" in res.output
