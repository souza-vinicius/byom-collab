"""Falha do `up` não pode deixar daemon local vivo nem esperar 30 min."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest
from typer.testing import CliRunner

from byom import cli, colab, config, remote

runner = CliRunner()


class _FakeRemote:
    def __init__(self, session, cfg):
        self.session = session
        self.cfg = cfg

    def start(self, forwards=(), on_progress=None):
        return None

    def stop(self):
        return None

    def gpu_name(self):
        return "Tesla T4"

    def upload(self, *args, **kwargs):
        return None

    def run(self, *args, **kwargs):
        return remote.RemoteResult(returncode=0, output="")


def _install(monkeypatch, config_obj: config.Config | None = None) -> config.Config:
    cfg = config_obj or config.Config()
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    config.save_config(cfg)
    monkeypatch.setattr(cli, "_sync_external_configs", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_orphans", lambda: None)
    monkeypatch.setattr(cli, "_bootstrap_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli.remote, "Remote", _FakeRemote)
    monkeypatch.setattr(colab, "new", lambda *a, **k: None)
    monkeypatch.setattr(colab, "stop", lambda name: None)
    monkeypatch.setattr(colab, "is_allocated", lambda name, endpoint="": False)
    monkeypatch.setattr(
        colab,
        "status_line",
        lambda name: colab.SessionLine(
            name=name, endpoint="m-s-abc", hardware="T4", shape="STANDARD", variant="GPU"
        ),
    )
    return cfg


def _children_on_port(port: int) -> subprocess.Popen[bytes]:
    """Processo local que segura a porta do proxy, como um daemon órfão."""
    script = (
        "import socket, time\n"
        "sock = socket.socket()\n"
        "sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        f"sock.bind(('127.0.0.1', {port}))\n"
        "sock.listen(1)\n"
        "time.sleep(60)\n"
    )
    return subprocess.Popen([sys.executable, "-c", script])


def _wait_port_taken(port: int, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not cli._port_free(port):
            return
        time.sleep(0.02)
    raise AssertionError(f"porta {port} não foi ocupada a tempo")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_porta_do_proxy_ocupada_aborta_antes_de_alocar():
    porta = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", porta))
        sock.listen(1)
        alocou: list[str] = []

        def allocate(*args, **kwargs):
            alocou.append("sim")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(colab, "new", allocate)
            patch.setattr(config, "load_config", lambda: config.Config(proxy_port=porta))
            resultado = runner.invoke(cli.app, ["up"])

    assert resultado.exit_code == 1
    assert f"a porta {porta} do proxy já está em uso" in resultado.output
    assert alocou == []


def test_falha_em_wait_ready_encerra_o_daemon_e_libera_a_porta(monkeypatch):
    porta = _free_port()
    filho = _children_on_port(porta)
    try:
        _wait_port_taken(porta)
        _install(monkeypatch, config.Config(proxy_port=porta))
        monkeypatch.setattr(cli, "TERMINATE_GRACE_S", 0.5)
        # O objetivo aqui é o teardown: o filho já ocupa a porta (como um daemon
        # órfão), então a checagem prévia não se aplica a este cenário.
        monkeypatch.setattr(cli, "_check_proxy_port", lambda cfg: None)
        monkeypatch.setattr(cli, "_start_daemon", lambda: filho.pid)
        monkeypatch.setattr(cli, "_pid_is_byom", lambda *a, **k: filho.poll() is None)
        monkeypatch.setattr(
            cli, "_wait_ready", lambda *a, **k: (_ for _ in ()).throw(
                TimeoutError("worker indisponível: túnel falhou")
            )
        )

        resultado = runner.invoke(cli.app, ["up"])

        assert resultado.exit_code == 1
        assert "Failed" in resultado.output
        filho.wait(timeout=10)
        assert filho.poll() is not None, "o daemon local tem de ser encerrado"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not cli._port_free(porta):
            time.sleep(0.05)
        assert cli._port_free(porta), "a porta do proxy tem de ficar livre"
    finally:
        if filho.poll() is None:
            filho.kill()
            filho.wait(timeout=5)


def test_daemon_que_morre_no_popen_aborta_citando_a_porta(monkeypatch):
    cfg = _install(monkeypatch, config.Config(proxy_port=_free_port()))
    monkeypatch.setattr(cli, "DAEMON_CONFIRM_TICKS", 3)
    monkeypatch.setattr(cli, "DAEMON_CONFIRM_INTERVAL_S", 0.01)
    # PID que não existe: o filho "morreu logo após iniciar".
    monkeypatch.setattr(cli, "_start_daemon", lambda: 4_194_303)
    monkeypatch.setattr(cli, "_pid_is_byom", lambda *a, **k: False)

    resultado = runner.invoke(cli.app, ["up"])

    assert resultado.exit_code == 1
    assert "morreu logo após iniciar" in resultado.output
    assert str(cfg.proxy_port) in resultado.output


def test_worker_unavailable_com_gave_up_falha_rapido(monkeypatch):
    cfg = config.Config()
    monkeypatch.setattr(cli, "READY_POLL_S", 0.01)
    monkeypatch.setattr(
        cli,
        "_daemon_status",
        lambda cfg, expected_session=None: {
            "session": expected_session,
            "worker": {"state": "unavailable", "error": "No watchdog heartbeat for 90s"},
            "tunnel": {"state": "connected"},
            "watchdog": {"state": "gave_up", "last_error": "watchdog desistiu após 5 reinícios"},
        },
    )

    inicio = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        cli._wait_ready(cfg, 30, expected_session="byom-x")
    decorrido = time.monotonic() - inicio

    assert decorrido < 10
    assert "worker indisponível" in str(excinfo.value)
    assert "watchdog" in str(excinfo.value)


def test_tunel_failed_tambem_falha_rapido(monkeypatch):
    cfg = config.Config()
    monkeypatch.setattr(cli, "READY_POLL_S", 0.01)
    monkeypatch.setattr(
        cli,
        "_daemon_status",
        lambda cfg, expected_session=None: {
            "session": expected_session,
            "worker": {"state": "unavailable", "error": "Gateway is inaccessible"},
            "tunnel": {"state": "failed", "last_error": "Tunnel failed to connect: 429"},
        },
    )

    inicio = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        cli._wait_ready(cfg, 30, expected_session="byom-x")

    assert time.monotonic() - inicio < 10
    assert "429" in str(excinfo.value)


def test_status_mudo_por_tres_ciclos_falha_rapido(monkeypatch):
    cfg = config.Config()
    monkeypatch.setattr(cli, "READY_POLL_S", 0.01)
    monkeypatch.setattr(cli, "_daemon_status", lambda cfg, expected_session=None: None)

    inicio = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        cli._wait_ready(cfg, 30, expected_session="byom-x")

    assert time.monotonic() - inicio < 10
    assert "parou de responder" in str(excinfo.value)


def test_daemon_morto_durante_a_espera_falha_rapido(monkeypatch):
    cfg = config.Config()
    monkeypatch.setattr(cli, "READY_POLL_S", 0.01)
    monkeypatch.setattr(cli, "_pid_is_byom", lambda *a, **k: False)
    monkeypatch.setattr(
        cli,
        "_daemon_status",
        lambda cfg, expected_session=None: {
            "session": expected_session,
            "worker": {"state": "initializing"},
            "tunnel": {},
        },
    )
    state = config.State(
        session="byom-x", endpoint="", nonce="n" * 32, created_at="t", daemon_pid=4_242
    )

    with pytest.raises(TimeoutError) as excinfo:
        cli._wait_ready(cfg, 30, expected_session="byom-x", state=state)

    assert "não existe mais" in str(excinfo.value)


def test_port_free_detecta_porta_em_uso():
    porta = _free_port()
    assert cli._port_free(porta) is True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", porta))
        sock.listen(1)
        assert cli._port_free(porta) is False
