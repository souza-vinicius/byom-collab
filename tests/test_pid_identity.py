"""Identidade dos processos locais: sem prova estrutural, nada é sinalizado."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from byom import cli, config


def _spawn(argv: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _esperar_argv(pid: int, token: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if token in config.process_cmdline(pid):
            return
        time.sleep(0.02)
    raise AssertionError(f"argv de {pid} não contém {token!r} a tempo")


@pytest.fixture
def dorminhoco(request):
    processo = _spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    request.addfinalizer(lambda: _matar(processo))
    return processo


def _matar(processo: subprocess.Popen[bytes]) -> None:
    if processo.poll() is None:
        processo.kill()
        processo.wait(timeout=5)


def test_processo_com_byom_no_argv_mas_sem_estrutura_nao_e_do_byom(dorminhoco):
    """Antes: `"byom" in cmdline` aceitava qualquer processo do usuário."""
    processo = _spawn(
        [sys.executable, "-c", "import sys, time; time.sleep(60)", "byom", "outro"]
    )
    try:
        _esperar_argv(processo.pid, "byom")
        assert cli._pid_is_byom(processo.pid, None, "byom-x") is False
    finally:
        _matar(processo)


def test_argv_estrutural_do_daemon_e_reconhecido(monkeypatch, dorminhoco):
    monkeypatch.setattr(cli, "_process_argv", lambda pid: [sys.executable, "-m", "byom", "_daemon"])

    assert cli._pid_is_byom(dorminhoco.pid, None, "byom-x") is True
    assert cli._structural_kind(["/usr/bin/python3", "-m", "byom", "_daemon"]) == "daemon"
    assert cli._structural_kind(["/opt/bin/byom", "_daemon"]) == "daemon"
    assert cli._structural_kind([sys.executable, "-c", "sleep", "byom"]) is None


def test_argv_estrutural_do_watchdog_exige_a_sessao(monkeypatch, dorminhoco):
    argv = ["colab", "--config", "x", "exec", "-s", "byom-x", "-f", "/w/watchdog.py"]
    monkeypatch.setattr(cli, "_process_argv", lambda pid: argv)

    assert cli._pid_is_byom(dorminhoco.pid, None, "byom-x") is True
    assert cli._pid_is_byom(dorminhoco.pid, None, "byom-outra") is False


def test_terminate_local_mata_so_com_identidade_estrutural(dorminhoco):
    """`down` não pode matar um processo do usuário por causa da palavra byom."""
    processo = _spawn(
        [sys.executable, "-c", "import time; time.sleep(60)", "byom-rodando"]
    )
    try:
        _esperar_argv(processo.pid, "byom-rodando")
        estado = config.State(
            session="byom-x",
            endpoint="",
            nonce="n" * 32,
            created_at="t",
            daemon_pid=processo.pid,
        )
        config.save_state(estado)

        cli._terminate_local(estado)

        time.sleep(0.2)
        assert processo.poll() is None
        assert config.load_state().daemon_pid is None
    finally:
        _matar(processo)


def test_terminate_local_mata_daemon_com_argv_estrutural(monkeypatch):
    processo = _spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        monkeypatch.setattr(
            cli, "_process_argv", lambda pid: [sys.executable, "-m", "byom", "_daemon"]
        )
        monkeypatch.setattr(cli, "TERMINATE_GRACE_S", 1.0)
        estado = config.State(
            session="byom-x",
            endpoint="",
            nonce="n" * 32,
            created_at="t",
            daemon_pid=processo.pid,
        )
        config.save_state(estado)

        cli._terminate_local(estado)

        processo.wait(timeout=10)
        assert processo.poll() is not None
    finally:
        _matar(processo)


def test_sem_proc_a_decisao_nunca_e_morto(monkeypatch):
    """No macOS `start_ticks` é sempre None: o PID não pode ser dado como morto."""
    monkeypatch.setattr(config, "process_start_ticks", lambda pid: None)
    monkeypatch.setattr(cli, "_has_proc", lambda: False)
    monkeypatch.setattr(
        cli, "_process_argv", lambda pid: ["/usr/bin/python3", "-m", "byom", "_daemon"]
    )

    veredito = cli._pid_is_byom(4_242, None, "byom-x")

    assert veredito is True
    # Argv desconhecido: indefinido, nunca "morto".
    monkeypatch.setattr(cli, "_process_argv", lambda pid: ["/usr/bin/python3", "-c", "x"])
    assert cli._pid_is_byom(4_242, None, "byom-x") is None
    assert cli._pid_is_byom(4_242, None, "byom-x") is not False


def test_sem_proc_usa_ps_para_ler_o_argv(monkeypatch):
    visto: list[list[str]] = []

    class FakeProc:
        returncode = 0
        stdout = b"/usr/bin/python3 -m byom _daemon\n"

    def fake_run(command, **kwargs):
        visto.append(list(command))
        return FakeProc()

    monkeypatch.setattr(config, "process_start_ticks", lambda pid: None)
    monkeypatch.setattr(cli, "_has_proc", lambda: False)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._pid_is_byom(4_242, None, "byom-x") is True
    assert visto and visto[0][:3] == ["ps", "-p", "4242"]


def test_pid_inexistente_continua_morto():
    assert cli._pid_is_byom(4_194_303, None) is False
