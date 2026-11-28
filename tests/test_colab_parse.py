"""Tests for the `colab sessions` and `colab status` parser."""

from __future__ import annotations

import pytest

from byom import colab

SESSIONS_OUTPUT = """[colab] No active sessions found on server.
"""

SESSIONS_OUTPUT_WITH_LINES = (
    "[byom-1a2b3c4d] /tun/m/abcdef | Hardware: L4 | Shape: STANDARD | Variant: GPU\n"
    "[?] /tun/m/ghijkl | Hardware: T4 | Shape: STANDARD | Variant: GPU\n"
)

STATUS_OUTPUT = """[colab] Creating session 'byom-1a2b3c4d'...
[byom-1a2b3c4d] /tun/m/abcdef | Hardware: L4 | Shape: STANDARD | Variant: GPU | Status: IDLE
  Last Execution: bootstrap.sh at 2026-11-09 18:00:00
"""


def test_no_sessions():
    assert colab.parse_session_lines(SESSIONS_OUTPUT) == []


def test_full_line():
    lines = colab.parse_session_lines(SESSIONS_OUTPUT_WITH_LINES)
    assert len(lines) == 2
    first = lines[0]
    assert first.name == "byom-1a2b3c4d"
    assert first.endpoint == "/tun/m/abcdef"
    assert first.hardware == "L4"
    assert first.shape == "STANDARD"
    assert first.variant == "GPU"
    assert first.status is None
    assert first.is_l4 is True


def test_unknown_name_becomes_none():
    lines = colab.parse_session_lines(SESSIONS_OUTPUT_WITH_LINES)
    assert lines[1].name is None
    assert lines[1].endpoint == "/tun/m/ghijkl"
    assert lines[1].is_l4 is False


def test_status_with_status_and_extra_line():
    lines = colab.parse_session_lines(STATUS_OUTPUT)
    assert len(lines) == 1
    assert lines[0].status == "IDLE"


def test_invalid_lines_are_ignored():
    text = (
        "loose text\n"
        "[colab] warning\n"
        "\n"
        "[byom-x] /tun/m/z | Hardware: L4 | Shape: STANDARD | Variant: GPU\n"
    )
    assert len(colab.parse_session_lines(text)) == 1


def test_base_command_uses_isolated_config(monkeypatch):
    monkeypatch.setenv("BYOM_COLAB_BIN", "/usr/local/bin/colab")
    cmd = colab.base_command()
    assert cmd[0] == "/usr/local/bin/colab"
    assert cmd[1] == "--config"
    assert cmd[2].endswith("colab-sessions.json")


def test_base_command_creates_the_config_directory(monkeypatch):
    """The CLI creates the lock before the directory, so it must exist."""
    from byom import config

    monkeypatch.setenv("BYOM_COLAB_BIN", "/usr/local/bin/colab")
    target = config.colab_sessions_path().parent
    assert not target.exists()

    colab.base_command()

    assert target.is_dir()


ORPHAN_OUTPUT = (
    "[?] m-s-usc1c8-35ulubuesk9zn | Hardware: CPU | Shape: Standard | Variant: DEFAULT\n"
)


def test_orphan_session_is_detected():
    lines = colab.parse_session_lines(ORPHAN_OUTPUT)
    assert len(lines) == 1
    assert lines[0].name is None
    assert lines[0].endpoint == "m-s-usc1c8-35ulubuesk9zn"


def test_new_warns_about_orphan_session(monkeypatch):
    """The CLI allocates before saving local state.

    If the write fails, the VM stays allocated without a record and `colab
    stop` cannot stop it. The failure must say so, otherwise the session
    silently consumes compute units.
    """

    def failing_run(args, **kwargs):
        raise colab.ColabCliError("`colab new` failed (exit code 1): FileNotFoundError")

    monkeypatch.setattr(colab, "run", failing_run)
    monkeypatch.setattr(
        colab,
        "sessions",
        lambda: colab.parse_session_lines(ORPHAN_OUTPUT),
    )

    with pytest.raises(colab.ColabCliError) as excinfo:
        colab.new("byom-test", "L4")

    message = str(excinfo.value)
    assert "m-s-usc1c8-35ulubuesk9zn" in message
    assert "compute unit usage" in message


def test_new_without_orphans_does_not_pollute_the_message(monkeypatch):
    def failing_run(args, **kwargs):
        raise colab.ColabCliError("some failure")

    monkeypatch.setattr(colab, "run", failing_run)
    monkeypatch.setattr(colab, "sessions", list)

    with pytest.raises(colab.ColabCliError) as excinfo:
        colab.new("byom-test")

    assert "CCUs" not in str(excinfo.value)


def test_new_warns_even_when_listing_fails(monkeypatch):
    def failing_run(args, **kwargs):
        raise colab.ColabCliError("some failure")

    def failing_sessions():
        raise colab.ColabCliError("no network")

    monkeypatch.setattr(colab, "run", failing_run)
    monkeypatch.setattr(colab, "sessions", failing_sessions)

    with pytest.raises(colab.ColabCliError) as excinfo:
        colab.new("byom-test")

    assert "Check Colab" in str(excinfo.value)


def test_new_classifica_503_como_indisponibilidade_do_backend(monkeypatch):
    """O `colab new` imprime `Service Unavailable`, sem o código HTTP.

    A TUI precisa distinguir indisponibilidade temporária do Colab de falta de
    unidades de computação: a mensagem genérica mandava o usuário conferir a
    cota mesmo com saldo sobrando.
    """

    def failing_run(args, **kwargs):
        raise colab.ColabCliError(
            "`colab new -s byom-test --gpu L4` failed (exit code 1): "
            "ColabRequestError: Failed to issue request POST "
            "https://colab.research.google.com/tun/m/assign?nbh=x: Service Unavailable"
        )

    monkeypatch.setattr(colab, "run", failing_run)
    monkeypatch.setattr(colab, "sessions", list)

    with pytest.raises(colab.ColabBackendUnavailable):
        colab.new("byom-test", "L4")


@pytest.mark.parametrize(
    ("mensagem", "tipo"),
    [
        (
            "[colab] Allocation refused (precondition failed). This can mean too many "
            "active sessions, or a temporary usage or capacity limit.",
            colab.ColabAllocationRefused,
        ),
        (
            "[colab] Backend rejected accelerator 'A100'. You may not have quota or "
            "entitlement for this accelerator on your account.",
            colab.ColabAcceleratorRejected,
        ),
    ],
)
def test_new_classifica_recusas_da_cli(monkeypatch, mensagem, tipo):
    def failing_run(args, **kwargs):
        raise colab.ColabCliError(mensagem)

    monkeypatch.setattr(colab, "run", failing_run)
    monkeypatch.setattr(colab, "sessions", list)

    with pytest.raises(tipo):
        colab.new("byom-test", "L4")


def test_new_mantem_erro_generico_quando_nao_reconhece_a_causa(monkeypatch):
    def failing_run(args, **kwargs):
        raise colab.ColabCliError("unexpected failure")

    monkeypatch.setattr(colab, "run", failing_run)
    monkeypatch.setattr(colab, "sessions", list)

    with pytest.raises(colab.ColabCliError) as excinfo:
        colab.new("byom-test", "L4")

    assert type(excinfo.value) is colab.ColabCliError


def test_new_adds_high_mem_flag(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args

    monkeypatch.setattr(colab, "run", fake_run)

    colab.new("byom-a100", "A100", high_mem=True)

    assert captured["args"] == ["new", "-s", "byom-a100", "--gpu", "A100", "--high-mem"]


def test_long_error_is_truncated_at_the_end():
    text = "start\n" * 1000 + "FileNotFoundError: real cause"
    truncated = colab.tail(text, limit=100)
    assert len(truncated) == 103
    assert truncated.endswith("FileNotFoundError: real cause")
    assert colab.tail("short", limit=100) == "short"


def test_status_line_exige_nome_correspondente(monkeypatch):
    text = (
        "[outra-sessao] /tun/m/outra | Hardware: L4 | Shape: STANDARD | Variant: GPU\n"
        "[byom-1a2b3c4d] /tun/m/propria | Hardware: L4 | Shape: STANDARD | Variant: GPU\n"
    )

    class FakeProc:
        stdout = text.encode()

    monkeypatch.setattr(colab, "run", lambda args, **kwargs: FakeProc())

    line = colab.status_line("byom-1a2b3c4d")
    assert line is not None
    assert line.endpoint == "/tun/m/propria"
    assert colab.status_line("sessao-ausente") is None


def test_exec_popen_isolates_process_and_closes_log_in_parent(monkeypatch, tmp_path):
    # O `colab` não precisa existir no PATH: a suíte roda com PATH mínimo.
    monkeypatch.setenv("BYOM_COLAB_BIN", "/usr/local/bin/colab")
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(colab.subprocess, "Popen", fake_popen)
    log_path = tmp_path / "watchdog.log"

    process = colab.exec_popen("byom-test", tmp_path / "watchdog.py", log_path=log_path)

    assert process.pid == 4242
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdout"].closed is True
    assert captured["kwargs"]["stderr"].closed is True


def test_unregistered_sessions_inclui_byom_sem_registro(monkeypatch):
    """Sessão `byom-*` sem state.json é órfã tanto quanto as `[?]`.

    O arquivo de sessões é exclusivo do byom, então um nome `byom-` sempre
    pertence a esta ferramenta: não há falso positivo de terceiros.
    """
    texto = (
        "[?] m-s-x | Hardware: CPU | Shape: Standard | Variant: DEFAULT\n"
        "[byom-aaaa1111] m-s-y | Hardware: L4 | Shape: STANDARD | Variant: GPU\n"
        "[byom-bbbb2222] m-s-z | Hardware: L4 | Shape: STANDARD | Variant: GPU\n"
        "[outra-sessao] m-s-w | Hardware: T4 | Shape: STANDARD | Variant: GPU\n"
    )
    monkeypatch.setattr(colab, "sessions", lambda: colab.parse_session_lines(texto))

    achadas = colab.unregistered_sessions({"byom-aaaa1111"})

    assert [line.name for line in achadas] == [None, "byom-bbbb2222"]
    assert [line.endpoint for line in achadas] == ["m-s-x", "m-s-z"]


def test_unregistered_sessions_sem_estado_considera_todos_os_byom(monkeypatch):
    texto = (
        "[byom-aaaa1111] m-s-y | Hardware: L4 | Shape: STANDARD | Variant: GPU\n"
        "[?] m-s-x | Hardware: CPU | Shape: Standard | Variant: DEFAULT\n"
    )
    monkeypatch.setattr(colab, "sessions", lambda: colab.parse_session_lines(texto))

    achadas = colab.unregistered_sessions(set())

    assert {line.endpoint for line in achadas} == {"m-s-y", "m-s-x"}


# -- 4.4: ANSI, coluna extra e "não entendi" ≠ "não há sessão" ----------------
ANSI_LINE = (
    "\x1b[32m[byom-1a2b3c4d]\x1b[0m \x1b[1m/tun/m/abcdef\x1b[0m | "
    "\x1b[36mHardware:\x1b[0m L4 | Shape: STANDARD | Variant: GPU"
)

EXTRA_COLUMN_LINE = (
    "[byom-1a2b3c4d] /tun/m/abcdef | Hardware: L4 | Shape: STANDARD | "
    "Variant: GPU | Status: IDLE | Region: us-central1 | Kernel: python3"
)


def test_linha_com_ansi_e_parseada(monkeypatch):
    monkeypatch.setattr(colab, "run", lambda args, **kwargs: _proc(ANSI_LINE + "\n"))

    linhas = colab.sessions()

    assert len(linhas) == 1
    assert linhas[0].name == "byom-1a2b3c4d"
    assert linhas[0].endpoint == "/tun/m/abcdef"
    assert linhas[0].hardware == "L4"
    assert linhas[0].is_l4 is True


def test_linha_com_coluna_extra_e_parseada(monkeypatch):
    monkeypatch.setattr(colab, "run", lambda args, **kwargs: _proc(EXTRA_COLUMN_LINE + "\n"))

    linhas = colab.sessions()

    assert len(linhas) == 1
    assert linhas[0].name == "byom-1a2b3c4d"
    assert linhas[0].endpoint == "/tun/m/abcdef"
    assert linhas[0].status == "IDLE"


def test_saida_com_lixo_levanta_erro_de_formato(monkeypatch):
    monkeypatch.setattr(
        colab,
        "run",
        lambda args, **kwargs: _proc("something changed entirely\nother: value\n"),
    )

    with pytest.raises(colab.ColabCliError) as excinfo:
        colab.sessions()

    mensagem = str(excinfo.value)
    assert "formato do `colab sessions` não reconhecido" in mensagem
    assert "something changed entirely" in mensagem


def test_lista_realmente_vazia_continua_devolvendo_lista_vazia(monkeypatch):
    for texto in ("", "\n", "[colab] No active sessions found on server.\n", "Sessions:\n"):
        monkeypatch.setattr(
            colab, "run", lambda args, _texto=texto, **kwargs: _proc(_texto)
        )
        assert colab.sessions() == []


def test_down_nao_limpa_state_quando_o_parse_falha(tmp_path, monkeypatch):
    """Formato desconhecido não é prova de que a VM morreu."""
    from typer.testing import CliRunner

    from byom import cli, config

    monkeypatch.setenv("BYOM_CONFIG_DIR", str(tmp_path / "byom"))
    config.save_config(config.Config())
    config.save_state(
        config.State(
            session="byom-viva", endpoint="m-s-viva", nonce="n" * 32, created_at="agora"
        )
    )
    monkeypatch.setattr(colab, "stop", lambda name: None)
    monkeypatch.setattr(
        colab,
        "sessions",
        lambda: (_ for _ in ()).throw(
            colab.ColabCliError("formato do `colab sessions` não reconhecido: x")
        ),
    )

    resultado = CliRunner().invoke(cli.app, ["down"])

    assert resultado.exit_code == 1
    assert "could not confirm" in resultado.output
    estado = config.load_state()
    assert estado is not None and estado.session == "byom-viva"


def _proc(texto: str):
    class FakeProc:
        returncode = 0
        stdout = texto.encode()
        stderr = b""

    return FakeProc()

