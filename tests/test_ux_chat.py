"""Ergonomia do chat e compatibilidade da interface com NO_COLOR e EOF."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from byom import cli, llm_chat, tui
from byom import mensagens as M

runner = CliRunner()


def test_chat_result_130_e_cancelamento_do_usuario():
    resultado = llm_chat.ChatResult(130, "Chat interrompido pelo usuário (Ctrl+C).\n")
    assert resultado.aborted is True


def test_ctrl_c_durante_o_chat_termina_sem_traceback(tmp_path, monkeypatch):
    byom_dir = tmp_path / "chat-ctrl-c"
    monkeypatch.setenv("BYOM_HOME", str(byom_dir))
    assert runner.invoke(cli.app, ["setup"]).exit_code == 0

    def interromper(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(llm_chat.subprocess, "run", interromper)
    res = runner.invoke(cli.app, ["chat", "--no-history"])

    assert res.exit_code == 0
    assert "Error" not in res.output
    assert "Traceback" not in res.output


def test_eof_no_prompt_de_voltar_encerra_o_menu(monkeypatch):
    class ConsoleSemEntrada:
        def print(self, *args, **kwargs):
            pass

        def input(self, *args, **kwargs):
            raise EOFError

    monkeypatch.setattr(tui, "is_interactive", lambda: True)

    with pytest.raises(KeyboardInterrupt):
        tui._aguardar_enter(ConsoleSemEntrada())


def test_estilo_do_select_respeita_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert ("selected", "noreverse") in tui._estilo_select().style_rules

    monkeypatch.setenv("NO_COLOR", "1")
    assert tui._estilo_select().style_rules == []


def test_mensagem_de_cancelamento_cita_ctrl_d():
    assert "Ctrl+D" in M.ERROS_COMUNS["cancelado"][1]


def test_menu_recria_console_quando_no_color_muda_apos_import(monkeypatch):
    import questionary

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(tui, "console", tui.Console(record=True, force_terminal=False))

    class Pergunta:
        def ask(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(questionary, "select", lambda *args, **kwargs: Pergunta())

    tui.mostrar_menu(cli)

    assert tui.console.no_color is True
