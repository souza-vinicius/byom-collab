"""Testes da camada interativa (TUI) e das mensagens centralizadas."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from byom import cli, tui
from byom import mensagens as M

runner = CliRunner()


def _saida(resultado) -> str:
    return (getattr(resultado, "stdout", "") or "") + (getattr(resultado, "stderr", "") or "")


def test_sem_argumentos_fora_de_tty_mostra_ajuda_e_mantem_exit_2():
    resultado = runner.invoke(cli.app, [])
    assert resultado.exit_code == 2
    assert "Usage:" in _saida(resultado)


def test_no_tui_forca_modo_cli():
    resultado = runner.invoke(cli.app, ["--no-tui"])
    assert resultado.exit_code == 2
    assert "Usage:" in _saida(resultado)


def test_tui_sem_tty_tem_erro_explicativo():
    resultado = runner.invoke(cli.app, ["--tui"])
    assert resultado.exit_code == 2
    saida = _saida(resultado)
    assert "terminal interativo" in saida
    assert "Causa provável" in saida
    assert "O que fazer" in saida


def test_tui_e_no_tui_juntos_prevalece_no_tui(monkeypatch):
    chamados: list[object] = []
    monkeypatch.setattr(tui, "mostrar_menu", lambda modulo: chamados.append(modulo))
    resultado = runner.invoke(cli.app, ["--tui", "--no-tui"])
    assert resultado.exit_code == 2
    assert chamados == []


def test_com_tty_sem_subcomando_abre_menu(monkeypatch):
    chamados: list[object] = []
    monkeypatch.setattr(tui, "is_interactive", lambda: True)
    monkeypatch.setattr(tui, "mostrar_menu", lambda modulo: chamados.append(modulo))
    resultado = runner.invoke(cli.app, [])
    assert resultado.exit_code == 0
    assert chamados == [cli]


def test_subcomando_nao_abre_menu(monkeypatch):
    def falha(modulo):  # pragma: no cover - só deve ser chamada se houver bug
        raise AssertionError("menu não deve abrir com subcomando")

    monkeypatch.setattr(tui, "is_interactive", lambda: True)
    monkeypatch.setattr(tui, "mostrar_menu", falha)
    resultado = runner.invoke(cli.app, ["profiles"])
    assert resultado.exit_code == 0
    assert "qwen" in _saida(resultado).lower()


def test_mensagem_de_erro_tem_tres_partes():
    texto = M.ERRO.format(falha="falha", causa="causa", acao="ação")
    assert "O que aconteceu:" in texto
    assert "Causa provável:" in texto
    assert "O que fazer:" in texto


def test_mensagem_de_resultado_vazio_explica_origem():
    falha, causa, acao = M.ERROS_COMUNS["vazio"]
    texto = M.ERRO.format(
        falha=falha.format(origem="profiles"),
        causa=causa.format(origem="profiles"),
        acao=acao.format(origem="profiles"),
    )
    assert "profiles" in texto
    assert "Verifique" in texto


def test_comando_equivalente_inclui_flags():
    assert (
        tui._comando_equivalente("models", {"gpu": "L4"})
        == "byom models --gpu L4"
    )
    assert (
        tui._comando_equivalente("configure", {"profile_id": "inkling-2-mini-q4-gguf"})
        == "byom configure --profile inkling-2-mini-q4-gguf"
    )
    assert tui._comando_equivalente("up", {"force": True}) == "byom up --force"


def test_todas_as_acoes_have_ajuda_e_proximo_passo():
    assert set(M.ACOES) == set(M.AJUDA_ACOES)
    assert set(M.ACOES) == set(M.ANTES)
    assert set(M.ACOES) == set(M.PROXIMO)


def test_down_recusa_confirmacao_nao_chama_comando(monkeypatch):
    monkeypatch.setattr(tui, "_confirmar", lambda chave: False)
    with pytest.raises(KeyboardInterrupt):
        tui._opcoes_down()
