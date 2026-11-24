"""TUI com config.toml inválido ou state.json ilegível.

A interface é o único lugar onde o usuário consegue consertar esses arquivos:
uma exceção de parse não pode virar "erro inesperado" e fechar o menu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from byom import cli, config, tui


class _Pergunta:
    """Resposta roteirizada de `questionary.select`."""

    def __init__(self, valor: Any) -> None:
        self.valor = valor

    def ask(self) -> Any:
        return self.valor


def _roteirizar(monkeypatch, respostas: list[Any]) -> list[Any]:
    """Responde `questionary.select` em ordem e devolve as escolhas do menu."""
    import questionary

    fila = iter(respostas)
    capturadas: list[Any] = []

    def fake_select(*args, **kwargs):
        capturadas.extend(kwargs.get("choices") or [])
        return _Pergunta(next(fila))

    monkeypatch.setattr(questionary, "select", fake_select)
    return capturadas


def _console(monkeypatch) -> Any:
    rec = tui.criar_console(width=100, height=30, record=True)
    monkeypatch.setattr(tui, "console", rec)
    return rec


def _toml_quebrado() -> Path:
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("proxy_port = = 4000\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# config.toml inválido
# ---------------------------------------------------------------------------


def test_toml_invalido_vira_config_error_com_arquivo_e_posicao():
    path = _toml_quebrado()

    with pytest.raises(config.ConfigError) as exc:
        config.load_config()

    mensagem = str(exc.value)
    assert str(path) in mensagem
    assert "line 1" in mensagem
    assert "column" in mensagem


def test_mostrar_menu_renderiza_com_config_invalida_e_explica_na_tela_4(monkeypatch):
    """O menu aparece, a barra acusa o erro e a opção 4 explica a causa."""
    path = _toml_quebrado()
    conteudo = path.read_text(encoding="utf-8")
    monkeypatch.setattr(cli.colab, "orphan_assignments", lambda: [])
    monkeypatch.setattr(cli.colab, "usage", lambda: "")
    escolhas = _roteirizar(monkeypatch, ["status", None])
    rec = _console(monkeypatch)

    tui.mostrar_menu(cli)

    saida = rec.export_text(clear=False)
    assert "erro na config" in saida
    # O menu foi construído (as escolhas existem) mesmo com o TOML inválido.
    rotulos = [str(getattr(choice, "title", "")) for choice in escolhas]
    assert any("Verificar estado da sessão" in rotulo for rotulo in rotulos)
    assert any("Selecionar GPU" in rotulo for rotulo in rotulos)
    assert "erro inesperado" not in saida
    assert "config.toml" in saida
    assert "line 1" in saida
    # Nada é gravado: o arquivo inválido continua exatamente como estava.
    assert path.read_text(encoding="utf-8") == conteudo


def _chave_ssh_padrao() -> None:
    cfg = config.Config()
    cfg.ssh_key_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ssh_key_path.write_text("PRIVATE", encoding="utf-8")


def test_tela_status_explica_config_invalida(monkeypatch, isolated_home):
    _toml_quebrado()
    rec = tui.criar_console(width=100, height=30, record=True)

    with pytest.raises(tui._ErroEtapa) as exc:
        tui._executar_status(rec, cli, tui.ler_estado_local())

    assert "config.toml" in exc.value.acao
    assert "line 1" in exc.value.causa


def test_tela_up_explica_config_invalida(isolated_home):
    _toml_quebrado()
    _chave_ssh_padrao()
    rec = tui.criar_console(width=100, height=30, record=True)

    with pytest.raises(tui._ErroEtapa) as exc:
        tui._executar_up(rec, cli, tui.ler_estado_local())

    assert "config.toml" in exc.value.acao


def test_tela_harness_explica_config_invalida(isolated_home):
    _toml_quebrado()
    rec = tui.criar_console(width=100, height=30, record=True)

    with pytest.raises(tui._ErroEtapa) as exc:
        tui._executar_harness(rec, cli, tui.ler_estado_local())

    assert "config.toml" in exc.value.acao


def test_tela_chat_explica_config_invalida(isolated_home):
    _toml_quebrado()
    config.save_state(
        config.State(session="byom-x", endpoint="e", nonce="n" * 32, created_at="2026-01-01")
    )
    rec = tui.criar_console(width=100, height=30, record=True)

    with pytest.raises(tui._ErroEtapa) as exc:
        tui._executar_chat(rec, cli, tui.ler_estado_local())

    assert "config.toml" in exc.value.acao
