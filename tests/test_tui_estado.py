"""TUI com state.json ilegível.

O arquivo pode esconder uma GPU alocada sem registro: a interface precisa
explicar e bloquear a troca de perfil em vez de estourar "erro inesperado"
depois da confirmação do usuário.
"""

from __future__ import annotations

from byom import cli, config, tui


def _state_truncado() -> None:
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"session": "byom-x"', encoding="utf-8")


def test_aplicar_perfil_com_state_ilegivel_mostra_painel_e_nao_grava(isolated_home):
    _state_truncado()
    itens = tui.listar_perfis_estruturados(tui.ler_estado_local())
    item = itens[0]
    rec = tui.criar_console(width=100, height=30, record=True)

    ok, _ = tui._aplicar_perfil_interativo(rec, item, pedir_confirmacao=False)

    assert ok is False
    assert not config.config_path().exists()
    saida = rec.export_text(clear=False)
    assert "state.json" in saida
    assert "byom status" in saida
    assert "O que fazer:" in saida


def test_troca_de_perfil_nao_e_liberada_com_state_ilegivel(isolated_home):
    _state_truncado()
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._troca_bloqueada_por_sessao(rec) == "menu"
    saida = rec.export_text(clear=False)
    assert "ilegível" in saida
    assert "Colab" in saida


def test_status_com_state_ilegivel_diz_estado_ilegivel_e_mantem_o_erro(
    monkeypatch, isolated_home
):
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config.save_config(config.Config())
    _state_truncado()
    monkeypatch.setattr(cli.colab, "orphan_assignments", lambda: [])
    monkeypatch.setattr(cli.colab, "usage", lambda: "")
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **k: "menu")
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._executar_status(rec, cli, tui.ler_estado_local()) == "menu"

    saida = rec.export_text(clear=False)
    assert "estado ilegível" in saida
    assert "sem sessão registrada" not in saida
    assert "state.json" in saida
