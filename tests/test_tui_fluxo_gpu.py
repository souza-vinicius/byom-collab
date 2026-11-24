"""Fluxo de navegação GPU + modelo da TUI, com `questionary.select` roteirizado.

Cobre o caminho da tela única (menu → Selecionar GPU → GPU → modelo → usar),
cada ramo do submenu e o retorno "Voltar para a lista" do detalhe do perfil.
"""

from __future__ import annotations

from typing import Any

import pytest

from byom import catalog, cli, config, tui


class _Pergunta:
    def __init__(self, valor: Any) -> None:
        self.valor = valor

    def ask(self) -> Any:
        return self.valor


def _roteirizar(monkeypatch, respostas: list[Any]) -> list[list[Any]]:
    """Responde `select` em ordem; um item chamável escolhe dentro das opções."""
    import questionary

    fila = iter(respostas)
    capturadas: list[list[Any]] = []

    def fake_select(*args, **kwargs):
        escolhas = list(kwargs.get("choices") or [])
        capturadas.append(escolhas)
        item = next(fila)
        return _Pergunta(item(escolhas) if callable(item) else item)

    monkeypatch.setattr(questionary, "select", fake_select)
    return capturadas


def _perfil_view(choice: Any) -> Any:
    valor = getattr(choice, "value", None)
    return valor if isinstance(valor, tui.PerfilView) else None


def _primeiro_perfil(escolhido: dict[str, Any]):
    def escolher(escolhas: list[Any]) -> Any:
        alvo = next(_perfil_view(choice) for choice in escolhas if _perfil_view(choice))
        escolhido["item"] = alvo
        return alvo

    return escolher


# ---------------------------------------------------------------------------
# Fluxo completo roteirizado
# ---------------------------------------------------------------------------


def test_mostrar_menu_grava_o_par_gpu_e_modelo(monkeypatch, isolated_home):
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **k: "menu")
    escolhido: dict[str, Any] = {}
    respostas = [
        "selecionar_gpu",  # menu
        "gpu",  # submenu Selecionar GPU
        "L4",  # lista de GPUs
        _primeiro_perfil(escolhido),  # modelos compatíveis com L4
        "usar",  # confirmação do par
        True,  # _confirmar("configure")
        None,  # menu principal: sair
    ]
    _roteirizar(monkeypatch, respostas)
    rec = tui.criar_console(width=100, height=30, record=True)
    monkeypatch.setattr(tui, "console", rec)

    tui.mostrar_menu(tui)

    item = escolhido["item"]
    cfg = config.load_config()
    assert cfg.profile == item.profile.id
    assert (cfg.gpu, cfg.gpu_high_mem) == ("L4", False)
    assert cfg.max_model_len == item.profile.settings["max_model_len"]
    assert config.check_compatibility(cfg).fits

    texto = config.config_path().read_text(encoding="utf-8")
    assert 'gpu = "L4"' in texto
    assert "gpu_high_mem = false" in texto
    assert f'profile = "{item.profile.id}"' in texto
    assert f"max_model_len = {item.profile.settings['max_model_len']}" in texto


def test_par_gpu_e_modelo_nao_grava_nada_antes_da_confirmacao(monkeypatch, isolated_home):
    """Escolher "outro modelo" e voltar não deixa resíduo em config.toml."""
    monkeypatch.setattr(tui, "_perguntar_proximo_passo", lambda *a, **k: "menu")
    respostas = [
        "selecionar_gpu",
        "gpu",
        "L4",
        _primeiro_perfil({}),
        "menu",  # desiste na confirmação do par
        None,
    ]
    _roteirizar(monkeypatch, respostas)
    rec = tui.criar_console(width=100, height=30, record=True)
    monkeypatch.setattr(tui, "console", rec)

    tui.mostrar_menu(tui)

    assert not config.config_path().exists()


# ---------------------------------------------------------------------------
# Ramos do submenu "Selecionar GPU"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("escolha", "atributo"), [("perfil", "_executar_configure"), ("gpu", "_executar_gpu_modelo")]
)
def test_submenu_selecionar_gpu_despacha_cada_caminho(monkeypatch, escolha, atributo):
    chamadas: list[str] = []

    def fake(*args, **kwargs):
        chamadas.append(atributo)
        return "menu"

    monkeypatch.setattr(tui, atributo, fake)
    _roteirizar(monkeypatch, [escolha])
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._executar_selecionar_gpu(rec, tui.ler_estado_local()) == "menu"
    assert chamadas == [atributo]


@pytest.mark.parametrize("escolha", ["menu", None])
def test_submenu_selecionar_gpu_volta_ao_menu(monkeypatch, escolha):
    _roteirizar(monkeypatch, [escolha])
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._executar_selecionar_gpu(rec, tui.ler_estado_local()) == "menu"


def test_lista_de_modelos_vazia_volta_para_outra_gpu(monkeypatch):
    """Sem nenhuma variante compatível com perfil pronto, o caminho recua."""
    vazio = catalog.CompatibleSelection("L4", (), 0)
    monkeypatch.setattr(
        tui, "renderizar_modelos_por_gpu", lambda con, gpu_id, estado: (vazio, {})
    )
    capturadas = _roteirizar(monkeypatch, ["outra_gpu"])
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._escolher_modelo_para_gpu(rec, tui.ler_estado_local(), "L4") == "outra_gpu"
    # A tela vazia oferece as duas saídas, não só "outra GPU".
    titulos = [getattr(c, "title", "") for c in capturadas[0]]
    assert "Escolher outra GPU" in titulos
    assert "Voltar ao menu principal" in titulos


def test_lista_de_modelos_vazia_permite_voltar_ao_menu(monkeypatch):
    vazio = catalog.CompatibleSelection("L4", (), 0)
    monkeypatch.setattr(
        tui, "renderizar_modelos_por_gpu", lambda con, gpu_id, estado: (vazio, {})
    )
    _roteirizar(monkeypatch, ["menu"])
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._escolher_modelo_para_gpu(rec, tui.ler_estado_local(), "L4") == "menu"


def test_caminho_gpu_modelo_repete_a_lista_de_gpus_quando_pedem_outra(monkeypatch):
    monkeypatch.setattr(tui, "_troca_bloqueada_por_sessao", lambda con: None)
    monkeypatch.setattr(tui, "renderizar_gpus", lambda con, estado: None)
    monkeypatch.setattr(tui, "_escolher_gpu", lambda con, estado: "L4")
    respostas = iter(["outra_gpu", "menu"])
    chamadas: list[str] = []

    def escolher_modelo(con, estado, gpu_id):
        chamadas.append(gpu_id)
        return next(respostas)

    monkeypatch.setattr(tui, "_escolher_modelo_para_gpu", escolher_modelo)
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._executar_gpu_modelo(rec, tui.ler_estado_local()) == "menu"
    assert chamadas == ["L4", "L4"]


# ---------------------------------------------------------------------------
# "Voltar para a lista" volta para a lista, não para o menu principal
# ---------------------------------------------------------------------------


def test_voltar_para_a_lista_reabre_a_lista_de_perfis(monkeypatch, isolated_home):
    escolhidos: list[Any] = []

    def escolher_perfil(escolhas: list[Any]) -> Any:
        alvo = next(_perfil_view(choice) for choice in escolhas if _perfil_view(choice))
        escolhidos.append(alvo)
        return alvo

    respostas = [escolher_perfil, "lista", escolher_perfil, "menu"]
    _roteirizar(monkeypatch, respostas)
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._executar_configure(rec, tui.ler_estado_local()) == "menu"
    # Duas passagens pela lista: a primeira caiu no detalhe e voltou para a lista.
    assert len(escolhidos) == 2
    assert not config.config_path().exists()


def test_caminho_gpu_modelo_consome_lista_reabrindo_a_lista_de_gpus(monkeypatch):
    monkeypatch.setattr(tui, "_troca_bloqueada_por_sessao", lambda con: None)
    monkeypatch.setattr(tui, "renderizar_gpus", lambda con, estado: None)
    monkeypatch.setattr(tui, "_escolher_gpu", lambda con, estado: "L4")
    respostas = iter(["lista", "menu"])
    chamadas: list[str] = []

    def escolher_modelo(con, estado, gpu_id):
        chamadas.append(gpu_id)
        return next(respostas)

    monkeypatch.setattr(tui, "_escolher_modelo_para_gpu", escolher_modelo)
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._executar_gpu_modelo(rec, tui.ler_estado_local()) == "menu"
    assert chamadas == ["L4", "L4"]


@pytest.mark.parametrize(
    ("escolha", "esperado"), [("__outra_gpu__", "outra_gpu"), ("__voltar__", "menu")]
)
def test_lista_de_modelos_permite_trocar_de_gpu_ou_voltar(monkeypatch, escolha, esperado):
    _roteirizar(monkeypatch, [escolha])
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._escolher_modelo_para_gpu(rec, tui.ler_estado_local(), "L4") == esperado
    assert not config.config_path().exists()


def test_tela_harness_opencode_com_jsonc_mostra_manual_e_automatico(
    monkeypatch, isolated_home
):
    """Só `.jsonc`: a tela mostra o bloco manual e o menu, sem erro interno.

    O painel manual usava o resolvedor do sync, que levanta exceção para
    `.jsonc`; a exceção subia como falha inesperada e a única saída era Enter.
    """
    pasta = isolated_home / ".config" / "opencode"
    pasta.mkdir(parents=True)
    (pasta / "opencode.jsonc").write_text(
        '{\n  // mantém\n  "model": "x"\n}\n', encoding="utf-8"
    )

    capturadas = _roteirizar(monkeypatch, ["opencode", "menu"])
    rec = tui.criar_console(width=100, height=30, record=True)

    assert tui._executar_harness(rec, cli, tui.ler_estado_local()) == "menu"

    saida = rec.export_text(clear=False)
    assert "Configuração manual — OpenCode" in saida
    assert "opencode.jsonc" in saida
    titulos = [getattr(choice, "title", "") for choice in capturadas[1]]
    assert any(t.startswith("Configurar automaticamente") for t in titulos)
