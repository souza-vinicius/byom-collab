"""Textos e telas da TUI: GPU mínima, avisos de leitura e saídas explícitas."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from byom import catalog, cli, config, tui
from byom import mensagens as M


class _Pergunta:
    def __init__(self, valor):
        self.valor = valor

    def ask(self):
        return self.valor


def _roteirizar(monkeypatch, respostas: list):
    import questionary

    fila = iter(respostas)

    def fake_select(*args, **kwargs):
        item = next(fila)
        return _Pergunta(item)

    monkeypatch.setattr(questionary, "select", fake_select)


def _console():
    return tui.criar_console(width=120, height=40, record=True)


# -- (a) GPU mínima calculada no contexto do próprio perfil -------------------
@pytest.mark.parametrize(
    ("profile_id", "esperado"),
    [
        ("qwen3.8-27b-q4-gguf", "L4"),
        ("inkling-2-mini-iq2-gguf", "T4"),
        ("gemma-4-31b-q4-gguf", "A100-40"),
        ("qwen-image-2.1-q4km", "T4"),
    ],
)
def test_gpu_minima_do_perfil_usa_o_contexto_dele(profile_id, esperado):
    perfis = {item.profile.id: item for item in tui.listar_perfis_estruturados(
        tui.ler_estado_local()
    )}

    assert perfis[profile_id].gpu_minima == esperado


def test_perfil_marcado_como_incompativel_nao_diz_exigir_a_propria_gpu(monkeypatch):
    """O antigo "Exige pelo menos L4; não cabe em L4" vinha de herdar min_gpu."""
    cfg = config.Config()
    config.apply_profile(cfg, "gemma-4-31b-q4-gguf")
    cfg.gpu = "L4"
    config.save_config(cfg)

    perfis = {item.profile.id: item for item in tui.listar_perfis_estruturados(
        tui.ler_estado_local()
    )}
    item = perfis["gemma-4-31b-q4-gguf"]

    assert item.cabe_gpu_atual is False
    assert item.gpu_minima != "L4"


def test_min_gpu_for_conhece_as_variantes_sem_perfil():
    assert catalog.min_gpu_for("gemma-4", "e2b-q4", 131072) in {"T4", "L4", "A100-40"}


# -- (b) tela de modelos compatíveis não é somente leitura -------------------
def test_tela_de_modelos_compativeis_avisa_que_nada_e_gravado(monkeypatch):
    import questionary

    cfg = config.Config()
    config.apply_profile(cfg, "qwen3.8-27b-32k")
    config.save_config(cfg)
    capturadas: list[list] = []

    def fake_select(*args, **kwargs):
        capturadas.append(list(kwargs.get("choices") or []))
        return _Pergunta("__voltar__")

    monkeypatch.setattr(questionary, "select", fake_select)
    rec = _console()

    tui._escolher_modelo_para_gpu(rec, tui.ler_estado_local(), "L4")

    texto = rec.export_text()
    assert M.NADA_GRAVADO_ATE_CONFIRMAR in texto
    assert M.SOMENTE_LEITURA not in texto
    titulos = [getattr(choice, "title", "") for choice in capturadas[0]]
    assert "Voltar ao menu principal" in titulos


# -- (c) órfãs: a TUI compara com a mensagem real da CLI --------------------
def test_marcador_de_sessao_sem_registro_vem_da_cli():
    runner = CliRunner()
    config.save_config(config.Config())

    with pytest.MonkeyPatch.context() as patch:
        orfa = cli.colab.SessionLine(
            name="byom-orfa", endpoint="m-s-orfa", hardware="L4", shape="STANDARD", variant="GPU"
        )
        patch.setattr(cli.colab, "unregistered_sessions", lambda known=None: [orfa])
        resultado = runner.invoke(cli.app, ["up"])

    assert M.MARCADOR_SEM_REGISTRO in resultado.output


def test_tui_compara_com_a_constante_e_nao_com_o_texto_antigo():
    """O texto da CLI mudou; o literal antigo não pode voltar à TUI."""
    import inspect

    fonte = inspect.getsource(tui)
    assert "M.MARCADOR_SEM_REGISTRO" in inspect.getsource(tui._executar_up)
    assert "Orphaned Colab sessions" not in fonte


# -- (d) estado ilegível tem painel próprio ---------------------------------
def test_opcoes_up_com_estado_ilegivel_usa_o_painel_de_estado(monkeypatch):
    path = config.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{invalido", encoding="utf-8")

    with pytest.raises(tui._ErroEtapa) as excinfo:
        tui._opcoes_up()

    assert "estado local" in excinfo.value.falha
    assert "state.json" in excinfo.value.falha or "state.json" in excinfo.value.causa
    assert "Caminho inexistente" not in excinfo.value.falha


# -- (e) escolher GPU oferece "Voltar ao menu principal" -------------------
def test_escolher_gpu_oferece_voltar_ao_menu(monkeypatch):
    import questionary

    capturadas: list[list] = []

    def fake_select(*args, **kwargs):
        capturadas.append(list(kwargs.get("choices") or []))
        return _Pergunta("__voltar__")

    monkeypatch.setattr(questionary, "select", fake_select)
    rec = _console()

    escolha = tui._escolher_gpu(rec, tui.ler_estado_local())

    assert escolha == "__voltar__"
    titulos = [getattr(choice, "title", "") for choice in capturadas[0]]
    assert "Voltar ao menu principal" in titulos


def test_gpu_modelo_volta_ao_menu_quando_pedem_voltar(monkeypatch):
    monkeypatch.setattr(tui, "_troca_bloqueada_por_sessao", lambda con: None)
    monkeypatch.setattr(tui, "renderizar_gpus", lambda con, estado: None)
    monkeypatch.setattr(tui, "_escolher_gpu", lambda con, estado: "__voltar__")
    rec = _console()

    assert tui._executar_gpu_modelo(rec, tui.ler_estado_local()) == "menu"


# -- (f) `byom models` não mostra "Contexto servido: 1" para imagem --------
def test_format_models_marca_contexto_de_imagem_como_n_a():
    texto = catalog.format_models()
    linhas_imagem = [
        linha for linha in texto.splitlines() if linha.startswith("qwen-image-2.1 |")
    ]
    assert linhas_imagem
    for linha in linhas_imagem:
        assert "| n/a (imagem) |" in linha
    assert "1 |" not in texto.split("qwen-image-2.1", 1)[1].splitlines()[0]


def test_tela_de_gpu_e_modelo_nao_diz_somente_leitura():
    """O fluxo grava config.toml no fim; o aviso tem de ser o de nada gravado."""
    from pathlib import Path

    fonte = (Path(__file__).resolve().parent.parent / "src" / "byom" / "tui.py").read_text(
        encoding="utf-8"
    )
    trecho = fonte.split("def _executar_gpu_modelo")[1].split("def _executar_selecionar_gpu")[0]
    assert "M.NADA_GRAVADO_ATE_CONFIRMAR" in trecho
    assert "M.SOMENTE_LEITURA" not in trecho
