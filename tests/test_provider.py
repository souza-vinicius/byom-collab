"""Fornecedor do modelo e publicador dos pesos na CLI e na TUI.

O fornecedor é o time/empresa que DESENVOLVE o modelo (dado do catálogo, com
fonte). O publicador é derivado em tempo de render da org do repositório de
onde o byom baixa os pesos — nunca hardcoded por perfil.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
from typer.testing import CliRunner

from byom import catalog, cli, tui
from byom import mensagens as M

runner = CliRunner()

#: Conferido em https://huggingface.co/<repo do desenvolvedor> em 2026-11-24.
PROVIDER_ESPERADO = {
    "inkling-2-mini": "Thinking Machines Lab",
    "inkling-2-mini-strata": "Thinking Machines Lab",
    "qwen3.8-27b-awq": "Qwen (Alibaba Cloud)",
    "qwen3.8-27b-gguf": "Qwen (Alibaba Cloud)",
    "gemma-4": "Google DeepMind",
    "muse-glimmer-30b": "Meta Superintelligence Lab",
    "qwen-image-2.1": "Qwen (Alibaba Cloud)",
    "qwen-image-2.1-turbo-uncensored": "Qwen (Alibaba Cloud)",
    "minimax-m3.1-flash": "MiniMax",
    "glm-5.3-flash": "Z.ai",
    "deepseek-v4.1-flash": "DeepSeek",
    "ternary-bonsai-2.5-27b": "Prism ML",
    "qwen3-tts": "Qwen (Alibaba Cloud)",
}


def _texto_perfis(largura: int) -> str:
    rec = tui.criar_console(width=largura, height=60, record=True)
    tui.renderizar_perfis(rec, tui.ler_estado_local())
    return rec.export_text(clear=False)


def _perfil(profile_id: str) -> tui.PerfilView:
    estado = tui.ler_estado_local()
    return next(
        item for item in tui.listar_perfis_estruturados(estado) if item.profile.id == profile_id
    )


def test_todo_modelo_declara_fornecedor_com_fonte():
    assert {model.id for model in catalog.MODELS} == set(PROVIDER_ESPERADO)
    for model in catalog.MODELS:
        assert model.provider.strip(), f"{model.id} sem fornecedor"
        assert model.provider == PROVIDER_ESPERADO[model.id], model.id
        fonte = catalog.PROVIDER_SOURCES[model.id]
        assert "huggingface.co" in fonte
        # Toda fonte é datada; a data varia por modelo (o pack do Strata foi
        # conferido em 2026-11-27, o resto em 2026-11-24).
        assert re.search(r"20\d\d-\d\d-\d\d", fonte), model.id


def test_package_owner_traduz_a_org_do_repo():
    assert catalog.package_owner("unsloth/Inkling-2-Mini-GGUF") == "Unsloth"
    assert catalog.package_owner("AtomicChat/Inkling-2-Mini-GGUF") == "AtomicChat"
    assert catalog.package_owner("Qwen/Qwen3-VL-8B-Instruct-GGUF") == "Qwen (Alibaba Cloud)"
    assert catalog.package_owner("Comfy-Org/Qwen-Image-2.1") == "Comfy-Org"
    assert catalog.package_owner("leejet/Qwen-Image-2.1-GGUF::qwen_image_2.1-Q4_K.gguf") == "leejet"
    assert catalog.package_owner("philbert440/Qwen3.8-27B-W4A16-AWQ") == "philbert440"
    # Org desconhecida volta como está: o mapa é só de nomes bonitos.
    assert catalog.package_owner("alguem/Modelo") == "alguem"
    assert catalog.package_owner("") == ""


def test_package_source_prefere_model_repo_e_cai_no_model():
    assert (
        catalog.package_source({"model_repo": "AtomicChat/Inkling-2-Mini-GGUF"}, "ignorado")
        == "AtomicChat/Inkling-2-Mini-GGUF"
    )
    assert (
        catalog.package_source(
            {"image_roles": {"diffusion": "leejet/Qwen-Image-2.1-GGUF", "vae": "Comfy-Org/x"}},
            "qwen-image-2.1",
        )
        == "leejet/Qwen-Image-2.1-GGUF"
    )
    assert (
        catalog.package_source({}, "philbert440/Qwen3.8-27B-W4A16-AWQ")
        == "philbert440/Qwen3.8-27B-W4A16-AWQ"
    )
    assert catalog.package_source(None, "") == ""


def test_perfil_unsloth_mostra_fornecedor_e_pacote_divergente():
    item = _perfil("inkling-2-mini-q4-gguf")
    assert item.fornecedor == "Thinking Machines Lab"
    assert item.pacote == "Unsloth"
    assert M.fornecedor_resumo(item.fornecedor, item.pacote) == (
        "Thinking Machines Lab · pesos Unsloth"
    )


def test_publicador_igual_ao_desenvolvedor_rende_uma_linha_so():
    item = _perfil("inkling-2-mini-q4-gguf")
    igual = dataclasses.replace(item, pacote=item.fornecedor)
    assert M.fornecedor_resumo(igual.fornecedor, igual.pacote) == item.fornecedor

    rec = tui.criar_console(width=100, height=40, record=True)
    tui.renderizar_detalhe_perfil(rec, igual, tui.ler_estado_local())
    texto = rec.export_text(clear=False)
    assert f"{M.ROTULO_FORNECEDOR}: {item.fornecedor}" in texto
    assert M.ROTULO_PUBLICADO_POR not in texto


def test_detalhe_do_perfil_mostra_publicador_quando_diverge():
    item = _perfil("qwen3.8-27b-32k")
    assert item.fornecedor == "Qwen (Alibaba Cloud)"
    assert item.pacote == "philbert440"
    rec = tui.criar_console(width=100, height=40, record=True)
    tui.renderizar_detalhe_perfil(rec, item, tui.ler_estado_local())
    texto = rec.export_text(clear=False)
    assert "Fornecedor: Qwen (Alibaba Cloud)" in texto
    assert "Publicado por: philbert440" in texto


@pytest.mark.parametrize("largura", [60, 80, 100, 120])
def test_lista_de_perfis_mostra_fornecedor_sem_estourar_a_largura(largura: int):
    rec = tui.criar_console(width=largura, height=60, record=True)
    tui.renderizar_perfis(rec, tui.ler_estado_local())
    texto = rec.export_text(clear=False)
    # A célula "Modelo" quebra no espaço quando a coluna fica estreita: comparar
    # no texto normalizado evita confundir quebra de linha com dado ausente.
    plano = " ".join(texto.split())

    assert "Thinking Machines Lab" in plano
    assert "…" not in texto
    for linha in texto.splitlines():
        assert len(linha) <= largura, f"linha estourou {largura} colunas: {linha!r}"
    # Cartões (<80) e 120+ têm espaço para o pacote; a tabela de 80/100 não.
    if largura < 80 or largura >= 120:
        assert "pesos Unsloth" in plano


def test_nome_curto_do_fornecedor_tira_o_parenteses_explicativo():
    assert M.fornecedor_curto("Qwen (Alibaba Cloud)") == "Qwen"
    assert M.fornecedor_curto("Meta Superintelligence Lab") == "Meta Superintelligence Lab"
    assert M.fornecedor_curto("") == ""


@pytest.mark.parametrize("largura", [100, 140, 160])
def test_tabela_de_modelos_compatíveis_mostra_fornecedor(largura: int):
    rec = tui.criar_console(width=largura, height=60, record=True)
    tui.renderizar_modelos_por_gpu(rec, "L4", tui.ler_estado_local())
    texto = rec.export_text(clear=False)
    plano = " ".join(texto.split())

    assert "Thinking Machines Lab" in plano
    for linha in texto.splitlines():
        assert len(linha) <= largura, f"linha estourou {largura} colunas: {linha!r}"
    if largura >= 160:
        assert M.COLUNA_FORNECEDOR in texto
        assert "pesos Unsloth" in plano


def test_cli_profiles_mostra_fornecedor_e_pacote():
    res = runner.invoke(cli.app, ["profiles"])
    assert res.exit_code == 0
    assert "Fornecedor: Thinking Machines Lab" in res.output
    assert "Pacote: Unsloth" in res.output
    assert "Fornecedor: Qwen (Alibaba Cloud)" in res.output
    assert "Pacote: philbert440" in res.output


def test_cli_models_mostra_fornecedor_e_pacote():
    res = runner.invoke(cli.app, ["models", "--gpu", "L4"])
    assert res.exit_code == 0
    assert "Fornecedor" in res.output
    assert "Pacote" in res.output
    assert "Unsloth" in res.output


def test_perfis_atomicchat_mostram_publicador_nas_duas_listas():
    estado = tui.ler_estado_local()
    ad = _perfil("inkling-2-mini-ad-q4k")
    assert ad.fornecedor == "Thinking Machines Lab"
    assert ad.pacote == "AtomicChat"
    assert ad.variante_curta == "AD-Q4_K • 4 bits"

    rec = tui.criar_console(width=140, height=60, record=True)
    tui.renderizar_perfis(rec, estado)
    plano = " ".join(rec.export_text(clear=False).split())
    assert "Thinking Machines Lab · pesos AtomicChat" in plano
    assert "Thinking Machines Lab · pesos Unsloth" in plano
