"""Testes da seleção de GPU e do filtro de modelos compatíveis."""

from __future__ import annotations

import builtins

import pytest

from byom import catalog

UNAVAILABLE = ("minimax-m3.1-flash", "glm-5.3-flash", "deepseek-v4.1-flash")


def test_gpu_valida_por_id():
    for gpu in catalog.GPUS:
        assert catalog.parse_gpu_choice(gpu.id).id == gpu.id
        assert catalog.parse_gpu_choice(gpu.id.lower()).id == gpu.id
        assert catalog.parse_gpu_choice(f"  {gpu.id}  ").id == gpu.id

    selection = catalog.compatible_variants("L4")
    assert selection.gpu_id == "L4"
    assert selection.compatible_count == len(selection.rows)
    assert f"GPU: L4" in catalog.format_compatible_models("L4")
    assert "Contexto doc | Contexto servido | Perfil" in catalog.format_compatible_models("L4")


def test_lista_numerada_e_gpu_valida_por_numero_na_interacao(monkeypatch):
    listed = catalog.format_gpus(numbered=True)
    assert "1. T4 (Tesla T4)" in listed
    assert "5. H100-80 (H100 80 GB)" in listed
    # A listagem padrão de `byom gpus` não muda.
    assert "T4 (Tesla T4) |" in catalog.format_gpus()

    monkeypatch.setattr(builtins, "input", lambda prompt="": "2")
    typed = input("GPU: ")
    selected = catalog.parse_gpu_choice(typed)
    assert selected.id == catalog.GPUS[1].id
    assert f"GPU: {selected.id}" in catalog.format_compatible_models(selected.id)


def test_gpu_invalida_com_mensagem_e_lista_de_ids_aceitos():
    for value in ("GPU-INEXISTENTE", "A100", "0", "99", ""):
        with pytest.raises(ValueError) as exc:
            catalog.parse_gpu_choice(value)
        message = str(exc.value)
        assert "Invalid GPU" in message
        for gpu in catalog.GPUS:
            assert gpu.id in message


def test_t4_filtra_modelos_grandes():
    selection = catalog.compatible_variants("T4")
    ids = {(row.model_id, row.variant_id) for row in selection.rows}
    assert ("gemma-4", "e2b-q4") in ids
    assert ("qwen-image-2.1", "q4km") in ids
    for excluded in (
        ("qwen3.8-27b-awq", "q4-awq"),
        ("gemma-4", "31b-q4"),
        ("muse-glimmer-30b", "q6"),
        ("qwen-image-2.1", "fp8-int8"),
    ):
        assert excluded not in ids
    assert selection.dropped_for_memory > 0

    text = catalog.format_compatible_models("T4")
    assert "Summary:" in text
    assert "discarded for insufficient VRAM" in text
    assert "gemma-4 | llama-cpp | 31b-q4" not in text

    # Variante sem perfil registrado: o contexto servido vem da própria variante.
    row = next(r for r in selection.rows if (r.model_id, r.variant_id) == ("gemma-4", "e2b-q4"))
    assert row.profile_id == ""
    assert row.context_served == catalog.variant_by_id("gemma-4", "e2b-q4").context
    assert "n/a (sem perfil)" in text


def test_a100_80_libera_variantes_que_nao_cabem_em_l4():
    l4 = {(row.model_id, row.variant_id) for row in catalog.compatible_variants("L4").rows}
    a100 = {(row.model_id, row.variant_id) for row in catalog.compatible_variants("A100-80").rows}
    for pair in (("gemma-4", "31b-q4"), ("muse-glimmer-30b", "q6")):
        assert pair not in l4
        assert pair in a100


def test_contexto_servido_e_perfil_da_gpu_na_filtragem():
    rows_40 = [
        row
        for row in catalog.compatible_variants("A100-40").rows
        if row.model_id == "qwen3.8-27b-awq"
    ]
    rows_80 = [
        row
        for row in catalog.compatible_variants("A100-80").rows
        if row.model_id == "qwen3.8-27b-awq"
    ]
    assert {row.profile_id for row in rows_40} == {"qwen3.8-27b-32k"}
    assert {row.profile_id for row in rows_80} == {"qwen3.8-27b-256k"}
    assert rows_40[0].context_served in catalog.served_contexts_for(
        "qwen3.8-27b-awq", "q4-awq"
    )
    assert rows_80[0].context_served in catalog.served_contexts_for(
        "qwen3.8-27b-awq", "q4-awq"
    )


def test_modelo_de_imagem_sem_kv_cache():
    variant = catalog.variant_by_id("qwen-image-2.1", "q4km")
    sentinel = catalog.profile_context("qwen-image-2.1-q4km").context_served
    fit = catalog.check_fit("qwen-image-2.1", "q4km", "T4", sentinel)
    assert fit.fits
    assert fit.required_gib == variant.memory_gib + catalog.RUNTIME_RESERVE_GIB

    row = next(
        row
        for row in catalog.compatible_variants("T4").rows
        if (row.model_id, row.variant_id) == ("qwen-image-2.1", "q4km")
    )
    assert row.context_served == sentinel
    assert "n/a (imagem)" in catalog.format_compatible_models("T4")


def test_modelos_indisponiveis_nunca_aparecem():
    for gpu in catalog.GPUS:
        selection = catalog.compatible_variants(gpu.id)
        assert all(row.model_id not in UNAVAILABLE for row in selection.rows)
        text = catalog.format_compatible_models(gpu.id)
        for model_id in UNAVAILABLE:
            assert f"{model_id} |" not in text


def test_selecao_nao_altera_configuracao(tmp_path, monkeypatch):
    config_dir = tmp_path / "byom"
    monkeypatch.setenv("BYOM_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "a100-40")

    typed = input("GPU: ")
    selected = catalog.parse_gpu_choice(typed)
    catalog.compatible_variants(selected.id)
    catalog.format_compatible_models(selected.id)

    assert selected.id == "A100-40"
    assert not config_dir.exists()
    assert not (tmp_path / "config.toml").exists()
    assert not (tmp_path / "state.json").exists()
