"""Testes do catálogo único de GPUs e modelos."""

from __future__ import annotations

import pytest

from byom import catalog


def test_catalogo_valido_e_vram_oficial_aproximada_no_colab():
    catalog.validate_catalog()
    gpus = {gpu.id: gpu for gpu in catalog.GPUS}
    assert {key: gpu.vram_gib for key, gpu in gpus.items()} == {
        "T4": 16,
        "L4": 24,
        "A100-40": 40,
        "A100-80": 80,
        "H100-80": 80,
    }
    assert gpus["T4"].bf16 is False
    assert all(gpu.vram_approximate for gpu in catalog.GPUS)
    assert "docs.cloud.google.com/compute/docs/gpus" in catalog.GPU_SOURCE


def test_resolve_gpu_respeita_high_mem():
    assert catalog.resolve_gpu("A100", False).id == "A100-40"
    assert catalog.resolve_gpu("A100", True).id == "A100-80"
    assert catalog.resolve_gpu("l4").id == "L4"
    with pytest.raises(ValueError):
        catalog.resolve_gpu("GPU-INEXISTENTE")


def test_compatibilidade_com_margem_e_alternativas():
    fit = catalog.check_fit("inkling-2-mini", "q4", "L4", 65536)
    assert fit.fits
    assert fit.required_gib == 24.0
    assert fit.available_gib == 24.0

    # A alternativa q2 serve 32768; com 65536 ela também não caberia em T4.
    too_big = catalog.check_fit("inkling-2-mini", "q4", "T4", 32768)
    assert not too_big.fits
    assert "tam" in too_big.reason or "GiB" in too_big.reason
    assert "inkling-2-mini:q2" in too_big.alternatives

    with pytest.raises(ValueError):
        catalog.check_fit("modelo-inexistente", "q4", "L4")


def test_modelos_grandes_sem_perfil_executavel():
    for model_id in ("minimax-m3.1-flash", "glm-5.3-flash", "deepseek-v4.1-flash"):
        model = catalog.model_by_id(model_id)
        assert all(not variant.available and variant.reason for variant in model.variants)
        result = catalog.check_fit(model_id, model.variants[0].id, "H100-80")
        assert not result.fits
        assert "NAO CABE" not in result.reason  # motivo legível, não rótulo de tabela


def test_saidas_dos_comandos():
    gpus = catalog.format_gpus((("p", "inkling-2-mini", "q4", 32768),))
    assert "T4 (Tesla T4) | 16 | nao | sim" in gpus
    assert "Fonte VRAM:" in gpus
    models = catalog.format_models()
    assert "qwen-image-2.1 | image | q4km" in models
    assert "NAO CABE EM GPU UNICA DO COLAB" in models
    assert "Contexto doc | Contexto servido" in models
    # Os dois números aparecem distintamente: documento à esquerda, perfil à direita.
    assert "inkling-2-mini | llama-cpp | q4" in models
    assert "1000000 | 65536" in models
    assert "qwen3.8-27b-awq | vllm | q4-awq" in models
    assert "262144 | 32768, 262144" in models
    assert "muse-glimmer-30b | llama-cpp | q4" in models
    assert "131072 | 65536" in models


def test_reasoning_e_propriedade_do_modelo_com_fonte():
    assert catalog.model_by_id("inkling-2-mini").reasoning_effort == "high"
    assert catalog.model_by_id("qwen3.8-27b-awq").reasoning_effort == "xhigh"
    assert catalog.model_by_id("qwen3.8-27b-gguf").reasoning_effort == "xhigh"
    assert catalog.model_by_id("muse-glimmer-30b").reasoning_effort == "high"
    assert catalog.model_by_id("gemma-4").reasoning_effort is None
    assert catalog.model_by_id("qwen-image-2.1").reasoning_effort is None
    for model in catalog.MODELS:
        if model.reasoning_effort is not None:
            assert model.reasoning_source
            assert model.reasoning_effort in catalog.REASONING_EFFORTS
    assert "2026-11-23" in catalog.REASONING_SOURCE_INKLING
    assert "chat_template.jinja" in catalog.REASONING_SOURCE_INKLING
    assert "unsloth.ai/docs/models/qwen3.8.md" in catalog.REASONING_SOURCE_QWEN38
    assert "unsloth.ai/docs/models/muse-glimmer.md" in catalog.REASONING_SOURCE_MUSE
    assert "NÃO define" in catalog.REASONING_SOURCE_GEMMA


def test_thinking_policy_e_chave_explicita_so_no_gemma():
    gemma = catalog.model_by_id("gemma-4")
    policy = gemma.thinking
    assert policy is not None
    assert policy == catalog.ThinkingPolicy()
    for model in catalog.MODELS:
        if model.id != "gemma-4":
            assert model.thinking is None
    assert "<|think|>" in policy.system_token
    assert "gemma-4.md" in policy.source


def test_padroes_de_imagem_e_equivalencia_documentada():
    image = catalog.model_by_id("qwen-image-2.1")
    assert image.generation_defaults == {"steps": 25, "guidance": 1.0}
    assert "qwen-image-2.1.md" in image.generation_source
    assert "sample_params.sample_steps" in image.generation_source
    assert "sample_params.guidance.txt_cfg" in image.generation_source
    for model in catalog.MODELS:
        if model.id != "qwen-image-2.1":
            assert not model.generation_defaults


def test_contexto_documentado_servido_e_motivo():
    doc = {
        ("inkling-2-mini", "q4"): 1_000_000,
        ("qwen3.8-27b-awq", "q4-awq"): 262_144,
        ("gemma-4", "31b-q4"): 262_144,
        ("muse-glimmer-30b", "q4"): 131_072,
    }
    for (model_id, variant_id), value in doc.items():
        variant = catalog.variant_by_id(model_id, variant_id)
        assert variant.context_doc == value
        assert variant.context <= value
        if (model_id, variant_id) == ("qwen3.8-27b-awq", "q4-awq"):
            # O perfil 256k serve o contexto máximo documentado; a redução
            # fica explícita em PROFILE_CONTEXTS, não na variante.
            continue
        assert variant.context < value
        assert variant.context_reason

    served = {item.profile_id: item.context_served for item in catalog.PROFILE_CONTEXTS}
    assert served["inkling-2-mini-q4-gguf"] == 65_536
    assert served["qwen3.8-27b-32k"] == 32_768
    assert served["qwen3.8-27b-256k"] == 262_144
    assert all(item.reason for item in catalog.PROFILE_CONTEXTS)
    assert len(served) == 15


def test_compatibilidade_continua_usando_contexto_servido():
    # 65.536 servidos cabem em L4; 1.000.000 documentados não caberiam.
    assert catalog.check_fit("inkling-2-mini", "q4", "L4", 65_536).fits
    assert not catalog.check_fit("inkling-2-mini", "q4", "L4", 1_000_000).fits
    ref = catalog.profile_context("inkling-2-mini-q4-gguf")
    assert ref.context_served == 65_536
    assert ref.context_doc == 1_000_000
    with pytest.raises(ValueError):
        catalog.profile_context("perfil-inexistente")
