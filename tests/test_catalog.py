"""Testes do catálogo único de GPUs e modelos."""

from __future__ import annotations

import pytest

from byom import catalog, config


def test_catalogo_valido_e_vram_oficial_aproximada_no_colab():
    catalog.validate_catalog()
    gpus = {gpu.id: gpu for gpu in catalog.GPUS}
    assert {key: gpu.vram_gib for key, gpu in gpus.items()} == {
        "T4": 15,
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


def test_colab_settings_for_gpu_fecha_o_ciclo_com_resolve_gpu():
    for gpu in catalog.GPUS:
        gpu_cfg, high_mem = catalog.colab_settings_for_gpu(gpu.id)
        assert catalog.resolve_gpu(gpu_cfg, high_mem).id == gpu.id

    # O mapeamento catálogo -> config.toml não é identidade para as A100.
    assert catalog.colab_settings_for_gpu("A100-40") == ("A100", False)
    assert catalog.colab_settings_for_gpu("A100-80") == ("A100", True)
    assert catalog.colab_settings_for_gpu("L4") == ("L4", False)
    with pytest.raises(ValueError):
        catalog.colab_settings_for_gpu("GPU-INEXISTENTE")


def test_compatibilidade_com_margem_e_alternativas():
    # 24,0 de 24,0 GiB era aprovado antes da regra de folga; agora reprova.
    no_limite = catalog.check_fit("inkling-2-mini", "q4", "L4", 65536)
    assert no_limite.required_gib == pytest.approx(24.65)
    assert no_limite.available_gib == 24.0
    assert not no_limite.fits
    assert "margem" in no_limite.reason

    com_folga = catalog.check_fit("inkling-2-mini", "q4", "A100-40", 65536)
    assert com_folga.fits
    assert com_folga.headroom_gib == pytest.approx(2.0)

    # Em L4 a alternativa q2 (com visão) ainda cabe em 65.536; em T4 só o q2 cabe.
    assert "inkling-2-mini:q2" in no_limite.alternatives
    too_big = catalog.check_fit("inkling-2-mini", "q4", "T4", 32768)
    assert not too_big.fits
    assert "GiB" in too_big.reason
    assert too_big.alternatives == ("inkling-2-mini:q2",)

    with pytest.raises(ValueError):
        catalog.check_fit("modelo-inexistente", "q4", "L4")


def test_kv_cache_respeita_o_dtype_do_runtime():
    """32K de KV custam 2 GiB em f16 e 1 GiB em fp8."""
    assert catalog.kv_cache_gib(32768, kv_cache_dtype="f16", runtime="vllm") == 2.0
    assert catalog.kv_cache_gib(32768, kv_cache_dtype="fp8", runtime="vllm") == 1.0
    assert catalog.kv_cache_gib(262144, kv_cache_dtype="fp8", runtime="vllm") == 8.0
    # llama-cpp não recebe --cache-type-k/-v: fica em f16 mesmo se declarar fp8.
    assert catalog.kv_cache_gib(32768, kv_cache_dtype="fp8", runtime="llama-cpp") == 2.0
    assert catalog.effective_kv_dtype("llama-cpp", "fp8") == "f16"
    assert catalog.effective_kv_dtype("vllm", "fp8") == "fp8"
    # Sem declaração em vllm, o argv do worker usa o default fp8.
    assert catalog.effective_kv_dtype("vllm", None) == "fp8"


def test_perfil_vllm_256k_conta_o_kv_em_fp8():
    """17 pesos + 2 runtime + 8 de KV fp8 = 27 GiB (antes: 35 com KV f16)."""
    fit = catalog.check_fit(
        "qwen3.8-27b-awq", "q4-awq", "A100-80", 262144, kv_cache_dtype="fp8"
    )

    assert fit.required_gib == pytest.approx(27.0)
    assert fit.fits
    em_f16 = catalog.check_fit(
        "qwen3.8-27b-awq", "q4-awq", "A100-80", 262144, kv_cache_dtype="f16"
    )
    assert em_f16.required_gib == pytest.approx(35.0)


def test_perfil_llama_cpp_conta_o_kv_que_o_argv_usa():
    """llama-cpp cobra f16 e o argv não emite flag de KV: os dois batem."""
    import importlib.util

    from byom import cli

    fit = catalog.check_fit("qwen3.8-27b-gguf", "q4", "L4", 32768)
    variante = catalog.variant_by_id("qwen3.8-27b-gguf", "q4")
    assert fit.required_gib == pytest.approx(
        variante.memory_gib + variante.vision_gib + catalog.RUNTIME_RESERVE_GIB + 2.0
    )

    spec = importlib.util.spec_from_file_location(
        "byom_build_command_kv", cli._worker_dir() / "build_vllm_command.py"
    )
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    cfg = {
        "runtime": "llama-cpp",
        "model": "x",
        "served_model_name": "x",
        "max_model_len": 32768,
        "vllm_port": 8000,
        "worker": {"main_pattern": "x.gguf"},
    }
    argv = modulo.build_llama_args(cfg)
    assert "--kv-cache-dtype" not in argv
    assert "--cache-type-k" not in argv and "--cache-type-v" not in argv


def test_igualdade_no_limite_nao_aprova():
    """14,0 de required + 1,0 de folga = 15,0 de 15,0 não é aprovação."""
    fit = catalog.check_fit("gemma-4", "e2b-q4", "T4", 131_072)
    assert fit.required_gib == pytest.approx(14.0)
    assert fit.headroom_gib == pytest.approx(1.0)
    assert fit.required_gib + fit.headroom_gib == pytest.approx(fit.available_gib)
    assert fit.fits is False
    assert "margem" in fit.reason
    # Uma GPU com folga de verdade continua aprovando o mesmo contexto.
    assert catalog.check_fit("gemma-4", "e2b-q4", "L4", 131_072).fits is True


def test_folga_nomeada_e_proporcional_ao_vram():
    assert catalog.vram_headroom_gib(15) == pytest.approx(1.0)
    assert catalog.vram_headroom_gib(24) == pytest.approx(1.2)
    assert catalog.vram_headroom_gib(40) == pytest.approx(2.0)
    assert catalog.vram_headroom_gib(80) == pytest.approx(4.0)


def test_required_igual_ao_disponivel_reprova():
    """A folga é obrigatória: 100% da VRAM nunca é aprovado."""
    fit = catalog.check_fit("gemma-4", "e2b-q4", "T4", 147_456)  # 4 + 2 + 9 = 15
    assert fit.required_gib == pytest.approx(15.0)
    assert fit.available_gib == pytest.approx(15.0)
    assert not fit.fits
    assert "margem" in fit.reason


def test_t4_rejeita_muse_q2_e_qwen_q3_citando_os_componentes():
    muse = catalog.check_fit("muse-glimmer-30b", "q2", "T4", 32768)
    assert not muse.fits
    for termo in ("pesos", "runtime", "KV", "visão"):
        assert termo in muse.reason

    qwen = catalog.check_fit("qwen3.8-27b-gguf", "q3", "T4", 32768)
    assert not qwen.fits
    for termo in ("pesos", "runtime", "KV"):
        assert termo in qwen.reason
    assert "visão" not in qwen.reason


def test_todo_perfil_aplicado_cabe_com_margem():
    for profile in config.available_profiles():
        cfg = config.Config()
        config.apply_profile(cfg, profile.id)
        fit = config.check_compatibility(cfg)
        folga = fit.available_gib - fit.required_gib
        assert fit.fits, (profile.id, fit.reason)
        assert folga >= catalog.vram_headroom_gib(fit.available_gib), (
            profile.id,
            fit.required_gib,
            fit.available_gib,
        )


def test_modelos_grandes_sem_perfil_executavel():
    for model_id in ("minimax-m3.1-flash", "glm-5.3-flash", "deepseek-v4.1-flash"):
        model = catalog.model_by_id(model_id)
        assert all(not variant.available and variant.reason for variant in model.variants)
        result = catalog.check_fit(model_id, model.variants[0].id, "H100-80")
        assert not result.fits
        assert "NAO CABE" not in result.reason  # motivo legível, não rótulo de tabela


def test_saidas_dos_comandos():
    gpus = catalog.format_gpus((("p", "inkling-2-mini", "q4", 32768),))
    assert "T4 (Tesla T4) | 15 | nao | sim" in gpus
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
    # 65.536 servidos (com visão e folga) cabem em A100-40, não em L4;
    # 1.000.000 documentados não cabem em nenhuma das duas.
    assert catalog.check_fit("inkling-2-mini", "q4", "A100-40", 65_536).fits
    assert not catalog.check_fit("inkling-2-mini", "q4", "L4", 65_536).fits
    assert not catalog.check_fit("inkling-2-mini", "q4", "A100-40", 1_000_000).fits
    ref = catalog.profile_context("inkling-2-mini-q4-gguf")
    assert ref.context_served == 65_536
    assert ref.context_doc == 1_000_000
    with pytest.raises(ValueError):
        catalog.profile_context("perfil-inexistente")
