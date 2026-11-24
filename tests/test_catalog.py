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
    # 18 + 0,65 de visão + 2 de runtime + 1,08 de KV em 64K (pela arquitetura)
    # = 21,73: com a folga de 1,2 GiB da L4 dá 22,93 <= 24 e a A100-40 deixou
    # de ser necessária para o q4 — era 24,65 com o heurístico de 2 GiB/32K.
    com_folga = catalog.check_fit("inkling-2-mini", "q4", "L4", 65_536)
    assert com_folga.required_gib == pytest.approx(21.73)
    assert com_folga.available_gib == 24.0
    assert com_folga.fits
    assert com_folga.headroom_gib == pytest.approx(1.2)

    # Quem não cabe na L4 é o q8: 28,73 GiB de pesos+KV contra 24 GiB.
    sem_folga = catalog.check_fit("inkling-2-mini", "q8", "L4", 131_072)
    assert sem_folga.required_gib == pytest.approx(28.73)
    assert not sem_folga.fits
    assert "margem" in sem_folga.reason
    assert "arquitetura" in sem_folga.reason
    assert "inkling-2-mini:q4" in sem_folga.alternatives

    # Em T4 o q4 não cabe nem em 32K; as alternativas são as variantes menores
    # que caberiam no mesmo contexto.
    too_big = catalog.check_fit("inkling-2-mini", "q4", "T4", 32768)
    assert not too_big.fits
    assert "GiB" in too_big.reason
    assert too_big.alternatives == ("inkling-2-mini:q2", "inkling-2-mini:ad-iq2xxs")

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
    assert "Modelo | Fornecedor | Runtime |" in models
    assert "qwen-image-2.1 | Qwen (Alibaba Cloud) | image | q4km" in models
    assert "NAO CABE EM GPU UNICA DO COLAB" in models
    assert "Contexto doc | Contexto servido" in models
    # Os dois números aparecem distintamente: documento à esquerda, perfil à direita.
    assert "inkling-2-mini | Thinking Machines Lab | llama-cpp | q4" in models
    assert "1000000 | 65536" in models
    assert "qwen3.8-27b-awq | Qwen (Alibaba Cloud) | vllm | q4-awq" in models
    assert "262144 | 32768, 262144" in models
    assert "muse-glimmer-30b | Meta Superintelligence Lab | llama-cpp | q4" in models
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
    assert served["inkling-2-mini-ad-q6k"] == 131_072
    assert served["inkling-2-mini-ad-q4k"] == 32_768
    assert served["inkling-2-mini-ad-iq2xxs"] == 32_768
    assert all(item.reason for item in catalog.PROFILE_CONTEXTS)
    assert len(served) == 18


def test_compatibilidade_continua_usando_contexto_servido():
    # 65.536 servidos do q4 (21,73 com o KV da arquitetura) cabem na L4; a T4
    # não comporta nem 32K do mesmo arquivo.
    assert catalog.check_fit("inkling-2-mini", "q4", "L4", 65_536).fits
    assert not catalog.check_fit("inkling-2-mini", "q4", "T4", 65_536).fits
    assert not catalog.check_fit("inkling-2-mini", "q4", "T4", 32_768).fits
    # 1M continua sendo checado contra a GPU: 18 + 0,65 + 2 + 15,34 = 35,99
    # cabe na A100-40 por 2,01 GiB além da folga e não cabe na L4.
    assert catalog.check_fit("inkling-2-mini", "q4", "A100-40", 1_000_000).fits
    assert not catalog.check_fit("inkling-2-mini", "q4", "L4", 1_000_000).fits
    ref = catalog.profile_context("inkling-2-mini-q4-gguf")
    assert ref.context_served == 65_536
    assert ref.context_doc == 1_000_000
    with pytest.raises(ValueError):
        catalog.profile_context("perfil-inexistente")


# ---------------------------------------------------------------------------
# Variantes AD da AtomicChat (mesmo modelo, publicador e receita diferentes)
# ---------------------------------------------------------------------------


def test_variantes_atomicchat_entram_no_modelo_existente_do_inkling():
    model = catalog.model_by_id("inkling-2-mini")
    assert {variant.id for variant in model.variants} == {
        "q2",
        "q4",
        "q8",
        "ad-q6k",
        "ad-q4k",
        "ad-iq2xxs",
    }
    # Um ModelEntry só: as variantes novas não criam família duplicada.
    assert [entry.id for entry in catalog.MODELS].count("inkling-2-mini") == 1


def test_numeros_das_variantes_atomicchat_vem_do_tree_do_hf():
    # Bytes do tree da API do HF em 2026-11-24, convertidos para GiB (2^30):
    # AD-Q6_K 23.787.472.864 B = 22,15 GiB; AD-Q4_K 16.525.559.776 B = 15,39 GiB;
    # AD-IQ2_XXS 10.425.495.136 B = 9,71 GiB; mmproj-F16 639.348.448 B = 0,60 GiB.
    esperado = {
        "ad-q6k": (22.15, 131_072, "A100-40"),
        "ad-q4k": (15.39, 32_768, "L4"),
        "ad-iq2xxs": (9.71, 32_768, "T4"),
    }
    for variant_id, (gib, context, gpu) in esperado.items():
        variant = catalog.variant_by_id("inkling-2-mini", variant_id)
        assert variant.memory_gib == gib
        assert variant.context == context
        assert variant.min_gpu == gpu
        assert variant.context_doc == 1_000_000
        assert variant.context_reason
        assert variant.vision_gib == 0.65
        assert all(not variant.needs_custom_build for variant in (variant,))
        notas = " ".join(variant.notes)
        assert "AtomicChat/Inkling-2-Mini-GGUF" in notas
        assert "2026-11-24" in notas
        # A pegadinha do layout antigo (PR #32695) não pode aparecer como opção.
        assert "Q4_K_M" not in notas or "não carrega" in notas


def test_variantes_atomicchat_nao_oferecem_os_presets_do_layout_antigo():
    model = catalog.model_by_id("inkling-2-mini")
    rotulos = " ".join(variant.label for variant in model.variants)
    assert "Q4_K_M" not in rotulos
    assert "Q5_K_M" not in rotulos
    ad = [variant for variant in model.variants if variant.id.startswith("ad-")]
    assert len(ad) == 3
    assert all(variant.label.startswith("AD-") for variant in ad)


def test_variante_ad_iq2xxs_cabe_na_t4_em_32k_com_o_kv_da_arquitetura():
    # 9,71 + 0,65 visão + 2 runtime + 0,58 de KV (arquitetura) = 12,94;
    # com a folga de 1,0 GiB dá 13,94 <= 15 GiB da T4.
    em_32k = catalog.check_fit("inkling-2-mini", "ad-iq2xxs", "T4", 32_768)
    assert em_32k.required_gib == pytest.approx(12.94)
    assert em_32k.fits
    # 128K no mesmo arquivo já estoura: 9,71 + 0,65 + 2 + 2,08 = 14,44, mais
    # a folga de 1,0 = 15,44 > 15 GiB.
    assert not catalog.check_fit("inkling-2-mini", "ad-iq2xxs", "T4", 131_072).fits
    assert catalog.check_fit("inkling-2-mini", "ad-iq2xxs", "L4", 32_768).fits


def test_variante_ad_q6k_nao_cabe_em_l4_nem_na_t4():
    assert not catalog.check_fit("inkling-2-mini", "ad-q6k", "T4", 32_768).fits
    assert not catalog.check_fit("inkling-2-mini", "ad-q6k", "L4", 32_768).fits
    assert catalog.check_fit("inkling-2-mini", "ad-q6k", "A100-40", 131_072).fits


# ---------------------------------------------------------------------------
# KV por arquitetura (`KvProfile`/`ModelEntry.kv_shape`)
# ---------------------------------------------------------------------------

def _entrada_sintetica(shape: catalog.KvProfile | None) -> catalog.ModelEntry:
    """Entrada mínima do catálogo, só para exercitar a conta do KV."""
    return catalog.ModelEntry(
        "sintetico",
        "Sintético",
        "llama-cpp",
        (catalog.Variant("v", "v", 1.0, 1024, "T4"),),
        "tests",
        kv_shape=shape,
    )


def _shape_hibrido() -> catalog.KvProfile:
    """Forma de atenção híbrida: 4 completas + 20 locais de 1024 tokens."""
    return catalog.KvProfile(
        layers_full=4,
        layers_local=20,
        kv_heads=8,
        head_dim=128,
        window=1024,
        bytes_per_elem=2,
        source="https://exemplo/config.json",
        checked_at="2026-11-24",
    )


def test_kv_por_arquitetura_so_cresce_nas_camadas_completas():
    shape = _shape_hibrido()

    # 2 (K,V) × 8 heads × 128 dim × 2 B = 4 KiB por token em UMA camada completa;
    # com 4 camadas completas são 16 KiB por token.
    assert shape.bytes_per_token(2) == 4 * 1024
    assert shape.bytes_per_token(2) * shape.layers_full == 16 * 1024
    # 20 camadas locais × 1024 tokens = 80 MiB fixos (não crescem com o ctx).
    apenas_locais = (
        shape.layers_local * shape.bytes_per_token(2) * shape.window / catalog.GIB
    )
    assert apenas_locais == pytest.approx(80 / 1024)

    assert shape.cache_gib(16_384) == pytest.approx(0.33, abs=0.005)
    assert shape.cache_gib(32_768) == pytest.approx(0.58, abs=0.005)
    assert shape.cache_gib(131_072) == pytest.approx(2.08, abs=0.005)


def test_kv_por_arquitetura_arredonda_para_cima_e_respeita_o_dtype():
    shape = _shape_hibrido()
    # 0,328125 GiB brutos: o orçamento nunca arredonda para baixo.
    assert shape.cache_gib(16_384) == 0.33

    modelo = _entrada_sintetica(shape)
    f16 = catalog.kv_cache_gib(32_768, modelo, runtime="llama-cpp")
    fp8 = catalog.kv_cache_gib(32_768, modelo, runtime="vllm", kv_cache_dtype="fp8")
    assert f16 == 0.58
    assert fp8 == 0.29  # metade do KV, arredondado para cima


def test_modelo_sem_kv_shape_continua_no_heurístico():
    modelo = _entrada_sintetica(None)
    assert catalog.kv_cache_gib(32_768, modelo, runtime="llama-cpp") == 2.0
    assert catalog.kv_cache_gib(8_192, modelo, runtime="llama-cpp") == 1.0  # piso
    assert catalog.required_vram_gib(
        catalog.Variant("v", "v", 9.71, 32_768, "T4", vision_gib=0.65),
        32_768,
        "llama-cpp",
        model=modelo,
    ) == pytest.approx(14.36)


def test_required_vram_usa_o_kv_da_arquitetura_quando_existe():
    modelo = _entrada_sintetica(_shape_hibrido())
    variante = catalog.Variant("v", "v", 9.71, 32_768, "T4", vision_gib=0.65)
    assert catalog.required_vram_gib(
        variante, 32_768, "llama-cpp", model=modelo
    ) == pytest.approx(12.94)


def test_kv_shape_invalido_falha_na_validacao():
    with pytest.raises(ValueError, match="sem fonte datada"):
        catalog._check_kv_shape(
            "x", catalog.KvProfile(4, 20, 8, 128, 1024, source="", checked_at="")
        )
    with pytest.raises(ValueError, match="sem janela"):
        catalog._check_kv_shape(
            "x", catalog.KvProfile(4, 20, 8, 128, -1, source="u", checked_at="2026-11-24")
        )
    with pytest.raises(ValueError, match="sem camadas"):
        catalog._check_kv_shape(
            "x", catalog.KvProfile(0, 0, 8, 128, 1024, source="u", checked_at="2026-11-24")
        )
    catalog._check_kv_shape("x", _shape_hibrido())


def test_kv_shape_note_diz_a_procedencia_do_numero():
    assert "heurístico" in catalog.kv_shape_note(_entrada_sintetica(None))
    nota = catalog.kv_shape_note(_entrada_sintetica(_shape_hibrido()))
    assert "arquitetura" in nota and "4 camadas" in nota
    assert "2026-11-24" in nota


# ---------------------------------------------------------------------------
# Inkling 2 Mini: KV lido do config.json e perfis revalidados
# ---------------------------------------------------------------------------

INKLING_CONFIG_URL = (
    "https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json"
)


def test_kv_do_inkling_vem_do_config_json_e_bate_com_a_conta_manual():
    modelo = catalog.model_by_id("inkling-2-mini")
    shape = modelo.kv_shape
    assert shape == catalog.inkling_kv_profile()
    assert (
        shape.layers_full,
        shape.layers_local,
        shape.kv_heads,
        shape.head_dim,
        shape.window,
    ) == (4, 20, 8, 128, 1024)
    assert shape.layers_total == 24  # num_hidden_layers do text_config
    assert shape.bytes_per_elem == 2  # torch_dtype bfloat16
    assert INKLING_CONFIG_URL in shape.source
    assert shape.checked_at == "2026-11-24"

    # A conta à mão: 16 KiB/token nas 4 completas + 80 MiB fixos das 20 locais.
    assert catalog.kv_cache_gib(16_384, modelo, runtime="llama-cpp") == pytest.approx(0.33)
    assert catalog.kv_cache_gib(32_768, modelo, runtime="llama-cpp") == pytest.approx(0.58)
    assert catalog.kv_cache_gib(131_072, modelo, runtime="llama-cpp") == pytest.approx(2.08)
    # O 1M do catálogo é 1.000.000 tokens (context_doc); 1.048.576 é o
    # `model_max_length` do config.json.
    assert catalog.kv_cache_gib(1_000_000, modelo, runtime="llama-cpp") == pytest.approx(15.34)
    assert catalog.kv_cache_gib(1_048_576, modelo, runtime="llama-cpp") == pytest.approx(16.08)


def test_modelos_sem_kv_shape_seguem_no_heurístico_de_2_gib_por_32k():
    for model_id in ("qwen3.8-27b-gguf", "gemma-4", "muse-glimmer-30b"):
        modelo = catalog.model_by_id(model_id)
        assert modelo.kv_shape is None, model_id
        assert catalog.kv_cache_gib(32_768, modelo, runtime="llama-cpp") == 2.0
    # vLLM segue o fp8 do argv: metade do KV, ainda pelo heurístico.
    awq = catalog.model_by_id("qwen3.8-27b-awq")
    assert awq.kv_shape is None
    assert catalog.kv_cache_gib(32_768, awq, runtime="vllm", kv_cache_dtype="fp8") == 1.0
    # O heurístico cobra o mesmo em 128K para qualquer arquitetura: 8 GiB.
    assert catalog.kv_cache_gib(
        131_072, catalog.model_by_id("gemma-4"), runtime="llama-cpp"
    ) == 8.0


def test_kv_da_arquitetura_muda_o_veredito_do_inkling_em_32k():
    """Antes: 2 GiB de KV em 32K reprovavam o AD-IQ2_XXS na T4. Agora cabe."""
    antes = 9.71 + 0.65 + catalog.RUNTIME_RESERVE_GIB + 2.0  # heurístico antigo
    assert antes + catalog.vram_headroom_gib(15) == pytest.approx(15.36)
    assert antes + catalog.vram_headroom_gib(15) > 15  # reprovava

    depois = catalog.check_fit("inkling-2-mini", "ad-iq2xxs", "T4", 32_768)
    assert depois.required_gib == pytest.approx(12.94)
    assert depois.fits is True

    perfil = config.load_profile("inkling-2-mini-ad-iq2xxs")
    assert perfil.settings["max_model_len"] == 32_768
    assert perfil.settings["gpu"] == "T4"
    assert catalog.profile_context(perfil.id).context_served == 32_768


def test_perfis_de_inkling_ficam_na_gpu_mais_barata_que_cabe():
    perfis = [
        perfil
        for perfil in config.available_profiles()
        if perfil.catalog_model == "inkling-2-mini"
    ]
    assert {perfil.id for perfil in perfis} == {
        "inkling-2-mini-iq2-gguf",
        "inkling-2-mini-q4-gguf",
        "inkling-2-mini-q8-gguf",
        "inkling-2-mini-ad-q6k",
        "inkling-2-mini-ad-q4k",
        "inkling-2-mini-ad-iq2xxs",
    }
    for perfil in perfis:
        contexto = int(perfil.settings["max_model_len"])
        atual = catalog.resolve_gpu(
            str(perfil.settings["gpu"]), bool(perfil.settings.get("gpu_high_mem"))
        )
        mais_barata = catalog.min_gpu_for(
            perfil.catalog_model, perfil.catalog_variant, contexto
        )
        assert atual.id == mais_barata, (perfil.id, atual.id, mais_barata)
        assert catalog.check_fit(
            perfil.catalog_model, perfil.catalog_variant, atual.id, contexto
        ).fits


# ---------------------------------------------------------------------------
# Qwen3.8-27B híbrido: camadas lineares não têm KV por token
# ---------------------------------------------------------------------------

QWEN38_CONFIG_URL = "https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json"


def _shape_linear() -> catalog.KvProfile:
    """64 camadas com `full_attention_interval` 4: 16 completas + 48 lineares.

    `text_config` do config.json oficial (2026-11-24): `num_hidden_layers` 64,
    `layout_types` com `full_attention` a cada 4 camadas,
    `num_key_value_heads` 4, `head_dim` 256, `dtype` bfloat16.
    """
    return catalog.KvProfile(
        16,
        48,
        4,
        256,
        0,
        bytes_per_elem=2,
        source=QWEN38_CONFIG_URL,
        checked_at="2026-11-24",
    )


def test_kv_de_camada_linear_cresce_so_nas_camadas_de_atencao_completa():
    shape = _shape_linear()

    # 2 (K,V) × 4 heads × 256 dim × 2 B = 4 KiB por token em UMA camada
    # completa; nas 16 completas, 64 KiB por token (o número dos docs).
    assert shape.bytes_per_token(2) == 4 * 1024
    assert shape.bytes_per_token(2) * shape.layers_full == 64 * 1024
    assert shape.cache_gib(32_768) == pytest.approx(2.0)
    assert shape.cache_gib(131_072) == pytest.approx(8.0)
    assert shape.cache_gib(262_144) == pytest.approx(16.0)
    # As 48 lineares não somam nada: o estado recorrente é fixo.
    assert shape.cache_gib(4_096) == pytest.approx(0.25)


def test_kv_shape_com_camada_linear_aceita_janela_zero():
    """Janela 0 é a marca de camada linear; janela negativa continua inválida."""
    catalog._check_kv_shape("ternary-bonsai-2.5-27b", _shape_linear())

    with pytest.raises(ValueError, match="sem janela"):
        catalog._check_kv_shape(
            "x", catalog.KvProfile(4, 20, 8, 128, -1, source="u", checked_at="2026-11-24")
        )


def test_kv_shape_note_de_camada_linear_nao_fala_em_janela_de_zero():
    nota = catalog.kv_shape_note(_entrada_sintetica(_shape_linear()))

    assert "16 camadas de atenção completa" in nota
    assert "48 lineares" in nota
    assert "janela de 0" not in nota
    assert QWEN38_CONFIG_URL in nota
