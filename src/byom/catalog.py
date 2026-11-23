"""Catálogo estático e validado de GPUs do Colab e de modelos Unsloth.

Fonte única de números: este módulo. Nenhum outro arquivo do pacote deve
repetir valores de VRAM, memória de quantização, contexto, reasoning_effort
ou padrões de geração de imagem. Os valores de VRAM são de documentação
oficial do Google Cloud (Compute Engine), não de uma tabela publicada pelo
Colab; por isso `vram_approximate` é True.

`reasoning_effort` é uma propriedade do MODELO, não um padrão global: a página
de cada modelo determina o valor (ou não determina nenhum, como em Gemma 4).
`Variant.context` é o contexto SERVIDO pelo perfil; `Variant.context_doc` é o
máximo documentado pela página do modelo. A diferença é uma margem de memória
explicitamente registrada em `PROFILE_CONTEXTS`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

RUNTIMES = ("vllm", "llama-cpp", "image")

#: União dos valores documentados pelas páginas conferidas em 2026-11-23.
#: Inkling: blog HF + chat_template.jinja; Qwen3.8 e Muse: páginas Unsloth.
#: String livre (inclusive número) NÃO é aceita nesta fronteira.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

DOCS_CHECKED_ON = "2026-11-23"

REASONING_SOURCE_INKLING = (
    "https://huggingface.co/blog/inkling2mini e "
    "https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/chat_template.jinja "
    "(2026-11-23): template default high (0.9); escala none/minimal/low/medium/"
    "high/xhigh/max; o uso de xhigh na avaliação veio do briefing e NÃO foi "
    "localizado como linha explícita nas páginas consultadas"
)
REASONING_SOURCE_QWEN38 = (
    "https://unsloth.ai/docs/models/qwen3.8.md (2026-11-23): "
    "'xhigh (default)' na lista de reasoning_effort e 'xhigh reasoning' nos benchmarks"
)
REASONING_SOURCE_MUSE = (
    "https://unsloth.ai/docs/models/muse-glimmer.md (2026-11-23): "
    "'Thinking Settings' com low/medium/high/xhigh e tabela de benchmarks "
    "'Muse Glimmer-30B High Reasoning'"
)
REASONING_SOURCE_GEMMA = (
    "https://unsloth.ai/docs/models/gemma-4.md (2026-11-23): a página NÃO define "
    "reasoning_effort; define <|think|> no system prompt e enable_thinking"
)
GENERATION_SOURCE_IMAGE = (
    "https://unsloth.ai/docs/models/qwen-image-2.1.md (2026-11-23): "
    "'Defaults: 25 steps, guidance 1'. Equivalência com sd-server em "
    "https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md "
    "(2026-11-23): /v1/images/generations aceita sd_cpp_extra_args dentro de prompt; "
    "steps -> sample_params.sample_steps, guidance -> sample_params.guidance.txt_cfg. "
    "A mesma pagina sugere --steps 20 --cfg-scale 6.0 ao sd-cli; adotamos os "
    "defaults documentados da API Unsloth (25/1) conforme o briefing"
)

# Margens determinísticas (heurística de engenharia, não valor de documentação):
# 2 GiB para runtime/processo e 2 GiB por 32K de contexto (mínimo 1 GiB).
RUNTIME_RESERVE_GIB = 2.0
KV_CACHE_GIB_PER_32K = 2.0
MIN_KV_CACHE_GIB = 1.0

GPU_SOURCE = (
    "Google Cloud, Compute Engine > GPU machine types > General comparison chart "
    "(docs.cloud.google.com/compute/docs/gpus), consultado em 2026-11-23"
)


@dataclass(frozen=True)
class Gpu:
    id: str
    label: str
    vram_gib: int
    bf16: bool
    colab_names: tuple[str, ...]
    vram_source: str
    vram_approximate: bool = True
    note: str = ""


@dataclass(frozen=True)
class ThinkingPolicy:
    """Transformação de prompt exigida pela página do Gemma 4 (2026-11-23).

    A chave explícita no catálogo — e não uma comparação do nome do modelo —
    é o que faz o gateway injetar o token e limpar o histórico.
    """

    system_token: str = "<|think|>"
    thought_open: str = "<|channel>thought"
    thought_close: str = "<channel|>"
    strip_history_thought: bool = True
    source: str = "https://unsloth.ai/docs/models/gemma-4.md (2026-11-23)"


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    memory_gib: float
    context: int  # contexto SERVIDO pelo perfil (usado no orçamento/localização)
    min_gpu: str
    context_doc: int = 0  # máximo documentado; 0 = não documentado nesta checagem
    available: bool = True
    reason: str = ""
    needs_custom_build: bool = False
    notes: tuple[str, ...] = ()
    context_reason: str = ""


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    runtime: str
    variants: tuple[Variant, ...]
    source: str
    notes: tuple[str, ...] = ()
    sampling: dict[str, float | int] = field(default_factory=dict)
    #: None = a página do modelo não documenta reasoning_effort.
    reasoning_effort: str | None = None
    reasoning_source: str = ""
    thinking: ThinkingPolicy | None = None
    generation_defaults: dict[str, int | float] = field(default_factory=dict)
    generation_source: str = ""


@dataclass(frozen=True)
class ProfileContext:
    """Contexto servido por UM perfil, com o motivo da redução documentada."""

    profile_id: str
    model_id: str
    variant_id: str
    context_doc: int
    context_served: int
    gpu_id: str
    reason: str


GPUS: tuple[Gpu, ...] = (
    Gpu("T4", "Tesla T4", 16, False, ("T4",), GPU_SOURCE,
        note="Sem bf16; geração de imagem usa fp16."),
    Gpu("L4", "L4", 24, True, ("L4",), GPU_SOURCE),
    Gpu("A100-40", "A100 40 GB", 40, True, ("A100",), GPU_SOURCE,
        note="No CLI do Colab, usar gpu=A100 com high_mem=false."),
    Gpu("A100-80", "A100 80 GB", 80, True, ("A100",), GPU_SOURCE,
        note="No CLI do Colab, usar gpu=A100 com high_mem=true."),
    Gpu("H100-80", "H100 80 GB", 80, True, ("H100",), GPU_SOURCE),
)

COLAB_GPU_SOURCE = (
    "FAQ oficial do Colab (research.google.com/colaboratory/faq.html, "
    "consultado em 2026-11-23): tipos de GPU variam com o tempo; a FAQ não "
    "publica uma tabela de VRAM."
)

INKLING_SOURCE = "https://unsloth.ai/docs/models/inkling-2-mini.md"
QWEN38_SOURCE = "https://unsloth.ai/docs/models/qwen3.8.md"
GEMMA4_SOURCE = "https://unsloth.ai/docs/models/gemma-4.md"
MUSE_SOURCE = "https://unsloth.ai/docs/models/muse-glimmer.md"
IMAGE_SOURCE = "https://unsloth.ai/docs/models/qwen-image-2.1.md"
SDCPP_SOURCE = (
    "https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md"
)

MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        "inkling-2-mini", "Inkling 2 Mini", "llama-cpp",
        (
            Variant("q2", "UD-IQ2_XXS (2 bit)", 11.0, 32_768, "T4",
                    context_doc=1_000_000,
                    needs_custom_build=True,
                    context_reason="1M cabe apenas com KV muito alto; perfil T4 usa 32K.",
                    notes=("10-11 GB em 2 bit; build loopado obrigatório nos arquivos da raiz.",)),
            Variant("q4", "UD-Q4_K_XL (4 bit)", 18.0, 65_536, "L4",
                    context_doc=1_000_000,
                    needs_custom_build=True,
                    context_reason="18 GiB pesos + 2 GiB runtime + 4 GiB KV = 24 GiB em L4.",
                    notes=("17,6 GB recomendados; mmproj-F16.gguf adiciona 639 MB.",)),
            Variant("q8", "UD-Q8_K_XL (8 bit)", 24.0, 131_072, "A100-40",
                    context_doc=1_000_000,
                    context_reason="24 GiB de pesos deixam margem para 8 GiB de KV na A100-40.",
                    notes=("Faixa de 8 bit: 24 GB.",)),
        ),
        INKLING_SOURCE,
        notes=(
            "26,4B MoE (5,4B ativos), visão, 1M de contexto.",
            "Requer llama.cpp mainline b12611+ com os arquivos Mainline_Support; "
            "builds anteriores perdem os scales NVFP4 nas passadas 2 e 3 do loop.",
            "Sampling: temperature 1.0, top_p 1.0, top_k 0, min_p 0.",
        ),
        sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
        reasoning_effort="high",
        reasoning_source=REASONING_SOURCE_INKLING,
    ),
    ModelEntry(
        "qwen3.8-27b-awq", "Qwen3.8 27B (AWQ, vLLM)", "vllm",
        (
            Variant("q4-awq", "W4A16-AWQ", 17.0, 262_144, "L4",
                    context_doc=262_144,
                    notes=("Faixa 4 bit da Unsloth: 16-19 GB; MTP pede 1-2 GB extras.",
                           "Dois perfis servem esta variante: 32K (A100-40) e 256K (A100-80).",)),
        ),
        QWEN38_SOURCE,
        notes=("Checkpoint atual do repo: philbert440/Qwen3.8-27B-W4A16-AWQ.",),
        sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        reasoning_effort="xhigh",
        reasoning_source=REASONING_SOURCE_QWEN38,
    ),
    ModelEntry(
        "qwen3.8-27b-gguf", "Qwen3.8 27B (GGUF)", "llama-cpp",
        (
            Variant("q3", "UD-Q3_K_XL (3 bit)", 12.0, 32_768, "T4",
                    context_doc=262_144,
                    context_reason="12 GiB de pesos em T4: 2 GiB runtime + 2 GiB KV em 32K."),
            Variant("q4", "UD-Q4_K_XL (4 bit)", 17.0, 32_768, "L4",
                    context_doc=262_144,
                    context_reason="17 GiB de pesos em L4: 2 GiB runtime + 2 GiB KV em 32K.",
                    notes=("Faixa 4 bit: 16-19 GB.",)),
        ),
        QWEN38_SOURCE,
        notes=("Denso 27B, visão, raciocínio, 256K; sampling 1.0/0.95/20/0.0.",),
        sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        reasoning_effort="xhigh",
        reasoning_source=REASONING_SOURCE_QWEN38,
    ),
    ModelEntry(
        "gemma-4", "Gemma 4", "llama-cpp",
        (
            Variant("e2b-q4", "E2B 4 bit", 4.0, 131_072, "T4", context_doc=131_072),
            Variant("e4b-q4", "E4B 4 bit", 6.0, 65_536, "T4",
                    context_doc=131_072,
                    context_reason="6 GiB de pesos em T4: 65.536 tokens mantêm 4 GiB de KV."),
            Variant("12b-q4", "12B UD-Q4_K_XL", 8.0, 32_768, "T4",
                    context_doc=262_144,
                    context_reason="8 GiB de pesos em T4: 32K mantêm 2 GiB de KV."),
            Variant("26b-a4b-q4", "26B-A4B UD-Q4_K_XL", 18.0, 32_768, "L4",
                    context_doc=262_144,
                    context_reason="18 GiB de pesos + 2 GiB runtime em L4 limitam o KV."),
            Variant("31b-q4", "31B UD-Q4_K_XL", 20.0, 131_072, "A100-40",
                    context_doc=262_144,
                    context_reason="20 GiB de pesos na A100-40: 131.072 tokens usam 8 GiB de KV."),
        ),
        GEMMA4_SOURCE,
        notes=(
            "Sampling: temperature 1.0, top_p 0.95, top_k 64.",
            "Thinking: token <|think|> no início do system prompt; em mult turno, "
            "manter só a resposta final visível no histórico.",
        ),
        sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0},
        # A página NÃO documenta reasoning_effort para Gemma 4. Não inventar valor.
        reasoning_effort=None,
        reasoning_source=REASONING_SOURCE_GEMMA,
        thinking=ThinkingPolicy(),
    ),
    ModelEntry(
        "muse-glimmer-30b", "Muse Glimmer 30B", "llama-cpp",
        (
            Variant("q2", "UD-Q2_K_XL (2 bit)", 12.0, 32_768, "T4",
                    context_doc=131_072,
                    needs_custom_build=True,
                    context_reason="12 GiB de pesos em T4: 32K mantêm 2 GiB de KV."),
            Variant("q4", "UD-Q4_K_XL (4 bit)", 17.0, 65_536, "L4",
                    context_doc=131_072,
                    needs_custom_build=True,
                    context_reason="17 GiB de pesos + 2 GiB runtime + 4 GiB KV = 23 GiB em L4."),
            Variant("q6", "UD-Q6_K_XL (6 bit)", 22.0, 131_072, "A100-40",
                    context_doc=131_072, needs_custom_build=True),
        ),
        MUSE_SOURCE,
        notes=(
            "Denso 30B com visão; 131.072 de contexto padrão, até 262.144.",
            "Exige llama.cpp com suporte a Muse Glimmer fixado em b12649; "
            "confirmar suporte na primeira execução real.",
            "Sampling: temperature 1.0, top_p 0.95, top_k 64; raciocínio "
            "low/medium/high/xhigh.",
        ),
        sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0},
        reasoning_effort="high",
        reasoning_source=REASONING_SOURCE_MUSE,
    ),
    ModelEntry(
        "qwen-image-2.1", "Qwen-Image-2.1", "image",
        (
            Variant("fp8-int8", "FP8 repo / INT8-ConvRot", 20.0, 0, "L4",
                    notes=("GPU com 24 GB ou mais; 7,26 GB de pesos INT8-ConvRot.",)),
            Variant("q4km", "GGUF Q4_K_M", 12.0, 0, "T4",
                    notes=("GPU de 12-16 GB com offload parcial para RAM; ~14,9 GB no total "
                           "com encoder e VAE.",)),
        ),
        IMAGE_SOURCE,
        notes=(
            "Modelo de difusão (7B de imagem + Qwen3-VL 8B), servido por sd-server "
            "em POST /v1/images/generations — NÃO é /v1/chat/completions.",
            "T4 Colab de 15 GB: 1024x1024 em ~120 s (25 passos, fp16); treino exige bf16.",
        ),
        generation_defaults={"steps": 25, "guidance": 1.0},
        generation_source=GENERATION_SOURCE_IMAGE,
    ),
    ModelEntry(
        "minimax-m3.1-flash", "MiniMax-M3.1-Flash", "llama-cpp",
        (
            Variant("q4", "UD-Q4_K_XL (4 bit)", 256.0, 16_384, "H100-80",
                    available=False,
                    reason="Não cabe em GPU única do Colab: 249,4 GB de download; "
                           "orçamento de 256 GB+ (RAM+VRAM) para 4 bit."),
        ),
        "https://unsloth.ai/docs/models/minimax-m3.1-flash.md",
        notes=("1 bit 129,4 GB; 2 bit 143,9 GB; orçamento 160 GB+ em 2 bit.",),
    ),
    ModelEntry(
        "glm-5.3-flash", "GLM-5.3-Flash", "llama-cpp",
        (
            Variant("q3", "UD-Q3_K_XL (3 bit)", 147.54, 32_768, "H100-80",
                    available=False,
                    reason="Não cabe em GPU única do Colab: 147,54 GB de download em "
                           "3 bit; docs recomendam dispositivo de 128 GB."),
        ),
        "https://unsloth.ai/docs/models/glm-5.3-flash.md",
        notes=("1 bit 93,09 GB; 3 bit UD-IQ3_XXS 120,37 GB; 4 bit 199,71 GB.",),
    ),
    ModelEntry(
        "deepseek-v4.1-flash", "DeepSeek-V4.1-Flash", "llama-cpp",
        (
            Variant("q4", "UD-Q4_K_XL (4 bit)", 500.7, 1_048_576, "H100-80",
                    available=False,
                    reason="Não cabe em GPU única do Colab: 500,7 GB de download em "
                           "4 bit; tabela das docs indica 305 GB + Engram em SSD."),
        ),
        "https://unsloth.ai/docs/models/deepseek-v4.1-flash.md",
        notes=("1 bit 128-145 GB; Q8_K_XL 508,4 GB; requer 128 GB+ de RAM.",),
    ),
)

#: Contexto REAL de cada perfil. `Variant.context` é o valor de referência da
#: variante; aqui a redução fica explícita por perfil/GPU, com o motivo.
PROFILE_CONTEXTS: tuple[ProfileContext, ...] = (
    ProfileContext("qwen3.8-27b-32k", "qwen3.8-27b-awq", "q4-awq", 262_144, 32_768,
                   "A100-40", "A100 40 GB sem high_mem: 32K limita o KV a 2 GiB."),
    ProfileContext("qwen3.8-27b-256k", "qwen3.8-27b-awq", "q4-awq", 262_144, 262_144,
                   "A100-80", "Contexto máximo documentado mantido em A100 80 GB (high_mem)."),
    ProfileContext("inkling-2-mini-iq2-gguf", "inkling-2-mini", "q2", 1_000_000, 32_768,
                   "T4", "1M pediria ~64 GiB de KV; perfil T4 usa 32K por margem de memória."),
    ProfileContext("inkling-2-mini-q4-gguf", "inkling-2-mini", "q4", 1_000_000, 65_536,
                   "L4", "18 GiB pesos + 2 GiB runtime + 4 GiB KV em 65.536 = 24 GiB em L4."),
    ProfileContext("inkling-2-mini-q8-gguf", "inkling-2-mini", "q8", 1_000_000, 131_072,
                   "A100-40", "24 GiB de pesos limitam o KV a 8 GiB na A100-40."),
    ProfileContext("qwen3.8-27b-q3-gguf", "qwen3.8-27b-gguf", "q3", 262_144, 32_768,
                   "T4", "12 GiB de pesos em T4: 32K mantém 2 GiB de KV."),
    ProfileContext("qwen3.8-27b-q4-gguf", "qwen3.8-27b-gguf", "q4", 262_144, 32_768,
                   "L4", "17 GiB de pesos em L4: 32K mantém 2 GiB de KV."),
    ProfileContext("gemma-4-e4b-q4-gguf", "gemma-4", "e4b-q4", 131_072, 65_536,
                   "T4", "6 GiB de pesos em T4: 65.536 tokens mantêm 4 GiB de KV."),
    ProfileContext("gemma-4-12b-q4-gguf", "gemma-4", "12b-q4", 262_144, 32_768,
                   "T4", "8 GiB de pesos em T4: 32K mantêm 2 GiB de KV."),
    ProfileContext("gemma-4-26b-a4b-q4-gguf", "gemma-4", "26b-a4b-q4", 262_144, 32_768,
                   "L4", "18 GiB de pesos + 2 GiB runtime em L4 limitam o KV."),
    ProfileContext("gemma-4-31b-q4-gguf", "gemma-4", "31b-q4", 262_144, 131_072,
                   "A100-40", "20 GiB de pesos na A100-40: 131.072 tokens usam 8 GiB de KV."),
    ProfileContext("muse-glimmer-30b-q2-gguf", "muse-glimmer-30b", "q2", 131_072, 32_768,
                   "T4", "12 GiB de pesos em T4: 32K mantêm 2 GiB de KV."),
    ProfileContext("muse-glimmer-30b-q4-gguf", "muse-glimmer-30b", "q4", 131_072, 65_536,
                   "L4", "17 GiB de pesos + 2 GiB runtime + 4 GiB KV em L4."),
    ProfileContext("qwen-image-2.1-q4km", "qwen-image-2.1", "q4km", 0, 1,
                   "T4", "Modelo de difusão não tem janela autoregressiva; 1 é sentinela."),
    ProfileContext("qwen-image-2.1-fp8", "qwen-image-2.1", "fp8-int8", 0, 1,
                   "L4", "Modelo de difusão não tem janela autoregressiva; 1 é sentinela."),
)


#: (profile_id, model_id, variant_id, max_model_len) — montado pela CLI a partir
#: dos perfis TOML; o catálogo não importa `config` para evitar ciclo.
ProfileRef = tuple[str, str, str, int]


def format_gpus(
    profile_refs: tuple[ProfileRef, ...] = (), *, numbered: bool = False
) -> str:
    lines = ["GPU | VRAM (GiB) | bf16 | VRAM aproximada | Perfis que cabem", ""]
    for index, gpu in enumerate(GPUS, 1):
        fits = []
        for profile_id, model_id, variant_id, max_len in profile_refs:
            result = check_fit(model_id, variant_id, gpu.id, max_len)
            if result.fits:
                fits.append(profile_id)
        aprox = "sim" if gpu.vram_approximate else "nao"
        prefix = f"{index}. " if numbered else ""
        lines.append(
            f"{prefix}{gpu.id} ({gpu.label}) | {gpu.vram_gib} | "
            f"{'sim' if gpu.bf16 else 'nao'} | {aprox} | {', '.join(fits) or '-'}"
        )
    lines += [
        "",
        f"Fonte VRAM: {GPU_SOURCE}",
        f"Nota Colab: {COLAB_GPU_SOURCE}",
    ]
    return "\n".join(lines)


def profile_context(profile_id: str) -> ProfileContext:
    for item in PROFILE_CONTEXTS:
        if item.profile_id == profile_id:
            return item
    raise ValueError(f"Perfil sem contexto registrado no catálogo: {profile_id!r}")


def served_contexts_for(model_id: str, variant_id: str) -> tuple[int, ...]:
    """Contextos efetivamente servidos pelos perfis de uma variante."""
    values = tuple(
        sorted(
            {
                item.context_served
                for item in PROFILE_CONTEXTS
                if item.model_id == model_id and item.variant_id == variant_id
            }
        )
    )
    return values or (variant_by_id(model_id, variant_id).context,)


def format_models() -> str:
    """Mostre contexto documentado E contexto servido — números distintos."""
    header = (
        "Modelo | Runtime | Variante | Memoria (GiB) | Contexto doc | "
        "Contexto servido | GPU minima | Status"
    )
    lines = [header, ""]
    for model in MODELS:
        for variant in model.variants:
            status = "disponivel" if variant.available else "NAO CABE EM GPU UNICA DO COLAB"
            doc = str(variant.context_doc) if variant.context_doc else "n/a"
            served = ", ".join(str(value) for value in served_contexts_for(model.id, variant.id))
            if variant.min_gpu == "T4" and model.runtime == "image":
                served = "n/a (imagem)"
            lines.append(
                f"{model.id} | {model.runtime} | {variant.id} ({variant.label}) | "
                f"{variant.memory_gib:g} | {doc} | {served} | {variant.min_gpu} | {status}"
            )
            if variant.context_doc and variant.context_doc > variant.context:
                lines.append(f"  -> reducao documentada: {variant.context_reason}")
            if not variant.available:
                lines.append(f"  -> {variant.reason}")
    lines += [
        "",
        "Contexto servido = valor do perfil (usado na checagem de compatibilidade); "
        "contexto doc = maximo da pagina oficial. Motivos de reducao: "
        "PROFILE_CONTEXTS e Variant.context_reason (documentacao consultada em "
        + DOCS_CHECKED_ON
        + ").",
        "Fontes: " + ", ".join(dict.fromkeys(m.source for m in MODELS)),
        "Fontes de reasoning e geracao: " + ", ".join(
            dict.fromkeys(
                m.reasoning_source or m.generation_source
                for m in MODELS
                if m.reasoning_source or m.generation_source
            )
        ),
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class CompatibilityResult:
    fits: bool
    required_gib: float
    available_gib: float
    gpu_id: str
    alternatives: tuple[str, ...]
    reason: str = ""


def gpu_by_id(gpu_id: str) -> Gpu:
    for gpu in GPUS:
        if gpu.id == gpu_id:
            return gpu
    raise ValueError(f"GPU desconhecida: {gpu_id!r}")


def resolve_gpu(colab_gpu: str, high_mem: bool = False) -> Gpu:
    """Traduza o valor de `config.toml` para um ID do catálogo."""
    label = colab_gpu.strip().upper()
    if label == "A100":
        return gpu_by_id("A100-80" if high_mem else "A100-40")
    for gpu in GPUS:
        if label in {name.upper() for name in gpu.colab_names}:
            return gpu
    raise ValueError(f"GPU do Colab desconhecida: {colab_gpu!r}")


def model_by_id(model_id: str) -> ModelEntry:
    for model in MODELS:
        if model.id == model_id:
            return model
    raise ValueError(f"Modelo desconhecido no catálogo: {model_id!r}")


def variant_by_id(model_id: str, variant_id: str) -> Variant:
    for variant in model_by_id(model_id).variants:
        if variant.id == variant_id:
            return variant
    raise ValueError(f"Variante {variant_id!r} desconhecida para {model_id!r}")


def kv_cache_gib(max_model_len: int) -> float:
    if max_model_len <= 0:
        return MIN_KV_CACHE_GIB
    return max(MIN_KV_CACHE_GIB, KV_CACHE_GIB_PER_32K * max_model_len / 32768)


def required_vram_gib(variant: Variant, max_model_len: int, runtime: str = "") -> float:
    kv = 0.0 if runtime == "image" else kv_cache_gib(max_model_len)
    return round(variant.memory_gib + RUNTIME_RESERVE_GIB + kv, 2)


def check_fit(
    model_id: str, variant_id: str, gpu_id: str, max_model_len: int = 32768
) -> CompatibilityResult:
    """Verifique se a variante cabe na GPU declarada, sem realocar nada."""
    variant = variant_by_id(model_id, variant_id)
    gpu = gpu_by_id(gpu_id)
    runtime = model_by_id(model_id).runtime
    required = required_vram_gib(variant, max_model_len, runtime)
    alternatives: list[str] = []
    if variant.available:
        for other in model_by_id(model_id).variants:
            if other.id == variant.id or not other.available:
                continue
            if required_vram_gib(other, max_model_len, runtime) <= gpu.vram_gib:
                alternatives.append(f"{model_id}:{other.id}")
    fits = variant.available and required <= gpu.vram_gib
    reason = ""
    if not variant.available:
        reason = variant.reason
    elif not fits:
        kv = 0.0 if runtime == "image" else kv_cache_gib(max_model_len)
        reason = (
            f"{variant.label} precisa de ~{required:.1f} GiB "
            f"(pesos {variant.memory_gib:g} + runtime {RUNTIME_RESERVE_GIB:g} + "
            f"KV ~{kv:.1f}), mas {gpu.label} tem {gpu.vram_gib} GiB."
        )
    return CompatibilityResult(
        fits=fits,
        required_gib=required,
        available_gib=float(gpu.vram_gib),
        gpu_id=gpu.id,
        alternatives=tuple(alternatives),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Seleção de GPU: funções puras (a camada Typer só pergunta e imprime).
# ---------------------------------------------------------------------------


def _invalid_gpu_message(choice: str) -> str:
    return (
        f"Invalid GPU {choice!r}. Accepted IDs: "
        + ", ".join(gpu.id for gpu in GPUS)
        + ". You can also enter the number shown in the GPU list."
    )


def parse_gpu_choice(choice: str) -> Gpu:
    """Aceite o número exibido ou um ID exato do catálogo, sem efeito colateral."""
    raw = choice.strip()
    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(GPUS):
            return GPUS[index - 1]
        raise ValueError(_invalid_gpu_message(choice))
    try:
        return gpu_by_id(raw.upper())
    except ValueError:
        pass
    # `resolve_gpu` traduz nomes do Colab, onde "A100" é ambíguo (40 ou 80).
    # Reutilizamos essa tradução para reconhecer aliases, mas nunca a usamos
    # para escolher uma GPU automaticamente: só IDs exatos são aceitos.
    with contextlib.suppress(ValueError):
        resolve_gpu(raw)
    raise ValueError(_invalid_gpu_message(choice))


@dataclass(frozen=True)
class CompatibleVariant:
    model_id: str
    runtime: str
    variant_id: str
    variant_label: str
    required_gib: float
    available_gib: float
    context_doc: int
    context_served: int
    profile_id: str  # vazio quando a variante não tem perfil em PROFILE_CONTEXTS


@dataclass(frozen=True)
class CompatibleSelection:
    gpu_id: str
    rows: tuple[CompatibleVariant, ...]
    dropped_for_memory: int

    @property
    def compatible_count(self) -> int:
        return len(self.rows)


def compatible_variants(gpu_id: str) -> CompatibleSelection:
    """Filtre variantes disponíveis com `check_fit` em cada contexto servido.

    Uma variante passa se pelo menos um contexto servido couber na GPU. O
    perfil exibido é o daquele contexto, preferindo o registrado para a própria
    GPU quando houver mais de um. Variantes indisponíveis são ignoradas.
    """
    gpu = gpu_by_id(gpu_id)
    rows: list[CompatibleVariant] = []
    dropped = 0
    for model in MODELS:
        for variant in model.variants:
            if not variant.available:
                continue
            candidates: list[tuple[int, int, str, int, CompatibilityResult]] = []
            for context in served_contexts_for(model.id, variant.id):
                fit = check_fit(model.id, variant.id, gpu.id, context)
                if not fit.fits:
                    continue
                refs = [
                    item
                    for item in PROFILE_CONTEXTS
                    if item.model_id == model.id
                    and item.variant_id == variant.id
                    and item.context_served == context
                ]
                if not refs:
                    # Sem perfil registrado, served_contexts_for já devolveu
                    # Variant.context como contexto de referência.
                    candidates.append((1, -context, "", context, fit))
                for ref in refs:
                    candidates.append(
                        (
                            0 if ref.gpu_id == gpu.id else 1,
                            -context,
                            ref.profile_id,
                            context,
                            fit,
                        )
                    )
            if not candidates:
                dropped += 1
                continue
            _exact_gpu, _negative_context, profile_id, context, fit = min(
                candidates, key=lambda item: item[:4]
            )
            rows.append(
                CompatibleVariant(
                    model_id=model.id,
                    runtime=model.runtime,
                    variant_id=variant.id,
                    variant_label=variant.label,
                    required_gib=fit.required_gib,
                    available_gib=fit.available_gib,
                    context_doc=variant.context_doc,
                    context_served=context,
                    profile_id=profile_id,
                )
            )
    return CompatibleSelection(gpu.id, tuple(rows), dropped)


def format_compatible_models(gpu_id: str) -> str:
    """Tabela de compatíveis no mesmo estilo de colunas de `format_models`."""
    selection = compatible_variants(gpu_id)
    gpu = gpu_by_id(gpu_id)
    header = (
        "Modelo | Runtime | Variante | Memoria (GiB) | VRAM (GiB) | "
        "Contexto doc | Contexto servido | Perfil"
    )
    lines = [
        f"GPU: {gpu.id} ({gpu.label}) — {gpu.vram_gib} GiB VRAM aproximada.",
        "",
        header,
        "",
    ]
    for row in selection.rows:
        doc = str(row.context_doc) if row.context_doc else "n/a"
        served = "n/a (imagem)" if row.runtime == "image" else str(row.context_served)
        profile = row.profile_id or "n/a (sem perfil)"
        lines.append(
            f"{row.model_id} | {row.runtime} | {row.variant_id} ({row.variant_label}) | "
            f"{row.required_gib:g} | {row.available_gib:g} | {doc} | {served} | {profile}"
        )
    lines += [
        "",
        f"Summary: {selection.compatible_count} compatible variant(s); "
        f"{selection.dropped_for_memory} discarded for insufficient VRAM.",
        "Unavailable catalog variants are excluded from the table and from both counts.",
        f"Fonte VRAM: {GPU_SOURCE}",
    ]
    return "\n".join(lines)


def validate_catalog() -> None:
    """Falhe rápido se o catálogo tiver IDs duplicados ou números impossíveis."""
    seen_gpus: set[str] = set()
    for gpu in GPUS:
        if gpu.id in seen_gpus or gpu.vram_gib <= 0 or not gpu.colab_names:
            raise ValueError(f"GPU inválida no catálogo: {gpu.id}")
        seen_gpus.add(gpu.id)
    seen_models: set[str] = set()
    for model in MODELS:
        if model.id in seen_models or model.runtime not in RUNTIMES or not model.source:
            raise ValueError(f"Modelo inválido no catálogo: {model.id}")
        seen_models.add(model.id)
        if model.reasoning_effort is not None and model.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort não documentado: {model.id}")
        if model.reasoning_effort is not None and not model.reasoning_source:
            raise ValueError(f"reasoning_effort sem fonte: {model.id}")
        if model.thinking is not None and not model.thinking.source:
            raise ValueError(f"Política de thinking sem fonte: {model.id}")
        for key, value in model.generation_defaults.items():
            if key not in {"steps", "guidance"} or isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError(f"Padrão de geração inválido: {model.id}:{key}")
        if model.generation_defaults and not model.generation_source:
            raise ValueError(f"Padrões de geração sem fonte: {model.id}")
        expected_defaults = {"steps", "guidance"}
        if model.generation_defaults and set(model.generation_defaults) != expected_defaults:
            raise ValueError(f"Padrões de geração incompletos: {model.id}")
        seen_variants: set[str] = set()
        for variant in model.variants:
            if variant.id in seen_variants or variant.memory_gib <= 0 or variant.context < 0:
                raise ValueError(f"Variante inválida: {model.id}:{variant.id}")
            if variant.context_doc < 0 or (
                variant.context_doc and variant.context > variant.context_doc
            ):
                raise ValueError(f"Contexto servido acima do documentado: {model.id}:{variant.id}")
            if variant.min_gpu not in seen_gpus:
                raise ValueError(f"GPU mínima desconhecida: {variant.min_gpu}")
            if not variant.available and not variant.reason:
                raise ValueError(f"Variante indisponível sem motivo: {model.id}:{variant.id}")
            if variant.context_doc > variant.context and not variant.context_reason:
                raise ValueError(f"Redução de contexto sem motivo: {model.id}:{variant.id}")
            seen_variants.add(variant.id)
    seen_profiles: set[str] = set()
    for item in PROFILE_CONTEXTS:
        if item.profile_id in seen_profiles:
            raise ValueError(f"Perfil duplicado em PROFILE_CONTEXTS: {item.profile_id}")
        seen_profiles.add(item.profile_id)
        variant = variant_by_id(item.model_id, item.variant_id)
        if item.context_doc != variant.context_doc or item.context_served > item.context_doc > 0:
            raise ValueError(f"Contexto inconsistente no perfil: {item.profile_id}")
        if variant.context and item.context_served != variant.context and len(
            served_contexts_for(item.model_id, item.variant_id)
        ) == 1:
            raise ValueError(f"Contexto servido do perfil divergente: {item.profile_id}")
        if gpu_by_id(item.gpu_id).vram_gib < gpu_by_id(variant.min_gpu).vram_gib:
            raise ValueError(f"GPU do perfil abaixo da mínima da variante: {item.profile_id}")
        if item.context_doc > item.context_served and not item.reason:
            raise ValueError(f"Redução de contexto do perfil sem motivo: {item.profile_id}")


validate_catalog()
