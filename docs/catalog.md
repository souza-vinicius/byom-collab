---
type: Architecture
title: VRAM Budget and GPU Eligibility
description: How byom decides whether a model variant fits a GPU — weights, vision projector, runtime allowance, dtype-aware KV cache and the mandatory headroom — and why equality never approves.
tags: [vram, catalog, quantizacao, kv-cache, gpu]
status: stable
generated: { by: hermes/deepseek-v4.1-flash, at: 2026-11-24T23:40:00Z }
sources:
  - id: catalog-module
    resource: /src/byom/catalog.py
    title: Single GPU/model catalog, VRAM math and fit decisions
  - id: profiles
    resource: /src/byom/profiles
    title: TOML profiles that pin GPU, context and runtime per model
  - id: prepare-runtime
    resource: /worker/prepare_runtime.py
    title: Downloads artifacts and resolves model/image role patterns
  - id: log-2026-11-24
    resource: /log.md
    title: Update log with the revalidated per-profile table
---

# VRAM Budget and GPU Eligibility

`check_fit(model, variant, gpu, context)` is the single place that answers "cabe?".
The number it prints is a sum of four things, each from a different source:[^catalog-module]

1. **pesos** — `Variant.memory_gib`, sempre em GiB, sempre derivado dos bytes
   publicados no repositório do modelo (não da estimativa da página). Um quant
   que não existe no repositório não pode aparecer: `prepare_runtime.match_file`
   levanta `ValueError: No model file matches` antes de qualquer download.
2. **visão** — `Variant.vision_gib`, o tamanho do `mmproj` que o padrão do perfil
   realmente resolve (Inkling 0,65 GiB; Gemma 26B/31B ~1,11 GiB; Muse 3,58 GiB;
   perfis de imagem 1,08 GiB). Projetor ausente = 0.
3. **runtime** — `RUNTIME_ALLOWANCE_GIB = 2`, reserva fixa para pesos do
   interpretador, contextos CUDA e buffers do servidor.
4. **KV** — `kv_cache_gib(...)`, proporcional ao contexto servido e **ao dtype
   que o argv realmente pede**: `KV_CACHE_DTYPE_FACTORS = {"fp8": 0.5,
   "f16": 1.0}` sobre `KV_GIB_PER_32K = 2`. Perfis vLLM herdam `fp8`
   (`--kv-cache-dtype fp8`); llama-cpp fica em f16; perfis de imagem não têm KV.

A aprovação exige folga de verdade:[^log-2026-11-24]

```
vram_headroom_gib(vram) = max(1.0, 5% de vram)
fits  <=>  required + headroom  <  vram
```

`<` e não `<=`: ficar exatamente no limite (14,0 de required + 1,0 de folga =
15,0 de 15,0 GiB na T4) é justamente o caso que a folga existe para reprovar.
A mensagem de recusa cita cada parcela (`pesos`, `runtime`, `KV`, `visão`) para
que o ajuste seja aritmético, não por tentativa.

## Contexto acima do documentado

`max_model_len` não pode passar do `context_doc` da variante: um perfil que pede
400.000 num modelo de 262.144 é recusado na carga com `ConfigError`. Quando o
contexto servido fica abaixo do documentado, a variante declara o motivo em
`context_reason` e a TUI mostra esse texto em vez de recalcular a GPU mínima por
conta própria.

## Onde os números aparecem

A mesma conta alimenta `byom models`, o submenu `Selecionar GPU` da TUI, o
`check_compatibility` do `config.toml` e o log de atualização. A tabela dos 15
perfis (GPU, contexto, required, folga, margem e `fits`) é regerada com
`scripts/vram_table.py` e registrada em [Update Log](/log.md); sondas de
comportamento ficam em `scripts/probe_revisao.py`.

[^catalog-module]: `/src/byom/catalog.py`
[^log-2026-11-24]: `/log.md` — entrada "Lotes 4 e 5 — orçamento de VRAM revalidado (2026-11-24)"
