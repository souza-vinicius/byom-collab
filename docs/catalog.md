---
type: Architecture
title: VRAM Budget and GPU Eligibility
description: How byom decides whether a model variant fits a GPU — weights, vision projector, runtime allowance, dtype-aware KV cache (heuristic or per-architecture), DSpark drafter and the mandatory headroom — and why equality never approves.
tags: [vram, catalog, quantizacao, kv-cache, gpu, dspark]
status: stable
generated: { by: hermes/deepseek-v4.1-flash, at: 2026-11-24T23:40:00Z }
sources:
  - id: catalog-module
    resource: /src/byom/catalog.py
    title: Single GPU/model catalog, VRAM math and fit decisions
  - id: profiles
    resource: /src/byom/profiles
    title: TOML profiles that pin GPU, context and runtime per model
  - id: worker
    resource: /worker/build_vllm_command.py
    title: Runtime argv, including the DSpark draft flags
  - id: inkling-config
    resource: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json
    title: Inkling 2 Mini official config.json (num_hidden_layers, local_layer_ids, sliding_window_size)
  - id: log-2026-11-24
    resource: /log.md
    title: Update log with the revalidated per-profile table
---

# VRAM Budget and GPU Eligibility

`check_fit(model, variant, gpu, context)` is the single place that answers "cabe?".
The number it prints is a sum of five things, each from a different source:[^catalog-module]

1. **pesos** — `Variant.memory_gib`, sempre em GiB, sempre derivado dos bytes
   publicados no repositório do modelo (não da estimativa da página). Um quant
   que não existe no repositório não pode aparecer: `prepare_runtime.match_file`
   levanta `ValueError: No model file matches` antes de qualquer download.
2. **visão** — `Variant.vision_gib`, o tamanho do `mmproj` que o padrão do perfil
   realmente resolve (Inkling 0,65 GiB; Gemma 26B/31B ~1,11 GiB; Muse 3,58 GiB;
   perfis de imagem 1,08 GiB). Projetor ausente = 0.
3. **runtime** — `RUNTIME_RESERVE_GIB = 2`, reserva fixa para pesos do
   interpretador, contextos CUDA e buffers do servidor.
4. **KV** — `kv_cache_gib(max_model_len, model, ...)`: proporcional ao contexto
   servido e **ao dtype que o argv realmente pede** (`KV_CACHE_DTYPE_FACTORS =
   {"fp8": 0.5, "f16": 1.0}`). Perfis vLLM herdam `fp8`
   (`--kv-cache-dtype fp8`); llama-cpp fica em f16; perfis de imagem não têm KV.
5. **drafter** — `catalog.drafter_gib(worker)`: 0,47 GiB quando o `[worker]`
   declara `dspark = true`, porque o llama-server carrega o drafter ao lado do
   modelo (`--model-draft`). Sem `dspark`, zero.

A aprovação exige folga de verdade:[^log-2026-11-24]

```
vram_headroom_gib(vram) = max(1.0, 5% de vram)
fits  <=>  required + headroom  <  vram
```

`<` e não `<=`: ficar exatamente no limite (14,0 de required + 1,0 de folga =
15,0 de 15,0 GiB na T4) é justamente o caso que a folga existe para reprovar.
A mensagem de recusa cita cada parcela (`pesos`, `runtime`, `KV`, `visão`,
`drafter`) para que o ajuste seja aritmético, não por tentativa.

## KV: heurístico ou arquitetura

O padrão continua sendo o heurístico `KV_CACHE_GIB_PER_32K = 2` GiB por 32K em
f16, com piso de 1 GiB. Ele é conservador de propósito e **superestima** modelos
com atenção híbrida, em que a maioria das camadas para de crescer numa janela.

Quem declara `ModelEntry.kv_shape` (`KvProfile`, com `source` + `checked_at` de
um `config.json` oficial) sai do heurístico e passa a usar:

```
2 (K,V) × kv_heads × head_dim × bytes × (layers_full × ctx + layers_local × min(ctx, window))
```

arredondado **para cima** no centésimo, com `bytes` do dtype nativo do config
multiplicado pelo fator do `kv_cache_dtype` do runtime (fp8 = metade). O Inkling
2 Mini é o caso medido: `text_config` com 24 camadas, `local_layer_ids` de 20
entradas (4 completas + 20 com janela de 1024), 8 heads de KV e `head_dim` 128 —
16 KiB por token nas completas e 80 MiB fixos nas locais. Em f16: 16K = 0,33 GiB,
32K = 0,58, 64K = 1,08, 128K = 2,08 — contra 1, 2, 4 e 8 do heurístico.[^inkling-config]

A interface não esconde a diferença: a tela de detalhe do perfil mostra
`Cache KV` com o valor, o dtype e a procedência (`catalog.kv_shape_note`) — a
URL e a data quando o número é medido, ou a palavra "heurístico" quando não é.

## DSpark

`[worker] dspark = true` faz `prepare_runtime` baixar `drafter_pattern` e
`build_llama_args` emitir `--model-draft <manifest["drafter"]> --spec-type
draft-dspark --spec-draft-n-max 3 --n-gpu-layers-draft 99`.[^worker] O perfil
que liga o campo entra no orçamento com os 0,47 GiB do drafter em todo lugar
que chama `check_fit`: `check_compatibility`, `byom models --gpu`, a TUI e
`scripts/vram_table.py`. `available_profiles` recusa `dspark = true` sem
`drafter_pattern` antes de qualquer download — e nenhum TOML embarcado liga o
campo enquanto o binário com suporte (PR #32218) não tiver URL e sha256 pinadas.

## Contexto acima do documentado

`max_model_len` não pode passar do `context_doc` da variante: um perfil que pede
400.000 num modelo de 262.144 é recusado na carga com `ConfigError`. Quando o
contexto servido fica abaixo do documentado, a variante declara o motivo em
`context_reason` e a TUI mostra esse texto em vez de recalcular a GPU mínima por
conta própria.

## Onde os números aparecem

A mesma conta alimenta `byom models`, o submenu `Selecionar GPU` da TUI, o
`check_compatibility` do `config.toml` e o log de atualização. A tabela dos 18
perfis (GPU, contexto, KV, required, folga, margem, `fits`, origem do KV e
drafter) é regerada com `scripts/vram_table.py` e registrada em
[Update Log](/log.md); sondas de comportamento ficam em
`scripts/probe_revisao.py`.

[^catalog-module]: `/src/byom/catalog.py`
[^worker]: `/worker/build_vllm_command.py` (`build_llama_args`)
[^inkling-config]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json (consultado em 2026-11-24)
[^log-2026-11-24]: `/log.md` — entrada "KV por arquitetura (2026-11-24)"
