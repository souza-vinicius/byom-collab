---
type: Architecture
title: VRAM Budget and GPU Eligibility
description: How byom decides whether a model variant fits a GPU — weights, vision projector, runtime allowance, dtype-aware KV cache (heuristic or per-architecture), DSpark drafter and the mandatory headroom — and why equality never approves. Includes the `strata` runtime, whose packs are one file each rather than a GGUF plus drafter.
tags: [vram, catalog, quantizacao, kv-cache, gpu, dspark, strata]
status: stable
generated: { by: hermes/deepseek-v4.1-flash, at: 2026-11-27T22:50:00Z }
sources:
  - id: catalog-module
    resource: /src/byom/catalog.py
    title: Single GPU/model catalog, VRAM math and fit decisions
  - id: profiles
    resource: /src/byom/profiles
    title: TOML profiles that pin GPU, context and runtime per model
  - id: worker
    resource: /worker/build_vllm_command.py
    title: Runtime argv, including the DSpark draft flags and the Strata run config
  - id: prepare
    resource: /worker/prepare_runtime.py
    title: Pinned downloads, pack checksum verification and the NVIDIA driver check
  - id: inkling-config
    resource: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json
    title: Inkling 2 Mini official config.json (num_hidden_layers, local_layer_ids, sliding_window_size)
  - id: strata-card
    resource: https://huggingface.co/strata-engine/Inkling-2-Mini-Strata
    title: Inkling 2 Mini packs for the Strata engine (Apache-2.0, base_model thinkingmachines/Inkling-2-Mini)
  - id: strata-release
    resource: https://github.com/Niko1221/Strata/releases/tag/v0.2.23
    title: Strata v0.2.23 release (control 0.1.70 + strata-inkling 0.1.70, min_driver 575.51)
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
   {"fp8": 0.5, "int8": 0.5, "fp16": 1.0, "f16": 1.0}`). Perfis vLLM herdam
   `fp8` (`--kv-cache-dtype fp8`); llama-cpp fica em f16; perfis de imagem não
   têm KV; `strata` herda `fp8` (`--kv fp8` do engine, 1 byte por elemento).
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
`drafter_pattern` antes de qualquer download.

O perfil embarcado é `inkling-2-mini-ad-q6k-dspark` (AD-Q6_K, 128K, A100-40,
27,35 GiB + 2,0 de folga). Ele fixa o build oficial **b12800**
(`v1.1.0-dev`, `--spec-type draft-dspark`), porque o PR #32218 foi mergeado em
2026-11-24 e entrou na **v1.1.0**: o fork do PR e o workflow de build que o
byom mantinha deixaram de existir. O drafter vem do mesmo `model_repo` dos
pesos (a resolução é por padrão dentro de `model_repo`), com 503.857.696 B.

## Runtime `strata` (packs `.strata`)

`strata-engine/Inkling-2-Mini-Strata` não publica GGUF nem safetensors: publica
um pack `.strata` por tamanho de card, no container do engine **Strata**
(`github.com/Niko1221/Strata`, MIT). O byom serve isso com o runtime próprio
`strata`, no mesmo molde do sd-server: binário fixado + pesos baixados + argv
montado.[^strata-card]

| Pack | Bytes (tree do HF, revisão `5623ceb`, 2026-11-27) | GiB | Card recomendado |
| --- | ---: | ---: | --- |
| `Inkling-2-Mini-SQ-10.strata` | 10.471.526.400 | 9,75 | 12 GiB |
| `Inkling-2-Mini-SQ-15.strata` | 14.348.316.672 | 13,36 | 16 GiB |
| `Inkling-2-Mini-SQv2-20.strata` | 19.374.768.128 | 18,04 | 24 GiB |
| `Inkling-2-Mini-Vision.strata` (não usado) | 636.526.592 | 0,59 | leitor de imagem |

Componentes pinados, da release **v0.2.23** (publicada em 2026-11-27T18:43:32Z,
`sha256` conferido com `sha256sum` nos dois tarballs):[^strata-release]

```
control  strata-control-0.1.70-linux-x86_64.tar.gz              94.828.716 B
         sha256 f38d85bd5bd19393808e2bd0c7867e34c4fed968065371d77f37c11c451eeb6a
engine   strata-inkling-0.1.70-linux-x86_64-cuda-cuda12.tar.gz  655.962.569 B
         sha256 cc078a9cf15e5e738a0396973748cbab352aed5f9cdf3a892802dfd407c6e526
```

O engine roda **Strata 0.2.23+**: é a versão do `control` que lê o layout
`.strata` **v1.1** (16 KiB-aligned) do pack SQv2-20 e passa `--native` ao
binário. O tarball do engine é achatado e **fica inteiro** no disco — o
executável resolve `libcublas.so.12` (105.140.976 B), `libcublasLt.so.12`
(749.210.000 B), `libcudart.so.12` e `libgomp.so.1` pelo próprio diretório, e é
por isso que o pacote tem 655.962.569 B.

`engine-component.json` declara `min_driver.nvidia = "575.51"`, `backend: cuda`,
`build_id ebda56cbaa19+cuda12.9` e `archs` sm_60..sm_120 (T4 = sm_75, A100 =
sm_80/86, L4 = sm_89). O `strata-feed.json` assinado da mesma release confirma
575.51 para o cuda12 e 580.65 para o cuda13 (461.245.704 B, não usado). Por isso
`prepare_runtime` consulta `nvidia-smi --query-gpu=driver_version` **antes de
qualquer download** e recusa o perfil com mensagem explícita quando o driver é
mais velho: um engine que se recusa a subir depois de 10 GiB baixados custaria a
sessão do Colab.[^prepare]

Como o argv é montado:[^worker]

```
<control>/python/bin/python3 -m serve.server --engine strata \
    --config <BYOM_DIR>/strata-<model>.json --host 127.0.0.1 \
    --port <vllm_port> --offline
```

O JSON é o run config que o próprio `control` lê (`exe`, `args`, `cwd`,
`model_name`, `log`, `tokenizer`, `arch`); `args` sai como
`--native <pack> --max-context <N> --kv <dtype> --parallel 1` mais o
`[worker] run_config_extra`, se o perfil declarar. `arch = "inkling2_loop"` é o
que faz o `serve` tratar o engine como Inkling (`serve/inkling_kind.py`), e o
`runtime.cmd` começa com `cd <control>` porque `-m serve.server` resolve `serve`
e `strata_control` pelo cwd.

O tokenizer é o único ponto que **não** funciona só com o pack. O
`serve/server.py` gera o `vocab.json` em três passos, e no 0.1.70 os dois
primeiros falham (medido com o SQ-10 real em 2026-11-27): ler o manifest do
`.strata` exige `zstandard`, que o CPython embarcado não tem ("this Python
cannot read zstd"), e `strata-inkling --export-tokenizer` devolve
`tokenizer_json: 0 bytes` (o pack não traz a seção). O que funciona é o passo 2:
`tokenizer.json` **e** `chat_template.jinja` ao lado do pack, de onde o servidor
monta `vocab.json`/`merges.txt`/`token_type.json` na primeira subida. Por isso
`prepare_runtime` baixa os dois do próprio repo do pack e o run config aponta
`tokenizer` para `<BYOM_DIR>/strata-tokenizer` (diretório gravável; o default do
control, `<control>/pack/full/tokenizer`, não existe no tarball).

Duas divergências de fonte, resolvidas com o binário na mão (2026-11-27):

1. **`--kv`.** A tabela de `docs/TERMINAL_LAUNCH.md` do control diz
   `int8 (default), q4_0, k8v4, fp16`, mas o engine 0.1.70 responde
   `unknown --kv value 'int8' (expected: auto, fp8, fp16, k8v8, k8v4)` — `int8`
   é formato do launcher/Qwen, não do Inkling (`strata_cli/launch/settings.py`,
   `INKLING_KV_MODES_CUDA`). O perfil declara `kv_cache_dtype = "fp8"` (1 byte
   por elemento, o mesmo 8 bits do `int8` do launcher, fator 0,5) e o argv emite
   `--kv fp8`; `available_profiles` recusa `int8`, `k8v8`, `k8v4` e `q4_0`.
2. **`SHA256SUMS.txt`.** O arquivo da revisão `5623ceb` tem 377 B e lista quatro
   packs — SQ-10, SQ-15, SQ-20 (o legado) e o de visão —, **não** o SQv2-20.
   `prepare_runtime` confere contra a linha do sums quando ela existe e, quando
   não existe, contra o `lfs.oid` do tree do HF (que **é** o sha256 do conteúdo:
   bate com o sums nos quatro packs que ele cobre). O manifest grava de onde veio
   o digest (`pack_sha256_source`), e divergência em qualquer das duas fontes é
   falha dura.

## Contexto acima do documentado

`max_model_len` não pode passar do `context_doc` da variante: um perfil que pede
400.000 num modelo de 262.144 é recusado na carga com `ConfigError`. Quando o
contexto servido fica abaixo do documentado, a variante declara o motivo em
`context_reason` e a TUI mostra esse texto em vez de recalcular a GPU mínima por
conta própria.

## Onde os números aparecem

A mesma conta alimenta `byom models`, o submenu `Selecionar GPU` da TUI, o
`check_compatibility` do `config.toml` e o log de atualização. A tabela dos 25
perfis (GPU, contexto, KV, required, folga, margem, `fits`, origem do KV e
drafter) é regerada com `scripts/vram_table.py` e registrada em
[Update Log](/log.md); sondas de comportamento ficam em
`scripts/probe_revisao.py`.

[^catalog-module]: `/src/byom/catalog.py`
[^worker]: `/worker/build_vllm_command.py` (`build_llama_args`, `build_strata_args`)
[^prepare]: `/worker/prepare_runtime.py` (`check_nvidia_driver`, `_prepare_strata`)
[^inkling-config]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json (consultado em 2026-11-24)
[^strata-card]: https://huggingface.co/strata-engine/Inkling-2-Mini-Strata (revisão `5623ceb`, consultado em 2026-11-27)
[^strata-release]: https://github.com/Niko1221/Strata/releases/tag/v0.2.23 (consultado em 2026-11-27)
[^log-2026-11-24]: `/log.md` — entrada "KV por arquitetura (2026-11-24)"
