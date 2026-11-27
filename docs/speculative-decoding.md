---
type: Architecture
title: Speculative Decoding in the Worker
description: How [worker.spec] and the legacy dspark shorthand become llama-server flags, which spec types embed the draft head in the main file, and how the VRAM budget charges the drafter file.
tags: [spec, mtp, dspark, dflash, llama-cpp, drafter, vram]
status: stable
generated: { by: hermes/deepseek-v4.1-flash, at: 2026-11-26T22:30:00Z }
sources:
  - id: worker-build
    resource: /worker/build_vllm_command.py
    title: Spec resolution, embedded-drafter types and the llama-server argv
  - id: worker-prepare
    resource: /worker/prepare_runtime.py
    title: Download of the main file, mmproj and separate drafter
  - id: config
    resource: /src/byom/config.py
    title: Profile-load validation of [worker.spec] and [worker.sampling]
  - id: catalog
    resource: /src/byom/catalog.py
    title: drafter_gib and the variant budget
  - id: prism-llamacpp
    resource: https://docs.prismml.com/run/llamacpp.md
    title: MTP and DFlash2 commands for Bonsai 2 (consultado em 2026-11-24)
  - id: prism-27b
    resource: https://docs.prismml.com/models/bonsai-2-5-27b.md
    title: Per-device drafts and cut-off, and the rule that an MTP file loads instead of the plain one
  - id: fork-help
    resource: https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b12128-d7b2896
    title: llama-server --help of the pinned fork build (prism-b12128-d7b2896)
  - id: b12800
    resource: https://github.com/ggml-org/llama.cpp/releases/tag/b12800
    title: Release oficial b12800 (1.1.0-dev) e o --help executado localmente (2026-11-26)
  - id: pr-32218
    resource: https://github.com/ggml-org/llama.cpp/pull/32218
    title: Inkling 2 DSpark support, mergeado em 2026-11-24 e incluído na v1.1.0
---

# Speculative Decoding in the Worker

Speculative decoding is a per-profile capability of the `llama-cpp` runtime: a
small draft model — or a draft **head** baked into the main file — proposes
tokens, and the full model accepts the ones it would have produced itself. The
output distribution does not change; only speed does.

## Consequence

Without a way to express it, the worker could only serve DSpark (hard-coded
`--spec-type draft-dspark --spec-draft-n-max 3`), and the Bonsai 2.5 27B — whose
MTP file carries the head **inside** the same `.gguf` — would either run without
drafting or need a second model it does not have.

## The `[worker.spec]` table

```toml
[worker.spec]
type = "draft-mtp"     # obrigatório; sem ele o load do perfil falha
n_min = 0              # --spec-draft-n-min
n_max = 3              # --spec-draft-n-max
p_min = 0.1            # --spec-draft-p-min
# drafter_pattern = "arquivo.gguf"   # só quando o rascunho é um SEGUNDO arquivo
```

`dspark = true` continua aceito e equivale a `type = "draft-dspark"` com
`n_max = 3` — o argv que os perfis DSpark já geravam é idêntico. Declarar
`dspark` e `[worker.spec]` juntos é erro: são a mesma configuração.

## Two shapes, one flag set

| Forma | Arquivo principal | Flags |
| :--- | :--- | :--- |
| Cabeça embutida (`draft-mtp`) | `-m ...-mtp.gguf` | `--spec-type`, `--spec-draft-n-min/-n-max/-p-min` |
| Rascunho separado (`draft-dspark`, `draft-dflash`, `draft-eagle3`, `draft-simple`) | `-m modelo.gguf` | as mesmas + `--model-draft <arquivo>` e `--n-gpu-layers-draft 99` |

O MTP do Bonsai é o primeiro caso: "Load an MTP file instead of the plain one,
not alongside it" (model page, 2026-11-24). O segundo arquivo nunca aparece no
argv, e `prepare_runtime` não grava `manifest["drafter"]`.

## VRAM

`catalog.drafter_gib` cobra o rascunho **só quando existe arquivo separado**
(`drafter_pattern`, no `[worker]` ou dentro da spec). Cabeça embutida custa
zero: o peso extra já está no arquivo contado em `Variant.memory_gib`. O valor
0,47 GiB é o do `dspark-Inkling-2-Mini-Q8_0.gguf`; outro rascunho precisa de
constante própria, porque a função não estima tamanho.

## Validation at load, not at runtime

`config._validate_spec` recusa, antes de alocar GPU ou baixar pesos: tabela
vazia, `type` ausente ou vazio, chave desconhecida, `n_min`/`n_max` não
inteiros ou negativos, `n_min > n_max`, `p_min` fora de 0..1, runtime diferente
de `llama-cpp`, e tipo sem cabeça embutida sem `drafter_pattern`.

## Flag names come from the binary

A fonte de sintaxe é o `--help` do binário fixado, não a memória. No build
oficial **b12800** (`llama-server --version` → `1.1.0-dev`, build 12800, commit
`47d6192fc`, executado em 2026-11-26): `-fa, --flash-attn [on|off|auto]`,
`--spec-type
none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,draft-dflash2,draft-lilicorr,ngram-...`,
`--spec-draft-n-min`, `--spec-draft-n-max`, `--spec-draft-p-min`,
`-md/--model-draft`, `-ngld/--n-gpu-layers-draft`, além de
`--presence-penalty`, `--repeat-penalty`, `--repeat-last-n` e
`--min-p ... (default: 0.05, 0.0 = disabled)`. O build do fork PrismML
(`prism-b12128-d7b2896`) continua sendo a fonte dos presets MTP do Bonsai,
medidos nele.

## Where it is used today

`inkling-2-mini-ad-q6k-dspark` é o perfil embarcado com DSpark: `dspark = true`,
`drafter_pattern = "dspark-Inkling-2-Mini-Q8_0.gguf"` (503.857.696 B, no mesmo
`model_repo` dos pesos) e binário oficial **b12800** — o primeiro release do
llama.cpp com `--spec-type draft-dspark`, depois que o PR #32218 entrou na
**v1.1.0**. O arquivo é um GGUF `general.architecture = dflash` com
`dflash.has_markov_context = true` e `target_layers = [5, 23, 41, 48]`; é essa
cabeça de Markov que caracteriza o DSpark em relação ao `dflash` simples. O
orçamento: 22,15 pesos + 0,65 visão + 2 runtime + 2,08 de KV em 128K + 0,47 do
drafter = 27,35, mais 2,0 de folga = 29,35 <= 40 GiB da A100-40. Nenhum ganho
de velocidade foi medido no Colab: o perfil é válido por construção (argv,
pin e orçamento), e a primeira execução real ainda vai dizer quanto o decode
acelera na A100-40.

`ternary-bonsai-25-27b-ptq1-mtp` (ternary Bonsai 2.5 27B, 64K na T4) declara
`type = "draft-mtp"` com `n_min = 0`, `n_max = 3`, `p_min = 0.1`, valores do
preset `presets/llama.cpp/mtp-cuda.json` do repositório do modelo, medidos no
build `prism-b12128-d7b2896`. A tabela por device do publicador cobre H200,
H100, RTX 5090, RTX 4090, M4 Pro e Core Ultra — **não** cobre T4, então o ganho
no Colab não é conhecido. Esse perfil não usa `--model-draft`: a cabeça vem no
próprio arquivo `-mtp.gguf`, carregado como `-m`.
