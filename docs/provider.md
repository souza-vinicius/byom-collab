---
type: Architecture
title: Model Provider and Weight Package
description: Two different names the interface shows for every model — the team that develops it (catalog data with a dated source) and the organization that publishes the weights byom downloads (derived from the repository at render time) — and where each one appears.
tags: [catalog, fornecedor, quantizacao, tui, cli]
status: stable
generated: { by: hermes/deepseek-v4.1-flash, at: 2026-11-24T23:55:00Z }
sources:
  - id: catalog-module
    resource: /src/byom/catalog.py
    title: PROVIDER_SOURCES, PACKAGE_OWNER_NAMES, package_owner and package_source
  - id: mensagens-module
    resource: /src/byom/mensagens.py
    title: Labels, fornecedor_resumo/fornecedor_curto and glossary entries
  - id: profiles
    resource: /src/byom/profiles
    title: TOML profiles that pin the repository the weights come from
  - id: log-2026-11-24
    resource: /log.md
    title: Update log with the provider field and the AtomicChat variants
---

# Model Provider and Weight Package

Um modelo tem dois nomes próprios diferentes, e o byom mostra os dois:

- **Fornecedor** — o time/empresa que **desenvolve** o modelo. Dado do catálogo
  (`ModelEntry.provider`), escrito à mão, com a fonte registrada em
  `catalog.PROVIDER_SOURCES` (URL + data). `validate_catalog` recusa modelo sem
  fornecedor e fornecedor sem fonte.[^catalog-module]
- **Publicado por** / **Pacote** — a org do repositório de onde o byom **baixa**
  os pesos. Não é dado de catálogo: é derivado em tempo de render do perfil.

Exemplo real: `Inkling 2 Mini` → fornecedor *Thinking Machines Lab*; pacote
*Unsloth* nos perfis `inkling-2-mini-{iq2,q4,q8}-gguf` e *AtomicChat* nos perfis
`inkling-2-mini-ad-*`. Mesmo modelo, receitas de quantização diferentes. Outro
caso: `qwen-image-2.1-turbo-uncensored` → fornecedor *Qwen (Alibaba Cloud)*;
pacote *AtomicChat* nos dois perfis, porque o denoiser Turbo e o encoder
abliterado vêm dos repositórios da AtomicChat.

## De onde vem o pacote

`package_source(worker, model)` escolhe o repositório na ordem do runtime:[^catalog-module]

| runtime | campo lido | exemplo |
| :--- | :--- | :--- |
| `llama-cpp` | `[worker].model_repo` | `unsloth/Inkling-2-Mini-GGUF` |
| `image` | `[worker.image_roles].diffusion` | `leejet/Qwen-Image-2.1-GGUF`, `AtomicChat/Qwen-Image-2.1-Turbo-GGUF` |
| `vllm` | `[config].model` | `philbert440/Qwen3.8-27B-W4A16-AWQ` |

`package_owner(spec)` fica só com a org (`org/repo::pattern` → `org`) e traduz via
`PACKAGE_OWNER_NAMES` (`unsloth` → `Unsloth`, `qwen` → `Qwen (Alibaba Cloud)`).
Org fora do mapa aparece como está escrita no repositório: derivar é o padrão,
embelezar é a exceção registrada.

Trocar o repositório de um perfil muda o rótulo exibido sem editar o catálogo —
não existe campo `publisher` no TOML, e um campo assim envelheceria calado.

## Onde aparece

- `byom profiles`: linha `Fornecedor: …` e, quando o pacote diverge, `Pacote: …`.
- `byom models --gpu <ID>`: colunas `Fornecedor` e `Pacote` (a segunda só entra
  quando algum perfil daquela GPU publica pesos de outra org).
- TUI, lista de perfis e tabela de modelos por GPU: o fornecedor vai na segunda
  linha da célula `Modelo`. A célula quebra (`overflow="fold"`) em vez de truncar
  as vizinhas: com 100 colunas a tabela já ocupava a linha inteira. Com 160+
  colunas existe a coluna `Fornecedor` dedicada; com 120+ o texto inclui
  `· pesos <publicador>`. Em cartões (< 80 colunas) a linha do fornecedor abre o
  painel, com os dois nomes.
- TUI, detalhe do perfil: `Fornecedor: …` e, se divergir, `Publicado por: …`;
  quando os dois são iguais só a primeira linha aparece (`mensagens.linhas_fornecedor`).

## O que este campo não é

- **Não** é a licença nem o dono do checkpoint vLLM: `qwen3.8-27b-awq` roda o
  repositório AWQ de `philbert440` (pacote), mas o modelo é da Qwen/Alibaba Cloud.
- **Não** autoriza inventar empresa: onde o card não cita a empresa, o valor é o
  nome de exibição da org do Hugging Face, e a nota em `PROVIDER_SOURCES` diz
  isso. Caso registrado: `glm-5.3-flash` → `Z.ai`, porque "Zhipu AI" não aparece
  nas páginas consultadas.

[^catalog-module]: `src/byom/catalog.py` — `PROVIDER_SOURCES`, `package_owner`,
    `package_source` e os dois formatadores da CLI.
