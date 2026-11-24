---
type: Architecture
title: Interactive TUI Architecture
description: Structured Rich + questionary TUI layer for byom with a single-screen PT-BR menu, GPU + model selection, responsive layout, and live step checklists.
tags: [tui, ui, rich, questionary, pt-br]
status: stable
generated: { by: hermes/deepseek-v4.1-flash, at: 2026-11-24T01:55:56Z }
sources:
  - id: tui-module
    resource: /src/byom/tui.py
    title: TUI presentation and flow controller
  - id: messages-module
    resource: /src/byom/mensagens.py
    title: Centralized PT-BR messages, translations, glossary, and formatters
  - id: catalog-module
    resource: /src/byom/catalog.py
    title: Single GPU/model catalog and the catalog <-> config.toml mapping
  - id: cli-module
    resource: /src/byom/cli.py
    title: CLI entrypoint and command logic
---

# Interactive TUI Architecture

`byom` exposes an interactive terminal interface when invoked without arguments in a real TTY (or with `--tui`), while preserving the traditional CLI when `--no-tui`, pipes, or subcommands are used.[^cli-module]

## Separation of Concerns

- **Catalog and business logic remain untouched**: [src/byom/catalog.py](/src/byom/catalog.py) and [src/byom/cli.py](/src/byom/cli.py) are not rewritten for presentation.
- **Structured rendering**: [src/byom/tui.py](/src/byom/tui.py) reads structured data (`config.available_profiles()`, `catalog.GPUS`, `catalog.compatible_variants()`, `config.load_state()`) and renders Rich tables, panels, and checklists with `highlight=False` and semantic styles.
- **Centralized PT-BR copy & translation layer**: [src/byom/mensagens.py](/src/byom/mensagens.py) owns all UI text, catalog translations (`DESCRICOES_PERFIS`, `MOTIVOS_CONTEXTO_PERFIS`, `NOTAS_MODELOS`, `NOTAS_GPUS`), contextual glossary entries (`GLOSSARIO`), and pure PT-BR formatters (`formatar_numero`, `formatar_gib`, `formatar_contexto`, `formatar_data_ptbr`, `encurtar_caminho`, `traduzir_raciocinio`).

## Navigation, Status Bar, and Responsiveness

- **Local-only status bar**: `ler_estado_local()` inspects local `config.toml`, SSH key existence, and `state.json` without making network or Colab CLI calls. `proximo_passo_recomendado()` highlights the next logical step (`init` -> `selecionar_gpu` -> `up` -> `status`).
- **Single-screen menu**: one flat list of seven actions plus the fixed `q. Sair` exit, with shortcuts `1`–`5`, `c`, and `h`. The item marked `recomendado` follows `proximo_passo_recomendado()`, and items with an unmet prerequisite carry a bracketed warning (`[requer preparar ambiente]`, `[sem sessão ativa]`, `[requer encerrar sessão]`). Selecting an action with an unmet prerequisite offers jumping directly to the required step.

  ```
  1. Preparar ambiente local
  2. Selecionar GPU
  3. Iniciar sessão no Colab
  4. Verificar estado da sessão
  5. Encerrar sessão no Colab
  c. Conversar com o modelo (llm chat)
  h. Configurar harness
  q. Sair
  ```

- **`Selecionar GPU` submenu (shortcut `2`)**: replaces the former top-level catalog screens (`profiles`, `gpus`, `models`) with one entry point that offers two paths and a way back.
  - *Selecionar perfil de modelo* reuses the `configure` flow: pre-configured profile list, detail screen, and `Usar este perfil` with the active-session guard.
  - *Selecionar uma GPU para ver os modelos compatíveis* renders the GPU table, then the compatible-model table for the chosen GPU, then an explicit confirmation panel (`GPU: L4 • Modelo: Qwen3.8 27B — 4 bits (GGUF)`, served context, estimated memory and the resulting `gpu`/`gpu_high_mem` values). The pair is written only after the confirmation and the `_confirmar` prompt.
- **GPU pinning on the GPU + model path**: `config.apply_profile()` copies `profile.settings`, which includes `gpu` and `gpu_high_mem`. On that path the chosen GPU wins: after applying the profile, the TUI sets `cfg.gpu`/`cfg.gpu_high_mem` from `catalog.colab_settings_for_gpu(gpu_id)` (`A100-40` -> `gpu="A100"`, `high_mem=false`; `A100-80` -> `gpu="A100"`, `high_mem=true`; others -> `gpu=<id>`, `high_mem=false`) and validates the pair with `config.check_compatibility(cfg)` before `config.save_config`. A pair that does not fit shows the required VRAM and writes nothing. The pre-configured-profile path keeps using the GPU declared by the profile.[^catalog-module]
- **Active-session guard**: `_troca_bloqueada_por_sessao()` blocks both paths while `state.json` exists, offering `Encerrar sessão no Colab` instead; `_aplicar_perfil_interativo()` checks `config.load_state()` again right before writing.
- **Explicit harness configuration screen (`harness`, shortcut `h`)**: On the TUI, starting a session (`up`) no longer edits OpenCode/dsh configurations implicitly. Instead, the `Configurar harness (OpenCode / dsh)` screen lets the user choose a harness (`OpenCode`, `dsh`, `Aider`, `Cline`), displays step-by-step manual instructions and config snippets for the active profile/endpoint, and offers an explicit option to configure the OpenCode or `dsh` connection automatically.
- **Chat screen (`chat`)**: `Conversar com o modelo (llm chat)` requires an active session (otherwise it offers to go to `up`), re-syncs the isolated `llm` setup to the current local endpoint via `llm_chat.run_setup`, asks whether to save history, then runs `llm_chat.run_chat`. It is also offered as the first next step after `up`. Leaving the chat with `Ctrl+C` or `Ctrl+D` is reported as `Conversa encerrada`; real failures show the captured `llm` stderr in `Detalhes técnicos`. See [Isolated llm chat Integration](/llm-chat.md).
- **Responsive layout**:
  - `>= 100` columns: full Rich tables.
  - `80–99` columns: compact single-line tables without mid-column wrapping.
  - `< 80` columns: vertical cards.
  - `<= 26` rows: pagination hint displayed above long listings.
- **Live execution & error reporting**: Side-effect operations (`init`, `up`, `down`) show an objective pre-action summary, ask for confirmation (`default=False` for `up` and `down`), execute via `ExecutorEtapas` (animated spinner, per-step elapsed time, slow-step hints and SSH attempt counter; it pins `console.file` so the live refresh thread keeps writing to the real terminal while stdout is redirected to capture raw output), and display raw logs only inside a dimmed `Detalhes técnicos` panel on failure or inspection.

[^tui-module]: TUI presentation and flow controller (`src/byom/tui.py`)
[^messages-module]: Centralized PT-BR messages, translations, glossary, and formatters (`src/byom/mensagens.py`)
[^catalog-module]: Single GPU/model catalog and the catalog <-> config.toml mapping (`src/byom/catalog.py`)
[^cli-module]: CLI entrypoint and command logic (`src/byom/cli.py`)
