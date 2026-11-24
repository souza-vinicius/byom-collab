---
type: Architecture
title: Interactive TUI Architecture
description: Structured Rich + questionary TUI layer for byom with PT-BR formatting, responsive layout, and live step checklists.
tags: [tui, ui, rich, questionary, pt-br]
status: stable
generated: { by: antigravity/gemini-4-argon, at: 2026-11-23T22:40:00Z }
sources:
  - id: tui-module
    resource: /src/byom/tui.py
    title: TUI presentation and flow controller
  - id: messages-module
    resource: /src/byom/mensagens.py
    title: Centralized PT-BR messages, translations, glossary, and formatters
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

- **Local-only status bar**: `ler_estado_local()` inspects local `config.toml`, SSH key existence, and `state.json` without making network or Colab CLI calls. `proximo_passo_recomendado()` highlights the next logical step (`init` -> `configure` -> `up` -> `status`).
- **Grouped menu & prerequisite handling**: Actions are grouped under `Preparar`, `Explorar catálogo`, and `Usar sessão no Colab` plus a final `Conversar` group, with shortcuts `1`–`9`, `c`, `h`, and `q`. Selecting an action with an unmet prerequisite offers jumping directly to the required step.
- **Explicit harness configuration screen (`harness`)**: On the TUI, starting a session (`up`) no longer edits OpenCode/dsh configurations implicitly. Instead, the `Configurar harness (OpenCode / dsh)` screen lets the user choose a harness (`OpenCode`, `dsh`, `Aider`, `Cline`), displays step-by-step manual instructions and config snippets for the active profile/endpoint, and offers an explicit option to configure the OpenCode or `dsh` connection automatically.
- **Chat screen (`chat`)**: `Conversar com o modelo (llm chat)` requires an active session (otherwise it offers to go to `up`), re-syncs the isolated `llm` setup to the current local endpoint via `llm_chat.run_setup`, asks whether to save history, then runs `llm_chat.run_chat`. It is also offered as the first next step after `up`. Leaving the chat with `Ctrl+C` or `Ctrl+D` is reported as `Conversa encerrada`; real failures show the captured `llm` stderr in `Detalhes técnicos`. See [Isolated llm chat Integration](/llm-chat.md).
- **Responsive layout**:
  - `>= 100` columns: full Rich tables.
  - `80–99` columns: compact single-line tables without mid-column wrapping.
  - `< 80` columns: vertical cards.
  - `<= 26` rows: pagination hint displayed above long listings.
- **Live execution & error reporting**: Side-effect operations (`init`, `up`, `down`) show an objective pre-action summary, ask for confirmation (`default=False` for `up` and `down`), execute via `ExecutorEtapas` (animated spinner, per-step elapsed time, slow-step hints and SSH attempt counter; it pins `console.file` so the live refresh thread keeps writing to the real terminal while stdout is redirected to capture raw output), and display raw logs only inside a dimmed `Detalhes técnicos` panel on failure or inspection.

[^tui-module]: TUI presentation and flow controller (`src/byom/tui.py`)
[^messages-module]: Centralized PT-BR messages, translations, glossary, and formatters (`src/byom/mensagens.py`)
[^cli-module]: CLI entrypoint and command logic (`src/byom/cli.py`)
