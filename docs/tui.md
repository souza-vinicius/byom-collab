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
  h. Configurar harness (OpenCode / dsh)
  q. Sair
  ```

- **`Selecionar GPU` (shortcut `2`)**: one path only, GPU -> model. The screen renders the GPU table (VRAM and how many runnable models fit), the compatible-model table for the chosen GPU, and an explicit confirmation panel (`GPU: L4 • Modelo: Qwen3.8 27B — 4 bits (GGUF)`, optimized context, estimated memory and the resulting `gpu`/`gpu_high_mem` values). The pair is written only after the confirmation and the `_confirmar` prompt. The former profile-selection path was removed; profiles survive only as internal runtime recipes.
- **Optimized context window**: for every profile of a variant, `catalog.max_fitting_context()` binary-searches the largest multiple of 1024 tokens that fits the GPU budget, capped by `Variant.context_doc`. The variant enters the table with the largest result, so a T4 gets the window that fits its 15 GiB and an A100-80 gets the documented maximum. The TUI copies the chosen `PerfilView` with that context (`_visao_otimizada`), shows it in the table and in the confirmation panel, and `_aplicar_perfil_interativo(..., max_model_len=...)` writes it to `config.toml` after `config.check_compatibility` approves the pair. The CLI `byom models --gpu` and `byom gpus` use the same calculation through `catalog.compatible_variants()`.[^catalog-module]
- **GPU pinning**: `config.apply_profile()` copies `profile.settings`, which includes `gpu` and `gpu_high_mem`. The chosen GPU wins: after applying the internal recipe, the TUI sets `cfg.gpu`/`cfg.gpu_high_mem` from `catalog.colab_settings_for_gpu(gpu_id)` (`A100-40` -> `gpu="A100"`, `high_mem=false`; `A100-80` -> `gpu="A100"`, `high_mem=true`; remaining IDs -> `gpu=<colab_names[0]>`, `high_mem=false`, e.g. `H100-80` -> `gpu="H100"`) and validates the pair with `config.check_compatibility(cfg)` before `config.save_config`. A pair that does not fit shows the required VRAM and writes nothing.[^catalog-module]
- **Active-session guard**: `_troca_bloqueada_por_sessao()` reads `state.json` itself and blocks the selection while a session is registered, offering `Encerrar sessão no Colab` instead. A `state.json` that exists but cannot be read or parsed raises `config.StateFileError`; the guard does not release the switch in that case — it renders the error panel (check the Colab panel, then run `byom status`) because the file may hide an allocated VM. `_aplicar_perfil_interativo()` reads the state again right before writing and refuses to write anything under the same condition.
- **Explicit harness configuration screen (`harness`, shortcut `h`)**: On the TUI, starting a session (`up`) no longer edits OpenCode/dsh configurations implicitly. Instead, the `Configurar harness (OpenCode / dsh)` screen lets the user choose a harness (`OpenCode`, `dsh`, `Aider`, `Cline`), displays step-by-step manual instructions and config snippets for the active profile/endpoint, and offers an explicit option to configure the OpenCode or `dsh` connection automatically. For OpenCode the displayed path comes from `opencode.config_path()`, which accepts a `.jsonc`; the automatic option uses the stricter `resolve_config_path()` and, for JSONC, explains that the file is preserved and the block shown above must be inserted by hand. Reading the path for the manual panel never raises, so a JSONC-only setup shows the instructions instead of a generic internal failure.
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
