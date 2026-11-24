# Update Log

## 2026-11-24
- **Fix**: `up` grava `state.json` imediatamente depois de `colab new` (endpoint vazio) e tenta enriquecer o endpoint em seguida; falha em `colab status` não apaga mais o registro. Qualquer falha posterior — inclusive `Ctrl+C`/`SystemExit` — encerra a VM pelo nome com `colab stop -s <nome>` e confirma com `is_allocated`, sem depender do `state.json`.
- **Fix**: `colab.unregistered_sessions(known)` devolve as linhas `[?]` e as sessões `byom-*` ausentes do `state.json`; `_check_orphans` lista nome, endpoint e o comando de parada. `config.up_lock()` toma um `flock` não bloqueante em `up.lock` durante todo o `up` (CLI e TUI): o segundo processo aborta com "já existe um up em andamento" antes de alocar GPU.
- **Fix**: `config.load_config()` traduz `tomllib.TOMLDecodeError` em `ConfigError` com caminho, linha e coluna. A TUI lê `config.toml` por `_carregar_config()`, renderiza o painel com a ação de corrigir/mover o arquivo e volta ao menu; o menu inicial continua desenhando com a barra em `erro na config`. `state.json` ilegível passou a bloquear a troca de perfil (nenhuma escrita) e o `status` mostra `estado ilegível` em vez de `sem sessão registrada`.
- **Fix**: "Voltar para a lista" no detalhe do perfil volta à lista de perfis (e à de GPUs no caminho GPU + modelo), não ao menu principal.
- **Update**: orçamento de VRAM com folga obrigatória `max(1 GiB, 5% da VRAM)`, custo do projetor de visão (0,65 GiB) somado em `required_vram_gib` e T4 com 15 GiB **utilizáveis**. Nenhum contexto servido foi reduzido: cinco perfis mudaram de GPU (Inkling 2 bit e Muse 2 bit → L4; Qwen3.8 3 bit → L4; Inkling 4 bit e Muse 4 bit → A100 40 GB). Tabela antes → depois:

  | perfil | GPU antes → depois | ctx | required antes | VRAM antes | cabia antes? | required depois | headroom | margem | fits |
  | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- |
  | `gemma-4-12b-q4-gguf` | T4 → T4 | 32.768 | 12.00 | 16 | sim | 12.00 | 1.00 | 3.00 | sim |
  | `gemma-4-26b-a4b-q4-gguf` | L4 → L4 | 32.768 | 22.00 | 24 | sim | 22.65 | 1.20 | 1.35 | sim |
  | `gemma-4-31b-q4-gguf` | A100-40 → A100-40 | 131.072 | 30.00 | 40 | sim | 30.65 | 2.00 | 9.35 | sim |
  | `gemma-4-e4b-q4-gguf` | T4 → T4 | 65.536 | 12.00 | 16 | sim | 12.00 | 1.00 | 3.00 | sim |
  | `inkling-2-mini-iq2-gguf` | T4 → L4 | 32.768 | 15.00 | 16 | sim | 15.65 | 1.20 | 8.35 | sim |
  | `inkling-2-mini-q4-gguf` | L4 → A100-40 | 65.536 | 24.00 | 24 | sim | 24.65 | 2.00 | 15.35 | sim |
  | `inkling-2-mini-q8-gguf` | A100-40 → A100-40 | 131.072 | 34.00 | 40 | sim | 34.65 | 2.00 | 5.35 | sim |
  | `muse-glimmer-30b-q2-gguf` | T4 → L4 | 32.768 | 16.00 | 16 | sim | 16.65 | 1.20 | 7.35 | sim |
  | `muse-glimmer-30b-q4-gguf` | L4 → A100-40 | 65.536 | 23.00 | 24 | sim | 23.65 | 2.00 | 16.35 | sim |
  | `qwen-image-2.1-fp8` | L4 → L4 | 1 | 22.00 | 24 | sim | 22.00 | 1.20 | 2.00 | sim |
  | `qwen-image-2.1-q4km` | T4 → T4 | 1 | 14.00 | 16 | sim | 14.00 | 1.00 | 1.00 | sim |
  | `qwen3.8-27b-256k` | A100-80 → A100-80 | 262.144 | 35.00 | 80 | sim | 35.00 | 4.00 | 45.00 | sim |
  | `qwen3.8-27b-32k` | A100-40 → A100-40 | 32.768 | 21.00 | 40 | sim | 21.00 | 2.00 | 19.00 | sim |
  | `qwen3.8-27b-q3-gguf` | T4 → L4 | 32.768 | 16.00 | 16 | sim | 16.00 | 1.20 | 8.00 | sim |
  | `qwen3.8-27b-q4-gguf` | L4 → L4 | 32.768 | 21.00 | 24 | sim | 21.00 | 1.20 | 3.00 | sim |
- **Update**: Tela inicial única em [Interactive TUI Architecture](./tui.md) — sete ações e `q. Sair`, atalhos `1`–`5` / `c` / `h` / `q`; `h` passa a ser `Configurar harness` e a Ajuda sai do menu (constante `MENU_AJUDA` e ramo `__ajuda__` removidos). `GRUPOS_MENU` foi substituído por `MENU_INICIAL` em `src/byom/mensagens.py`.
- **Update**: `profiles`, `gpus` e `models` deixam de ser itens de topo e passam ao submenu `Selecionar GPU` (atalho `2`), que oferece perfil pronto ou par GPU + modelo.
- **Addition**: caminho GPU + modelo grava `cfg.gpu`/`cfg.gpu_high_mem` derivados de `catalog.colab_settings_for_gpu()` e valida com `config.check_compatibility()` antes de salvar; par que não cabe não é gravado.
- **Update**: `proximo_passo_recomendado()` passa a recomendar `selecionar_gpu` no lugar de `configure`, e `TECLAS_MENU` descreve os atalhos atuais.

## 2026-11-23
- **Fix**: `byom chat` e a tela `chat` tratam `Ctrl+C`/`Ctrl+D` como encerramento normal (`Conversa encerrada`) e exibem o stderr real do `llm` em `Detalhes técnicos` quando a conversa falha de verdade.
- **Update**: [Interactive TUI Architecture](./tui.md) now documents the `chat` screen, the animated step spinner fix and SSH key validation before `up`.
- **Addition**: Added [Isolated llm chat Integration](./llm-chat.md) documenting centralized `byom_home()` resolution and the `setup`, `chat`, and `uninstall` CLI commands (`src/byom/llm_chat.py`, `src/byom/config.py`, `src/byom/cli.py`).
- **Update**: Added `pstack-skill` under `.agents/skills/pstack-skill/` adapted for Antigravity and OpenCode (session-model inheritance via `.agents/pstack-models.md`, harness tool/path mappings, and mandatory session-start rule in `AGENTS.md`).
- **Update**: Updated [Interactive TUI Architecture](./tui.md) with the explicit harness selection screen (`harness`) providing manual instructions and opt-in automatic setup for OpenCode/`dsh`, decoupling it from TUI `up`.
- **Update**: Added [Interactive TUI Architecture](./tui.md) documenting the redesigned `src/byom/tui.py` and `src/byom/mensagens.py` presentation layer.
- **Initialization**: Created `docs/` OKF v0.2 knowledge base bundle with `index.md` and `log.md`.
