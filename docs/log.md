# Update Log

## 2026-11-24
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
