# Update Log

## 2026-11-23
- **Fix**: `byom chat` e a tela `chat` tratam `Ctrl+C`/`Ctrl+D` como encerramento normal (`Conversa encerrada`) e exibem o stderr real do `llm` em `Detalhes técnicos` quando a conversa falha de verdade.
- **Update**: [Interactive TUI Architecture](./tui.md) now documents the `chat` screen, the animated step spinner fix and SSH key validation before `up`.
- **Addition**: Added [Isolated llm chat Integration](./llm-chat.md) documenting centralized `byom_home()` resolution and the `setup`, `chat`, and `uninstall` CLI commands (`src/byom/llm_chat.py`, `src/byom/config.py`, `src/byom/cli.py`).
- **Update**: Added `pstack-skill` under `.agents/skills/pstack-skill/` adapted for Antigravity and OpenCode (session-model inheritance via `.agents/pstack-models.md`, harness tool/path mappings, and mandatory session-start rule in `AGENTS.md`).
- **Update**: Updated [Interactive TUI Architecture](./tui.md) with the explicit harness selection screen (`harness`) providing manual instructions and opt-in automatic setup for OpenCode/`dsh`, decoupling it from TUI `up`.
- **Update**: Added [Interactive TUI Architecture](./tui.md) documenting the redesigned `src/byom/tui.py` and `src/byom/mensagens.py` presentation layer.
- **Initialization**: Created `docs/` OKF v0.2 knowledge base bundle with `index.md` and `log.md`.
