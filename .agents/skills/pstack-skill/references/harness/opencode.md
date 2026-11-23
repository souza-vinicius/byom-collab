# pstack no OpenCode

Mapeamento de ferramentas, subagentes, modelos e caminhos para executar a `pstack-skill` no **OpenCode** (incluindo sessões conectadas ao modelo `byom/<perfil>` no Google Colab em `http://127.0.0.1:4000/v1`). Leia este arquivo uma vez no início da sessão sempre que estiver rodando no OpenCode.

## Ferramentas (Tools)

| Referência genérica / Cursor | OpenCode |
| --- | --- |
| `Task` (delegar subagente) | `task` |
| `Read` | `read` |
| `Write` | `write` |
| `Edit` | `edit` |
| `Bash` | `bash` |
| `Grep` | `grep` |
| `Glob` | `glob` |
| `AskQuestion` | `question` |
| `TodoWrite` / abrir todolist | `todowrite` / `todoread` |
| `WebFetch` / `WebSearch` | `webfetch` / `websearch` |
| Executar procedure (`**nome**`) | `skill` (se registrada) ou `read` em `.agents/skills/pstack-skill/references/skills/<nome>.md` |
| Abrir playbook | `read` em `.agents/skills/pstack-skill/playbooks/<playbook>.md` e copiar os passos |

## Subagentes (`task`)

| Campo / padrão | OpenCode (`task`) |
| --- | --- |
| `subagent_type: generalPurpose`, `readonly: false` | `subagent_type: "general"` |
| `readonly: true` | `subagent_type: "explore"` |
| `environment: "cloud"` / escrita paralela | Criar worktree isolado com `git worktree add` via `bash` e passar o caminho no prompt do `task` |
| `model` | **Omitir override de modelo** (`inherit-parent`) — usa sempre o modelo em execução na sessão |
| Primeiro conteúdo do prompt | `Read .agents/skills/pstack-skill/worker.md before anything else.` |
| `resume` / continuar subagente | Passar o ID da sessão retornado pelo `task` quando aplicável, ou iniciar novo `task` com escopo consolidado |
| `file_attachments` | Listar os caminhos relativos ou absolutos diretamente no prompt |

## Modelos

- O arquivo [`.agents/pstack-models.md`](../../../../pstack-models.md) define `inherit-parent` para todos os papéis (`worker`, `builder`, `judge`, `peer`).
- No OpenCode, nunca passe slugs externos (como `opus`, `gpt-*` ou `claude-*`) em chamadas `task`. Omita o parâmetro de modelo para que todos os papéis herdem o **modelo ativo na sessão atual** (por exemplo, `byom/inkling-2-mini-q4-gguf` ou outro perfil selecionado no `byom`).
- Painéis multi-modelo (`arena`, `interrogate`) executam passes independentes com contexto limpo no modelo da sessão.

## Caminhos (Paths)

| Conceito | Caminho no OpenCode |
| --- | --- |
| Skills do projeto | `.agents/skills/` (descoberta nativa pelo OpenCode) e `.opencode/skills/` |
| Skills globais | `~/.config/opencode/skills/` |
| Regras carregadas | `AGENTS.md` na raiz do projeto e `instructions` em `~/.config/opencode/opencode.json` (`opencode.jsonc`) |
| Dados e sessões locais | `~/.local/share/opencode/` |

## Recursos exclusivos do Cursor

O OpenCode não possui Cursor Automations, Bugbot nem Cursor cloud agents:
- Para verificações e PRs, use `bash` com `gh` ou scripts do projeto.
- Para isolamento de escrita concorrente, crie worktrees git locais via `bash`.
