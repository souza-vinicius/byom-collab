# pstack no Antigravity

Mapeamento de ferramentas, subagentes, modelos e caminhos para executar a `pstack-skill` no **Antigravity**. Leia este arquivo uma vez no início da sessão sempre que estiver rodando no Antigravity.

## Ferramentas (Tools)

| Referência genérica / Cursor | Antigravity |
| --- | --- |
| `Task` (delegar subagente) | `invoke_subagent` (ou `define_subagent` / `send_message` / `manage_subagents`; `run_workflow` para fan-out em Python) |
| `Read` | `read_file` |
| `Write` | `write_file` |
| `Edit` | `edit_file` |
| `Bash` | `run_command` (`manage_task` para tarefas em background; `schedule` para timers/cron) |
| `Grep`, `Glob` | `run_command` com `rg` ou `find` |
| `AskQuestion` | `ask_question` (múltipla escolha) ou resposta direta em texto para perguntas simples |
| `WebSearch` | `search_web` |
| `TodoWrite` / abrir todolist | Manter a lista de tarefas estruturada na resposta ou em artefato dentro de `~/.gemini/antigravity-cli/brain/<conversation-id>/` |
| Executar procedure (`**nome**`) | `read_file` em `.agents/skills/pstack-skill/references/skills/<nome>.md` e seguir os passos |
| Abrir playbook | `read_file` em `.agents/skills/pstack-skill/playbooks/<playbook>.md` e copiar os passos |

## Subagentes (`invoke_subagent`)

| Campo / padrão | Antigravity (`invoke_subagent`) |
| --- | --- |
| `subagent_type: generalPurpose`, `readonly: false` | `TypeName: "self"`, `Role: "<descrição curta do papel>"` |
| `readonly: true` | `TypeName: "research"` |
| `environment: "cloud"` / escrita paralela em worktree | `Workspace: "share"` (cópia compartilhando repositório) ou `Workspace: "branch"` (clone isolado) |
| `environment: "local"` / somente leitura | `Workspace: "inherit"` (padrão) |
| `model` | `Model: "inherit"` — **sempre herda o modelo ativo da sessão** |
| Primeiro conteúdo do `Prompt` | `Read /home/vinicius/Projetos/byom-collab/.agents/skills/pstack-skill/worker.md before anything else.` |
| `resume` / continuar subagente existente | `send_message` usando o `conversationId` retornado por `invoke_subagent` |
| `file_attachments` | Listar os caminhos absolutos diretamente no `Prompt` |

## Modelos

- O arquivo [`.agents/pstack-models.md`](../../../../pstack-models.md) define `inherit-parent` para todos os papéis (`worker`, `builder`, `judge`, `peer`).
- No Antigravity, passe sempre `Model: "inherit"` em `invoke_subagent` para executar **no modelo ativo da sessão atual**.
- Painéis multi-modelo (`arena`, `interrogate`) executam passes sequenciais ou paralelos independentes com contexto limpo (`TypeName: "self"`) usando o mesmo modelo da sessão, sem tentar invocar slugs externos.

## Caminhos (Paths)

| Conceito | Caminho no Antigravity |
| --- | --- |
| Skills do projeto | `.agents/skills/` (ex.: [`.agents/skills/pstack-skill/SKILL.md`](../../SKILL.md)) |
| Skills globais | `~/.gemini/config/skills/` |
| Regras sempre carregadas | `AGENTS.md`, `GEMINI.md`, `.agents/rules/*.md` |
| Artefatos e scratch | `~/.gemini/antigravity-cli/brain/<conversation-id>/` |
| Transcrições da sessão | `~/.gemini/antigravity-cli/brain/<conversation-id>/.system_generated/logs/transcript.jsonl` |

## Recursos exclusivos do Cursor

O Antigravity não possui Cursor Automations, Bugbot nem Cursor cloud agents:
- Para passos de PR/review, utilize comandos da forge (`gh`) via `run_command` quando autorizado.
- Para worktrees isolados sem cloud agents, use `Workspace: "share"` / `"branch"` em `invoke_subagent` ou `git worktree` via `run_command`.
