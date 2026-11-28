# AGENT_PROGRESS — Auditoria Profunda e Evolução Contínua (byom)

Objetivo: elevar CLI UX, fluxo Colab, chat LLM, testes e documentação sem regredir nenhum comando, flag ou fluxo existente.

Condição de parada: todos os itens abaixo em `[x]`, linter e suíte verde, probe e validação OKF verdes, Resumo Executivo preenchido.
Limite de segurança: 25 ciclos de execução.

Regras de ouro: zero regressões; escopo controlado (ideias novas só no BACKLOG.md); lint + testes após cada bloco; mudanças pequenas e commits atômicos; sem TODO, sem remoção de tratamento de erro, sem pular arquivos extensos.

## Baseline (Fase 1, medido antes de qualquer alteração)

- `env -u BYOM_COLAB_BIN uv run pytest -q`: 583 passed em 18,97 s.
- `uv run ruff check .`: All checks passed.
- `uv run python scripts/probe_revisao.py`: exit 0, todas as sondas passaram.
- `bash .agents/skills/okf-open-knowledge-format/scripts/validate.sh docs/`: exit 0, bundle OKF v0.2 conforme.
- Branch: `main`, árvore limpa no início.

## Fase 1 — Diagnóstico e Baseline

- [x] F1.1 Mapear entrypoint, parser de argumentos e comandos (`src/byom/cli.py`, `__main__.py`). Resumo: Typer com callback interativo, 12 comandos; `up`, `down`, `status`, `chat`, `init`, `configure`, `profiles`, `gpus`, `models`, `setup`, `uninstall`, `_daemon` oculto.
- [x] F1.2 Mapear autenticação/conexão Colab, provisionamento GPU/modelo, túnel e daemon (`colab.py`, `remote.py`, `tunnel.py`, `daemon.py`). Resumo: sessões remotas via `BYOM_COLAB_BIN`/CLI local, worker tar, daemon local de keepalive e proxy HTTP.
- [x] F1.3 Mapear loop do chat, TUI, mensagens e formatos de saída (`tui.py`, `llm_chat.py`, `mensagens.py`, `opencode.py`, `proxy.py`). Resumo: TUI em 2992 linhas com comandos slash, streaming via proxy OpenAI-compatible.
- [x] F1.4 Rodar baseline de pytest, ruff, probe de VRAM e validador OKF (resultados acima).
- [x] F1.5 Consolidar gargalos de UX e plano detalhado. Resumo: 3 auditorias read-only (CLI/Colab, TUI/chat, testes/docs); priorizar erros sem orientação, restauração de harness em falha de `up`, portas locais, Ctrl+C/EOF no chat, NO_COLOR e docs desatualizados.

## Fase 2 — Fluxos de Navegação e Experiência do Usuário no Terminal

- [x] F2.1 Ciclo de vida transparente. `up` agora anuncia 5 etapas numeradas (checagem, preparação, alocação, worker/modelo, espera); mensagem final de prontidão inalterada.
- [x] F2.2 Erros acionáveis (CLI). `init` mostra a saída real da CLI e orienta conforme o erro; timeout do worker inclui o último estado/motivo; `up` verifica também a porta do túnel e devolve o harness se falhar após sincronizá-lo; `down` devolve o harness mesmo sem confirmação; comparação de GPU ignora caixa e rejeita valor vazio. Ctrl+C/SIGINT no chat fica para o próximo ciclo.
- [x] F2.3 Ergonomia do chat. Ctrl+C no `llm chat` virou cancelamento limpo (código 130); EOF no prompt de voltar encerra o menu sem traceback; dica de Ctrl+D antes do chat; texto de cancelamento atualizado. Comandos slash, multilinha, histórico e confirmações existentes preservados.

## Fase 3 — Redesenho da Interface de Terminal e Clareza de Informações

- [x] F3.1 Hierarquia visual consistente. Painéis, símbolos ASCII/Unicode e separação prompt/resposta/log da TUI preservados; linha de abertura do chat usa o mesmo bullet e estilo das demais etapas.
- [x] F3.2 Feedback em tempo real. `up` emite 5 etapas numeradas e o último motivo em caso de timeout; spinner existente da TUI e streaming herdado do `llm chat` mantidos (nenhum renderizador novo criado — iria além do escopo).
- [x] F3.3 Textos diretos. `--help` traz a sequência `init → configure → up → status → chat → down`; ajuda do `chat` cita Ctrl+D; etapas do `up` e erros do `init`/timeout ganharam orientação concreta.
- [x] F3.4 Compatibilidade. `NO_COLOR` agora vale também para o questionary e o console é recriado se a variável mudar após a importação; EOF em modo não-interativo encerra sem traceback; layout 60/80/100/120 colunas segue coberto pelos testes existentes.

## Fase 4 — Revisão e Fortalecimento de Testes Unitários

- [x] F4.1 Testes de flags e comandos. Novos casos para `setup --model-id`, `chat -m`, `up --force/--opencode-config/--dsh-config` e comparação de hardware.
- [x] F4.2 Estados Colab com mocks. `_wait_ready` coberto nos estados ready, daemon morto, silêncio do daemon, desconexão e erro de cota; nenhum teste usa sessão real.
- [x] F4.3 Formatação e encerramento. Cobertos Ctrl+C no chat, EOF no prompt, mensagem de cancelamento, `NO_COLOR`, `_orphan_line` e restauração de harness.
- [x] F4.4 Suíte verde e determinística: 606 testes em ~19 s com `BYOM_COLAB_BIN` removido; ruff limpo.

## Fase 5 — Documentação e Backlog Estratégico

- [x] F5.1 README atualizado: Windows como pré-requisito, exemplos de `setup --model-id/--api-key` e `chat -m/--no-history`, fluxo em 5 etapas, tabela de erros com ação, runtime `strata` e 25 perfis.
- [x] F5.2 Docs sincronizadas (`llm-chat.md`, `tui.md`, `troubleshooting.md`, `connection-path.md`, `index.md`, `log.md`); validador OKF v0.2 passa com 8 conceitos.
- [x] F5.3 BACKLOG.md criado com 12 ideias estruturadas (problema, impacto, esforço, critérios de aceite); nenhuma foi implementada nesta rodada.

## Fase 6 — Revisão Final (QA) e Entrega

- [ ] F6.1 Diff review: código morto, imports não usados, logs de debug, vazamento de credenciais/tokens.
- [ ] F6.2 Linter, suíte completa, probe de VRAM e validação OKF reexecutados no estado final.
- [ ] F6.3 Resumo Executivo preenchido abaixo.

## Registro de ciclos

| Ciclo | Item | Resultado | Verificação |
| --- | --- | --- | --- |
| 0 | F1.1–F1.4 | Baseline medido | 583 testes, ruff/probe/OKF verdes |
| 1 | F1.5 | Auditorias consolidadas; plano priorizado | leitura de 3 relatórios, nenhuma edição |
| 2 | F2.1, F2.2 (CLI), F4 (parcial) | `cli.py` + 10 testes novos | ruff limpo; 593 testes passam |
| 3 | F2.3, F3.1–F3.4 | `llm_chat.py`, `tui.py`, `mensagens.py`, `cli.py`, 6 testes novos | ruff limpo; 599 testes passam |
| 4 | F4.1–F4.4 | 7 testes novos de flags/estados/formatação | ruff limpo; 606 testes passam |
| 5 | F5.1–F5.3 | README, 6 docs OKF, BACKLOG, log | OKF/Ruff/Pytest verdes; 606 testes passam |

## Resumo Executivo

A preencher na Fase 6.
