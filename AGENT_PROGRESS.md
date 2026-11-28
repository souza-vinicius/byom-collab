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
- [ ] F2.3 Ergonomia do chat: comandos slash existentes, entrada multilinha se aplicável, histórico, confirmação antes de encerrar sessão cara/ativa.

## Fase 3 — Redesenho da Interface de Terminal e Clareza de Informações

- [ ] F3.1 Hierarquia visual consistente: cores, símbolos compatíveis, separação entre prompt, resposta e logs.
- [ ] F3.2 Feedback em tempo real: spinners/status no boot do Colab e carregamento do modelo; streaming limpo de Markdown/código.
- [ ] F3.3 Textos de `--help`, banner, status da sessão e avisos diretos e sem ruído.
- [ ] F3.4 Compatibilidade: terminais estreitos, `NO_COLOR`, modo não-interativo.

## Fase 4 — Revisão e Fortalecimento de Testes Unitários

- [ ] F4.1 Testes de parsing de argumentos/flags e comandos internos do chat.
- [ ] F4.2 Testes com mocks dos estados da sessão Colab: sucesso, timeout, desconexão, erro de cota.
- [ ] F4.3 Testes de formatação/renderização de saída e encerramento (SIGINT/EOF).
- [ ] F4.4 Suíte 100% verde, rápida, determinística, sem sessão Colab real.

## Fase 5 — Documentação e Backlog Estratégico

- [ ] F5.1 README.md atualizado: pré-requisitos, instalação, guia rápido, referência de flags/comandos, troubleshooting.
- [ ] F5.2 Docs `docs/` sincronizadas com o estado atual e validadas pelo validador OKF.
- [ ] F5.3 BACKLOG.md criado com sugestões estruturadas (problema, impacto, esforço, critérios de aceite).

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

## Resumo Executivo

A preencher na Fase 6.
