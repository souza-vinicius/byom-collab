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
- [x] F4.4 Suíte verde e determinística: 607 testes em ~19 s com `BYOM_COLAB_BIN` removido; ruff limpo.

## Fase 5 — Documentação e Backlog Estratégico

- [x] F5.1 README atualizado: Windows como pré-requisito, exemplos de `setup --model-id/--api-key` e `chat -m/--no-history`, fluxo em 5 etapas, tabela de erros com ação, runtime `strata` e 25 perfis.
- [x] F5.2 Docs sincronizadas (`llm-chat.md`, `tui.md`, `troubleshooting.md`, `connection-path.md`, `index.md`, `log.md`); validador OKF v0.2 passa com 8 conceitos.
- [x] F5.3 BACKLOG.md criado com 12 ideias estruturadas (problema, impacto, esforço, critérios de aceite); nenhuma foi implementada nesta rodada.

## Fase 6 — Revisão Final (QA) e Entrega

- [x] F6.1 Diff review. `git diff --check` limpo; dois revisores independentes (código e docs) não encontraram regressões, código morto, imports não usados, debug ou vazamento de token/nonce. Única observação acionável (EOF dentro dos handlers de erro da TUI) corrigida e testada.
- [x] F6.2 Validação final no estado final: ruff limpo, 607 testes passam sem `BYOM_COLAB_BIN`, `scripts/probe_revisao.py` exit 0, validador OKF exit 0, `git diff --check` limpo.
- [x] F6.3 Resumo Executivo preenchido abaixo.

## Registro de ciclos

| Ciclo | Item | Resultado | Verificação |
| --- | --- | --- | --- |
| 0 | F1.1–F1.4 | Baseline medido | 583 testes, ruff/probe/OKF verdes |
| 1 | F1.5 | Auditorias consolidadas; plano priorizado | leitura de 3 relatórios, nenhuma edição |
| 2 | F2.1, F2.2 (CLI), F4 (parcial) | `cli.py` + 10 testes novos | ruff limpo; 593 testes passam |
| 3 | F2.3, F3.1–F3.4 | `llm_chat.py`, `tui.py`, `mensagens.py`, `cli.py`, 6 testes novos | ruff limpo; 599 testes passam |
| 4 | F4.1–F4.4 | 7 testes novos de flags/estados/formatação | ruff limpo; 606 testes passam |
| 5 | F5.1–F5.3 | README, 6 docs OKF, BACKLOG, log | OKF/Ruff/Pytest verdes; 606 testes passam |
| 6 | F6.1–F6.3 | Revisão independente + fix de EOF em handlers | ruff limpo; 607 testes passam; probe e OKF verdes |

## Resumo Executivo

Auditoria concluída em 6 ciclos de execução (limite de segurança 25). Nenhum comando, flag, argumento, fluxo do Colab ou interação de chat existente foi removido: `init`, `profiles`, `gpus`, `models`, `configure`, `up` (`--force`, `--opencode-config`, `--dsh-config`), `status`, `down`, `setup` (`--url`, `--model`, `--model-id`, `--api-key`), `chat` (`--system/-s`, `--no-history/--no-log/-n`, `--continue/-c`, `--model/-m`), `uninstall` (`--yes/-y`), `--tui`/`--no-tui` e o daemon oculto seguem disponíveis. Snapshots da CLI continuam verdes.

**Fase 1.** Baseline medido (583 testes, ruff/probe/OKF limpos) e mapa completo dos fluxos; três auditorias read-only consolidaram os gargalos em `AGENT_PROGRESS.md`.

**Fase 2.** `up` passou a anunciar cinco etapas; `init` explica a falha real da CLI do Colab; timeout do worker traz o último estado; `up` verifica as portas 4000 e 4001 antes de alocar e devolve o harness se falhar após sincronizá-lo; `down` devolve o harness mesmo sem confirmação; GPU passa a ser comparada sem diferenciar maiúsculas e rejeita valor vazio. Ctrl+C no chat virou cancelamento limpo; EOF no menu e dentro de painéis de erro encerra sem traceback.

**Fase 3.** Mensagens de ajuda, banner e erros mais diretos, com orientação de correção; questionary respeita `NO_COLOR` e o console é recriado se a variável mudar após a importação; layout responsivo existente preservado. Nenhum renderizador novo foi criado: streaming e spinner já pertencem ao `llm`/TUI e mudá-los seria nova funcionalidade.

**Fase 4.** 24 testes unitários novos, todos com mocks, cobrindo flags, estados do Colab (ready, daemon morto, silêncio, desconexão, cota), cancelamento do chat, EOF, `NO_COLOR`, portas e formatação de órfãs. Nenhum teste depende de sessão real do Colab.

**Fase 5.** README revisado (pré-requisitos, exemplos, referência de flags, erros com ação, 25 perfis e runtime `strata`); seis documentos OKF sincronizados e validados; `BACKLOG.md` com 12 ideias estruturadas, nenhuma implementada.

**Fase 6.** Diff revisado por duas frentes independentes e verificado localmente; um achado de EOF em handlers foi corrigido. Estado final: 607 testes, ruff limpo, probe de VRAM exit 0, validador OKF exit 0, árvore limpa com quatro commits locais na `main` (sem push ou PR).

Princípios que mudaram decisões: `sequence-verifiable-units` (lint+testes após cada bloco e commit atômico), `laziness-protocol` (etapas numeradas em vez de novo spinner; nenhuma camada extra), `test-behavior-not-implementation` (testes assertam saída observável e `ChatResult`, não detalhes internos), `prove-it-works` (executei ruff, pytest, probe e OKF, além das duas revisões), `experience-first` (erros acionáveis, Ctrl+C/EOF limpos, `NO_COLOR`) e `never-block-on-the-human` (execução contínua conforme autorização).

---

# Rodada 2 — Open WebUI autoconfigurado (2026-11-28)

Objetivo: garantir que `byom web` já abra conectado à API do Byom, com modelo padrão e variáveis de ambiente corretas, preservando overrides do usuário e todas as funcionalidades existentes. Fases 3–6 reaproveitam a estrutura da rodada 1, com foco no frontend.

Condição de parada: todos os itens `[x]`, quatro gates verdes, Resumo Executivo preenchido. Limite: 25 ciclos.

## Baseline (medido antes de qualquer edição nesta rodada)

- Branch `feat/auditoria-cli-byom` (PR #19), árvore limpa.
- `env -u BYOM_COLAB_BIN uv run pytest -q`: 631 passed em ~20 s.
- `uv run ruff check .`: All checks passed.
- `uv run python scripts/probe_revisao.py`: exit 0.
- `validate.sh docs/`: exit 0.
- Fonte oficial consultada em 2026-11-28: `https://docs.openwebui.com/reference/env-configuration/` (`DEFAULT_MODELS` = IDs separados por vírgula pré-selecionados em novo chat; `OPENAI_API_BASE_URL(S)`, `OPENAI_API_KEY(S)`, `OPENAI_API_CONFIGS`).

## Fase 1 — Diagnóstico e Baseline

- [x] R1.1 Mapear `byom web`: `webui.py` já injeta URL/chave, `WEBUI_AUTH=false`, `ENABLE_OLLAMA_API=false`, `DATA_DIR` isolado e segredo estável; `cli.py:1376` imprime endpoints.
- [x] R1.2 Medir baseline e confirmar fonte oficial das variáveis (acima).
- [x] R1.3 Identificar lacunas: `DEFAULT_MODELS` ausente; `build_env` sobrescreve variáveis que o usuário já definiu; CLI não mostra o modelo padrão nem avisa quais variáveis ficaram sob controle do usuário.

## Fase 2 — Fluxos, UX e Autoconfiguração do Open WebUI

- [x] R2.1 Injetar `DEFAULT_MODELS` a partir de `config.served_model_name`, que é o id publicado por `/v1/models` (gateway: id sintético no runtime `tts`, alias nos demais).
- [x] R2.2 Preservar overrides do usuário: variáveis presentes no ambiente não devem ser sobrescritas; byom preenche só o que falta. `DATA_DIR`, segredo e variáveis de isolamento seguem gerenciadas pelo byom quando ausentes.
- [x] R2.3 Tornar a autoconfiguração observável: CLI informa o modelo padrão e lista variáveis preservadas do ambiente, sem alterar a saída existente.
- [x] R2.4 Testes unitários mockados para R2.1–R2.3; Zero regressões nos testes atuais.
- [x] R2.5 `ENABLE_PERSISTENT_CONFIG=false` impede que URL salva no `webui.db` vença as variáveis injetadas; override do usuário preservado.

## Fase 3 — Clareza de informações no terminal

- [x] R3.1 Textos do `byom web` descrevem o que foi autoconfigurado e como fazer override, sem novo comando ou flag.
- [x] R3.2 Ajuda e mensagens de erro existentes permanecem intactas (snapshots da CLI continuam verdes).
- [x] R3.3 `rich_markup_mode="rich"` e exemplos em 9 textos de `help=`; tags não vazam na saída.
- [x] R3.4 `byom status` mostra Colab → GPU → Byom → Chat/WebUI → encerramento, modelo, URL do Byom e URL padrão do WebUI, sem cor e legível em pipe.
- [ ] R3.5 Markdown formatado nas respostas do LLM: não implementado — `llm chat` transmite direto ao TTY; capturar quebraria interatividade. Registrado no BACKLOG #14.

## Fase 4 — Testes unitários

- [x] R4.1 Casos novos em `tests/test_webui.py`: override preservado, `DEFAULT_MODELS`, saída da CLI, consistência com `/v1/models`.
- [x] R4.2 Suíte completa verde com `BYOM_COLAB_BIN` removido; nenhum teste fala com Colab/Open WebUI reais.
- [x] R4.3 Novos casos para `ENABLE_PERSISTENT_CONFIG`, mapa de estágios no `status` e renderização do help.

## Fase 5 — Documentação e backlog

- [x] R5.1 `docs/webui.md` com as variáveis injetadas, `DEFAULT_MODELS`, override e fonte oficial datada; `docs/index.md` e `docs/log.md` sincronizados.
- [x] R5.2 README atualizado onde descreve o frontend; ideias não implementadas registradas em `BACKLOG.md` (#13 `OPENAI_API_CONFIGS`, #14 Markdown no chat).

## Fase 6 — Revisão final

- [ ] R6.1 Revisão independente (código/testes/docs) com verificação própria de cada achado.
- [ ] R6.2 Quatro gates reexecutados no estado final + `git diff --check`.
- [ ] R6.3 Resumo Executivo da rodada 2 preenchido.

## Registro de ciclos (rodada 2)

| Ciclo | Item | Resultado | Verificação |
| --- | --- | --- | --- |
| 0 | R1.1–R1.3 | Baseline e plano | 631 testes, ruff/probe/OKF verdes |
| 1 | R2.1–R2.4, R3.1–R3.2, R4.1–R4.2 | autoconfig com `DEFAULT_MODELS`, overrides preservados, CLI observável e 3 testes novos | 634 testes, ruff/probe/OKF verdes |
| 2 | R5.1–R5.2 | README, docs/webui.md, index, log e BACKLOG #13 | OKF válido; 634 testes |
| 3 | R2.5 | `ENABLE_PERSISTENT_CONFIG=false` + teste/docs | 634 testes, ruff/OKF verdes |
| 4 | R3.3–R3.4, R4.3 | help rico com exemplos e `status` com mapa de estágios | 634 testes; tags Rich não vazam |
