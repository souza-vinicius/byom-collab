# BACKLOG — ideias fora do escopo da auditoria

Este arquivo registra funcionalidades que **não** foram implementadas na rodada de auditoria/evolução incremental. Nada aqui é promessa de entrega. Cada item descreve o problema, o impacto esperado, o esforço estimado (P/M/G) e critérios de aceite sugeridos.

Formato: `Impacto` é Alto/Médio/Baixo para quem usa a CLI; `Esforço` é P (horas), M (dias) ou G (semana+).

## 1. Exportar e pesquisar histórico do chat

- **Problema:** o histórico fica em `logs.db` dentro de `BYOM_HOME`, mas não há comando para listar, exportar ou buscar conversas antigas.
- **Impacto:** Alto. **Esforço:** M.
- **Critérios de aceite:** novo subcomando `byom history --export <arquivo>` e `--search <termo>`; saída determinística em JSONL/Markdown; nenhuma alteração no formato do `logs.db` existente; testes com banco temporário.

## 2. Troca dinâmica de modelo na sessão ativa

- **Problema:** `byom up` fixa o modelo da sessão; trocar exige `down` + `up` e recarrega a GPU.
- **Impacto:** Alto. **Esforço:** G.
- **Critérios de aceite:** comando para listar modelos carregáveis e solicitar troca ao gateway; orçamento de VRAM revalidado antes da troca; falha não desaloca a sessão; documentação do worker e do gateway atualizada.

## 3. Anexar arquivos/contexto ao chat

- **Problema:** `byom chat` não tem forma oficial de incluir arquivos do projeto como contexto.
- **Impacto:** Alto. **Esforço:** M.
- **Critérios de aceite:** flag de contexto com limite explícito de tokens; nenhum arquivo enviado sem confirmação; segredos e `.gitignore` respeitados; testes com diretório temporário.

## 4. Reconexão automática da sessão do Colab após perda da VM

- **Problema:** o túnel já reconecta, mas uma VM desalocada exige intervenção manual (`byom down` + `byom up`).
- **Impacto:** Alto. **Esforço:** G.
- **Critérios de aceite:** política explícita de retry com limite de tentativas e custo; nunca alocar GPU sem confirmação; estado local e harness preservados; testes com sessão mockada em estados `stopped`/`unknown`.

## 5. Renderizador de streaming com Markdown e blocos de código

- **Problema:** o streaming vem do `llm chat` herdado; o byom não controla a formatação de Markdown, tabelas ou blocos de código.
- **Impacto:** Médio. **Esforço:** M.
- **Critérios de aceite:** saída legível em 80 colunas, com `NO_COLOR` e em pipe; nenhuma regressão no comportamento do `llm`; testes de snapshot de formatação.

## 6. Indicador de progresso animado no `up` da CLI

- **Problema:** a CLI imprime etapas numeradas, mas não tem spinner; a TUI já tem `ExecutorEtapas`.
- **Impacto:** Médio. **Esforço:** P.
- **Critérios de aceite:** spinner apenas em TTY, desativado com `NO_COLOR` e em pipe; texto final idêntico ao modo não-interativo; snapshots existentes preservados.

## 7. Espelhar stderr do chat enquanto o modelo responde

- **Problema:** `run_chat` captura o stderr em pipe e o mostra só no fim.
- **Impacto:** Médio. **Esforço:** P.
- **Critérios de aceite:** avisos do `llm` aparecem em tempo real sem duplicar o `ChatResult.stderr`; pipe sem TTY continua determinístico; testes com subprocesso falso.

## 8. Avisar sessões `byom-*` sem registro no `status`

- **Problema:** `status` mostra apenas sessões `[?]`; o `up` bloqueia também sessões nomeadas sem registro local.
- **Impacto:** Médio. **Esforço:** P.
- **Critérios de aceite:** `status` lista ambos os formatos com o comando exato de encerramento; nenhuma chamada extra à CLI do Colab quando já há `orphan_assignments()`; testes com lista mockada.

## 9. Latência e uso de tokens na barra de status

- **Problema:** `status` mostra latência do túnel, mas não tokens usados nem latência de inferência por requisição.
- **Impacto:** Médio. **Esforço:** M.
- **Critérios de aceite:** métricas já disponíveis no gateway expostas sem novo protocolo; ausência de dados mostrada como `?`, nunca como zero; testes com payload parcial.

## 10. Comandos slash e entrada multilinha no chat

- **Problema:** o chat depende dos comandos do `llm`; a TUI só oferece atalhos de menu (`1-5`, `c`, `h`, `q`).
- **Impacto:** Médio. **Esforço:** G.
- **Critérios de aceite:** `/help`, `/clear` e `/exit` documentados e testados; multilinha preservada; nenhum comando existente removido.

## 11. Diagnóstico automático de configuração

- **Problema:** falhas de GPU, cota, driver e CLI do Colab são descobertas durante o `up`.
- **Impacto:** Médio. **Esforço:** M.
- **Critérios de aceite:** subcomando somente leitura que verifica `ssh`, CLI do Colab, autenticação, portas locais, perfil e VRAM; saída com causa e ação por item; nenhuma sessão alocada.

## 12. Paridade da TUI em Windows

- **Problema:** `byom_home()` cobre `win32`, mas a TUI usa questionary/prompt_toolkit cujo suporte a terminal não foi validado no Windows.
- **Impacto:** Baixo. **Esforço:** M.
- **Critérios de aceite:** execução em PowerShell/Windows Terminal com e sem `NO_COLOR`; quedas de Ctrl+C/EOF tratadas; documentação de requisitos ajustada.
