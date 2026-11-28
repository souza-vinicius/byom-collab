# byom-collab

Execute um modelo de código em uma GPU do Google Colab e use-o com agentes de
codificação locais por meio de uma API compatível com OpenAI em
`http://127.0.0.1:4000/v1`. A máquina virtual é desligada automaticamente quando
fica ociosa para evitar desperdício de unidades de computação.

> **Status:** MVP experimental. O túnel, a supervisão e o desligamento
> automático já foram validados no Colab; a inferência com os perfis Qwen
> 32k/256k ainda está em validação.

## Requisitos

* macOS ou Linux com `ssh` e [`uv`](https://docs.astral.sh/uv/).
* A [Colab CLI](https://github.com/googlecolab/google-colab-cli), instalada e
  autenticada (o `byom` não possui fluxo próprio de login):

  ```bash
  uv tool install google-colab-cli
  colab sessions    # conclui o fluxo de login
  ```

* Uma conta no Colab com unidades de computação disponíveis e acesso à GPU
  exigida pelo perfil selecionado (A100 para os perfis Qwen; runtime de memória
  estendida para o perfil 256k).

## Início rápido

```bash
uv sync
uv run byom                                        # abre a interface interativa (TUI) em um terminal real
uv run byom init                                   # verifica dependências, cria a chave SSH e a config
uv run byom gpus                                   # catálogo estático de GPUs e perfis compatíveis
uv run byom models                                 # solicita uma GPU e lista os modelos compatíveis
uv run byom models --gpu L4                        # lista modelos compatíveis com L4, sem prompt
uv run byom configure --profile qwen3.8-27b-32k    # seleciona um perfil de modelo
uv run byom up                                     # aloca a GPU e inicia o endpoint local
uv run byom setup                                  # configura o chat isolado (`llm`) apontando para o endpoint local
uv run byom chat                                   # conversa no terminal com o modelo remoto (histórico isolado em logs.db)
uv run byom web                                    # abre o frontend web (Open WebUI) em 127.0.0.1:8080
uv run byom status                                 # mostra sessão, worker, túnel e uso da conta
uv run byom down                                   # encerra a sessão
uv run byom uninstall                              # remove apenas o diretório exclusivo do byom
```

* `byom` sem subcomandos em um terminal interativo (TTY) abre o menu em PT-BR
  (use `--no-tui` para forçar a ajuda da CLI ou `--tui` para exigir o menu). Na
  TUI, a opção **Configurar harness (OpenCode / dsh)** permite escolher o
  cliente, consultar as instruções passo a passo para configuração manual e,
  opcionalmente, aplicar a conexão automaticamente (sem alterar arquivos de
  harness durante o `Iniciar sessão no Colab`). A opção **Conversar com o
  modelo (llm chat)** (atalho `c`) abre o chat no terminal com a sessão ativa.
* Na TUI, a opção **Selecionar GPU** (atalho `2`) tem um caminho único:
  **GPU -> modelo**. A tela mostra a GPU, a VRAM e quantos modelos têm receita
  de execução nela; ao escolher a GPU, a tabela lista os modelos que cabem com a
  **maior janela de contexto que a VRAM permite**, até o máximo documentado do
  modelo. GPU menor recebe um contexto menor; GPU maior recebe o contexto
  máximo. Um painel de confirmação mostra a janela otimizada, a memória estimada
  e os valores de `gpu`/`gpu_high_mem`. **Nada é gravado em `config.toml` antes da
  confirmação**: voltar, trocar de GPU ou sair pelo menu não deixa resíduo, e um
  par GPU + modelo que não cabe é recusado sem escrever nada. Os perfis TOML
  continuam existindo como receitas internas de runtime, mas não são escolhidos
  na TUI.
* `byom profiles` lista as receitas internas de runtime; `byom gpus` exibe o
  catálogo estático de GPUs. Encerre a sessão antes de trocar de modelo.
* `byom models --gpu L4` lista as variantes que cabem nessa GPU com o contexto
  otimizado, sem solicitar entrada. Sem `--gpu`, `byom models`
  imprime a lista numerada de GPUs, solicita um número ou ID (sem diferenciar
  maiúsculas/minúsculas) e exibe somente as variantes compatíveis. Uma entrada
  inválida interrompe com os IDs aceitos; nenhuma GPU é escolhida
  automaticamente.
* A lista de GPUs é estática e seus valores de VRAM são aproximados, obtidos da
  fonte do Google Cloud registrada em `src/byom/catalog.py` — não da sua conta
  do Colab. Selecionar uma GPU em `byom models` apenas filtra a listagem: não
  consulta a conta, Colab ou Hugging Face, não aloca VM, não baixa modelo e não
  grava `config.toml`, `state.json` nem qualquer outro arquivo de configuração.
  Use `byom configure --profile ...` quando quiser persistir uma escolha.
* `byom gpus` **não** consulta a conta. Escolha a GPU em `config.toml`;
  `byom up` recusa iniciar se a GPU declarada não comportar a variante
  selecionada (ele nunca aloca uma GPU diferente).
* `byom up --force` encerra a sessão anterior antes de criar uma nova (o modelo
  é carregado do zero).

### Chat no terminal (`byom setup`, `byom chat` e `byom uninstall`)

O `byom` integra o `llm chat` em um diretório exclusivo, sem alterar ou ler a
configuração global do `llm` no seu sistema:

* **Diretório exclusivo (`BYOM_HOME`)**: por padrão, o `byom` resolve a pasta de
  acordo com o sistema operacional (`$XDG_CONFIG_HOME/byom` ou `~/.config/byom`
  no Linux, `~/Library/Application Support/byom` no macOS e `%APPDATA%\byom` no
  Windows). Você pode sobrescrever esse caminho definindo `BYOM_HOME`.
* **Configurar e atualizar a URL do túnel (`byom setup`)**: comando idempotente
  que cria o diretório do `byom`, o arquivo-marcador `.byom-managed`, as
  preferências (`llm-chat.json`) e o arquivo `extra-openai-models.yaml` (com
  permissão `0600`). Quando a URL mudar em uma nova sessão ou túnel do Colab,
  execute novamente apenas alterando `--url` (as demais preferências são
  preservadas):

  ```bash
  uv run byom setup --url http://127.0.0.1:4000/v1 --model qwen3.8-27b
  # Em uma nova sessão/túnel com URL diferente:
  uv run byom setup --url https://novo-tunel.trycloudflare.com/v1
  ```

* **Conversar (`byom chat`)**: executa `python -m llm chat` como subprocesso
  passando `LLM_USER_PATH` apontando para o diretório do `byom` somente no
  ambiente do subprocesso. O histórico fica salvo em `logs.db` dentro desse
  diretório. Requer `byom setup` prévio.
  * `--system` / `-s`: define o system prompt inicial da conversa.
  * `--no-history` / `--no-log` / `-n`: desliga a gravação em `logs.db` durante a
    sessão (controlado via arquivo temporário `logs-off` no diretório do `byom`).
  * `--continue` / `-c`: continua a conversa anterior salva em `logs.db`.
* **Remover dados locais (`byom uninstall`)**: lista o conteúdo do diretório do
  `byom`, pede confirmação (use `--yes` para pular), valida o arquivo-marcador
  `.byom-managed` (recusando remover `HOME`, `/`, caminhos vazios ou symlinks),
  remove apenas o diretório do `byom` sem tocar no `llm` global e imprime o
  comando para desinstalar o pacote (`pip uninstall byom`).

### Frontend web (`byom web`)

O `byom web` sobe uma instância isolada do [Open WebUI](https://openwebui.com)
em `http://127.0.0.1:8080`, conectada ao proxy local em
`http://127.0.0.1:4000/v1`. A interface cobre conversas com histórico, system
prompt por conversa, upload de arquivos e parâmetros do modelo, sem alterar
nenhuma configuração global do Open WebUI.

* **Isolamento**: dados em `$BYOM_HOME/webui` (SQLite, uploads e cache) e segredo
  de sessão em `$BYOM_HOME/webui-secret` com permissão `0600`. O comando cria o
  marcador `.byom-managed`, então `byom uninstall` remove tudo.
* **Sem login**: `WEBUI_AUTH=false`, porque a interface escuta apenas em
  loopback. Para multiusuário, use outro diretório de dados.
* **Primeira execução**: baixa o Open WebUI 0.13.5 e o Python 3.12 pelo `uv`
  (o pacote exige `<3.13`). As execuções seguintes reutilizam o cache.
* **Opções**: `--port` / `-p` muda a porta (padrão 8080); `--no-browser` não abre
  o navegador. `Ctrl+C` encerra o frontend.
* **Limite de runtime**: o byom serve um perfil por sessão. A lista de modelos
  do Open WebUI reflete o runtime ativo. Modelos de imagem recusam chat; a
  geração de imagem do Open WebUI exige um modelo de chat e uma conexão de
  imagem ao mesmo tempo, combinação que uma sessão única do byom não oferece.

Detalhes em [Open WebUI Frontend](docs/webui.md).

## Catálogos de GPUs e modelos

`src/byom/catalog.py` é a fonte única de números de GPU/VRAM e
modelo/quantização; os comandos apenas os renderizam. `byom up` verifica
`pesos + projetor de visão (0,65 GiB quando o perfil carrega mmproj) + 2 GiB de
runtime + cache KV + drafter DSpark (0,47 GiB, só com `dspark = true`) + folga`
contra a GPU declarada em
`config.toml`. O KV é `2 GiB por 32K` (heurístico, em f16) para quem não declara
`ModelEntry.kv_shape`, e pela arquitetura — `2 × kv_heads × head_dim × bytes ×
(camadas completas × ctx + locais × min(ctx, janela))` — para quem declara.
A folga é `max(1 GiB, 5% da VRAM)` — driver, contexto CUDA,
alocadores e fragmentação —, então 100% da VRAM nunca é aprovado (é heurística
de engenharia, não número de documentação). Modelos de imagem não reservam cache
KV autorregressivo.

GPUs (a VRAM é **aproximada para o Colab**: a FAQ oficial do Colab afirma que os
tipos de GPU variam ao longo do tempo e não publica uma tabela de VRAM; os
números abaixo vêm de `docs.cloud.google.com/compute/docs/gpus`, "General
comparison chart", consultado em 2026-11-23):

| GPU (ID no catálogo) | VRAM | bf16 | Valor na Colab CLI |
| :--- | ---: | :--- | :--- |
| T4 | 15 GiB (utilizáveis no Colab; o datasheet diz 16) | não | `T4` |
| L4 | 24 GiB | sim | `L4` |
| A100 40 GB | 40 GiB | sim | `A100`, `gpu_high_mem=false` |
| A100 80 GB | 80 GiB | sim | `A100`, `gpu_high_mem=true` |
| H100 80 GB | 80 GiB | sim | `H100` |

Contexto doc é o máximo na página oficial; o contexto servido exibido por
`byom models` é a maior janela que cabe na GPU, até esse máximo (a TUI mostra o
mesmo número ao escolher GPU + modelo). `byom models --gpu <ID>` usa o mesmo
orçamento de `byom up` para filtrar a listagem; ele não altera a GPU declarada:

| Modelo | Fornecedor | Runtime | Variante | Memória | Contexto doc | Contexto servido | GPU mínima |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| Inkling 2 Mini | Thinking Machines Lab | llama.cpp | 2 bit (7,90 GiB) / 4 bit / 8 bit — pesos Unsloth | 7,9 / 18 / 24 GiB | 1.000.000 | 32.768 / 65.536 / 131.072 | **T4** / **L4** / A100 40 |
| Inkling 2 Mini | Thinking Machines Lab | llama.cpp | AD-Q6_K / AD-Q4_K / AD-IQ2_XXS — pesos AtomicChat | 22,2 / 15,4 / 9,7 GiB | 1.000.000 | 131.072 / 32.768 / 32.768 | A100 40 / **L4** / **T4** |
| Qwen3.8 27B | Qwen (Alibaba Cloud) | vLLM (AWQ) | W4A16-AWQ | 17 GiB | 262.144 | 32.768 ou 262.144 | A100 40 / A100 80 (256K) |
| Qwen3.8 27B | Qwen (Alibaba Cloud) | llama.cpp | 3 bit / 4 bit | 12 / 17 GiB | 262.144 | 32.768 | L4 |
| Ternary Bonsai 2.5 27B | Prism ML | llama.cpp | Q2_0 (upstream) / PTQ1_0 / PTQ1_0-MTP (fork PrismML nos dois últimos) | 7,15 / 5,58 / 5,80 GiB | 262.144 | 32.768 / 65.536 / 65.536 | **T4** nos três |
| Gemma 4 | Google DeepMind | llama.cpp | E4B / 12B / 26B-A4B / 31B, 4 bit | 6 / 8 / 18 / 20 GiB | 131.072 (E2B/E4B), 262.144 (demais) | 65.536 / 32.768 / 32.768 / 131.072 | T4 / T4 / **A100 40** / A100 40 |
| Muse Glimmer 30B | Meta Superintelligence Lab | llama.cpp | 2 bit / 4 bit / 6 bit | 12 / 17 / 22 GiB | 131.072 (até 262.144) | 32.768 / 65.536 / 131.072 | L4 / A100 40 / A100 40 |
| Qwen-Image-2.1 | Qwen (Alibaba Cloud) | image | INT8-ConvRot (12,07 GiB) / GGUF Q4_K (9,22 GiB) | 12,1 / 9,2 GiB | n/a | n/a | L4 / **T4** |
| Qwen3-TTS | Qwen (Alibaba Cloud) | tts | 0.6B CustomVoice (2,32 GiB) / 1.7B CustomVoice (4,21 GiB) | 2,3 / 4,2 GiB + pipeline | n/a | n/a (fala) | **T4** / **L4** |

**Fornecedor** é quem desenvolve o modelo; é dado do catálogo (`ModelEntry.provider`),
com a fonte registrada em `catalog.PROVIDER_SOURCES` (URL + data) e conferida em
2026-11-24. **Publicado por** (ou **Pacote**) é a org do repositório de onde o byom
baixa os pesos — derivada em tempo de render do `model_repo`, do papel `diffusion`
ou do `model` do vLLM, nunca fixada por perfil. É por isso que o mesmo Inkling 2
Mini aparece com pesos Unsloth e com pesos AtomicChat: muda o publicador e a
receita de quantização, não o modelo. `byom profiles` imprime as duas linhas
(`Fornecedor:` e, quando divergem, `Pacote:`), `byom models --gpu <ID>` traz as
duas colunas, e a TUI mostra o fornecedor na própria linha do perfil. Detalhes em
[Model Provider and Weight Package](docs/provider.md).

Os perfis AD da AtomicChat usam o **mesmo** binário `b12611` dos perfis
Unsloth: esses arquivos já estão no layout merged do PR #32694. Ficaram de fora
`Inkling-2-Mini-Q4_K_M.gguf` (17,38 GB) e `Q5_K_M` (19,81 GB), que continuam no
layout antigo do PR #32695 e **não carregam** em b12600+. O drafter DSpark
(`dspark-Inkling-2-Mini-Q8_0.gguf`, 503.857.696 B no repositório da AtomicChat)
ganhou perfil em `inkling-2-mini-ad-q6k-dspark`, pinado no build **b12800**
(`v1.1.0-dev`), o primeiro release oficial com `--spec-type draft-dspark` — o
PR #32218 foi mergeado em 2026-11-24 e entrou na **v1.1.0**, então o fork do PR
e o workflow de build do byom não são mais necessários. O `AD-IQ2_XXS` serve
**32K** na T4 porque o KV do Inkling é contado pela
arquitetura (0,58 GiB em 32K, e não os 2 GiB do heurístico) — o motivo está em
`Variant.context_reason`.

Memória e GPU são conta, não estimativa de página: cada número vem dos bytes
publicados no Hugging Face (conferidos em 2026-11-24) e o perfil declara a GPU
onde `required + folga` cabe de verdade. A tabela com `required`, folga e margem
de cada perfil está em [docs/log.md](docs/log.md#2026-11-24).

A redução é dado estruturado, não texto solto: `src/byom/catalog.py` mantém
`Variant.context_doc`, `Variant.context` e `PROFILE_CONTEXTS`, uma entrada por
perfil com a GPU e o motivo. O cache KV cresce com o contexto **e com a
arquitetura**: sem `ModelEntry.kv_shape` vale o heurístico de 2 GiB por 32K em
f16 (1 GiB em fp8, o dtype que o argv pede — `--kv-cache-dtype` nos perfis
vLLM, f16 fixo nos llama-cpp); com `kv_shape`, a conta é
`2 × kv_heads × head_dim × bytes × (camadas completas × ctx + camadas locais ×
min(ctx, janela))`, arredondada para cima. O Inkling 2 Mini declara a forma do
`config.json` oficial (`4` camadas completas + `20` com janela de 1024, 16 KiB
por token e 80 MiB fixos): 16K = 0,33 GiB, 32K = 0,58, 128K = 2,08 — contra
1/2/8 do heurístico. `byom models` mostra ambos os números; `byom profiles` e
`byom configure` exibem o par documentado/servido com a justificativa, e a TUI
mostra a procedência do KV (arquitetura com URL e data, ou "heurístico"). O
perfil que liga `[worker] dspark = true` também soma os 0,47 GiB do drafter
DSpark. As checagens de compatibilidade continuam usando o contexto SERVIDO — o
máximo documentado nunca é usado para alocar uma GPU incapaz de comportar o
cache KV.

`reasoning_effort` também é uma propriedade do modelo no catálogo, com a fonte
registrada em `ModelEntry.reasoning_source` (todas conferidas em 2026-11-23):

| Modelo | Padrão documentado | Fonte |
| :--- | :--- | :--- |
| Inkling 2 Mini | `high` | huggingface.co/blog/inkling2mini e `thinkingmachines/Inkling-2-Mini/chat_template.jinja` — padrão do template 0.9; escala none/minimal/low/medium/high/xhigh/max. O briefing também cita `xhigh` na avaliação da Thinking Machines, mas essa linha exata **não** foi localizada nas páginas consultadas. |
| Qwen3.8 27B | `xhigh` | unsloth.ai/docs/models/qwen3.8.md — "xhigh (default)" e "xhigh reasoning" nas configurações de benchmark |
| Muse Glimmer 30B | `high` | unsloth.ai/docs/models/muse-glimmer.md — "Thinking Settings" lista low/medium/high/xhigh e a tabela de benchmark diz "High Reasoning" |
| Gemma 4 | nenhum | unsloth.ai/docs/models/gemma-4.md — a página **não** define `reasoning_effort`; define `<|think|>` e `enable_thinking`. Nenhum valor foi inventado. |
| Qwen-Image-2.1 | nenhum | modelo de difusão; sem campo de raciocínio de chat |

A lista de valores aceitos em `config.toml`, `session.json` e no gateway é
exatamente a união documentada por essas páginas: `none`, `minimal`, `low`,
`medium`, `high`, `xhigh`, `max` (além de `null`/ausência para Gemma e imagem).
Números como `0.9` são aceitos pelo template do Inkling, mas são rejeitados
propositalmente na nossa fronteira: nenhuma string livre ou número é aceita.

Notas dos perfis: Inkling 2 Mini usa o build mainline `b12611` do llama.cpp com
os GGUFs de `Mainline_Support` (a extensão `.gguf_file` do repositório é
renomeada para `.gguf` no download). Builds mainline anteriores perdem os scales
NVFP4 por especialista na segunda e terceira passadas em loop e produzem saída
incorreta. Os perfis `inkling-2-mini-ad-*` usam o **mesmo** `b12611` porque os
arquivos `AD-*` da AtomicChat já estão no layout merged do PR #32694 e o
publicador exige `b12600` ou mais novo com `--jinja`; nenhum perfil usa
`Q4_K_M`/`Q5_K_M`. O Ternary Bonsai 2.5 27B tem três variantes e só a primeira
roda no llama.cpp upstream: `Q2_0` precisa de build **b11556** ou mais novo
(o perfil usa o mesmo `b12611` dos perfis Inkling) e builds mais velhos carregam
o arquivo e geram lixo sem avisar; `PTQ1_0` e o `PTQ1_0-MTP` exigem o binário do
fork **PrismML**, fixado em `prism-b12128-d7b2896` (asset CUDA 12.8 x64 de
228.535.760 B, sha256 conferido). O irmão `prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit`
é **MLX** (Apple Silicon) e ficou fora do catálogo: o byom roda no Colab (CUDA) e
não tem runtime MLX, então nenhum perfil seria executável. DSpark está ligado em
**um** perfil, `inkling-2-mini-ad-q6k-dspark`: o suporte entrou no upstream
(PR #32218, mergeado em 2026-11-24, presente na **v1.1.0**) e o perfil fixa o
build oficial **b12800** (`llama-b12800-bin-ubuntu-cuda-12.8-x64.tar.gz`,
221.624.554 B, sha256 `088bc70b…587892d` conferido contra o asset baixado). A
conta é 22,15 + 0,65 visão + 2 runtime + 2,08 de KV em 128K + 0,47 do drafter =
27,35, mais 2,0 de folga = 29,35 <= 40 GiB da A100-40; o custo do drafter entra
no orçamento em qualquer perfil que ligue `[worker] dspark = true`, e
`available_profiles` recusa o campo sem `drafter_pattern`. Muse Glimmer usa o
release fixado `b12649` do upstream.
Qwen3.8 mantém o perfil AWQ existente para vLLM e adiciona opções GGUF. Os
perfis do Gemma 4 carregam uma chave explícita `thinking_policy` vinda do
catálogo: o gateway adiciona `<|think|>` no início do system prompt (criando a
mensagem de sistema quando ausente) e remove blocos `<|channel>thought … <channel|>`
de turnos anteriores do assistente antes de encaminhar. Um cliente que já
incluiu o token, limpou o histórico ou enviou `enable_thinking: false` mantém
sua intenção. Todos os binários são baixados de URLs fixadas com SHA-256 dos
assets de release; repositório do modelo, quantização e padrão de revisão são
fixos no perfil — sem `@latest`. Os assets CUDA dos perfis `llama-cpp` são os
builds `cuda-12.8` do upstream (`b12611`/`b12649`) e do fork PrismML (12.8):
eles trazem SASS para sm_86/sm_89/sm_120 e PTX para sm_75/sm_80/sm_90, então
L4 (SASS) e T4/A100/H100 (JIT no primeiro load) funcionam. Os assets `cuda-13.x`
não servem: só têm SASS para sm_86/89/120/121 e PTX novo demais para o driver
580 do Colab (`CUDA error: the provided PTX was compiled with an unsupported
toolchain` na T4). O runtime recebe `libcudart.so.12`/`libcublas.so.12` dos
pacotes `nvidia-*-cu12` que a imagem mantém no Python do kernel, porque o
`ldconfig` da VM só publica CUDA 13. O bootstrap imprime o fim do log do runtime
quando ele morre no início, em vez de deixar só o `server exited unexpectedly`
do tmux.

Fora de escopo (sem perfil executável): MiniMax-M3.1-Flash (1 bit 129,4 GB,
2 bit 143,9 GB, 4 bit 249,4 GB / orçamento de 256 GB+), GLM-5.3-Flash (1 bit
93,09 GB, 3 bit 120,37–147,54 GB, 4 bit 199,71 GB) e DeepSeek-V4.1-Flash (1 bit
128–145 GB, download 4 bit 500,7 GB, tabela 305 GB + Engram em SSD). Nenhum
deles cabe em uma única GPU do Colab.

### Runtimes e endpoints

* `vllm` e `llama-cpp` atendem `POST /v1/chat/completions` e `GET /v1/models`
  pelo mesmo proxy e gateway. O `llama-server` é iniciado com o GGUF fixado, o
  projetor de visão `mmproj` quando presente e o rascunho DSpark quando
  solicitado pelo perfil.
* `image` atende `POST /v1/images/generations` e `POST /v1/images/edits` pela
  API de imagens compatível com OpenAI do `sd-server`, além de `GET /v1/models`
  para metadados do modelo. Trata-se de um modelo de difusão, e **não** de um
  modelo de chat: `POST /v1/chat/completions` retorna HTTP 501 neste runtime. As
  requisições recebem `prompt` e opções de imagem, não `messages`; as respostas
  contêm dados de imagem (`data[].b64_json`), não tokens ou motivos de parada.
  A página da Unsloth documenta padrões de **25 steps e guidance 1** (conferido
  em 2026-11-23); ambos vivem apenas em `ModelEntry.generation_defaults`. O
  sd-server **não** aceita um campo `guidance` no nível raiz: o gateway mapeia o
  par para a extensão nativa documentada dentro de `prompt` —
  `<sd_cpp_extra_args>{"sample_params":{"sample_steps":25,"guidance":{"txt_cfg":1.0}}}</sd_cpp_extra_args>`
  (`examples/server/api.md`, conferido em 2026-11-23). O cliente pode
  sobrescrever `steps`/`guidance` no corpo da requisição ou enviar seu próprio
  bloco `sd_cpp_extra_args`. Divergência registrada: a mesma página da Unsloth
  sugere `--steps 20 --cfg-scale 6.0` para o fluxo via sd-cli; este projeto
  adota os padrões documentados da API (25/1) conforme o briefing, não um valor
  inventado para o sd.cpp. `POST /v1/images/edits` usa multipart e mantém os
  padrões do servidor. A página documenta cerca de 120 s por imagem 1024x1024
  em 25 passos numa Colab T4 de 15 GB; o perfil GGUF Q4_K usa **T4** (9,22 GiB
  de pesos + 1,08 do projetor de visão + 2 de runtime + 1 de folga = 13,30, que
  cabe nos 15 GiB utilizáveis) e o perfil INT8-ConvRot fica em L4 (12,07 + 1,08
  + 2 + 1 = 16,15 > 15). A T4 não possui bf16, portanto o treino é recusado
  pela Unsloth; treino em bf16 exige arquitetura Ampere ou superior. O nome do
  asset Linux fixado não declara seu backend; verifique o uso de GPU na primeira
  execução real. Se for um build de CPU, compile a partir da tag fixada com
  `-DSD_CUDA=ON` e coloque o binário em `/content/byom/sdcpp/sd-server`: o
  `runtime_manifest.json` aponta para o symlink `/content/byom/sd-server`, que
  o `prepare_runtime` cria a partir do diretório `binary_dir` do perfil. Não há
  variável de ambiente para o caminho do binário — o manifest é a única fonte.
* Repositórios no Hugging Face e a CLI do Colab são **verificados apenas na
  primeira execução real**: este repositório não os aciona durante os testes
  locais. Confira os arquivos de modelo resolvidos pelos padrões fixados e a GPU
  efetivamente alocada por `colab new` antes de depender de um perfil.

Fontes consultadas em 2026-11-23: páginas de modelos da Unsloth
(`inkling-2-mini.md`, `qwen3.8.md`, `gemma-4.md`, `muse-glimmer.md`,
`qwen-image-2.1.md`, além das três páginas "Flash"), `blog/inkling2mini` e
`thinkingmachines/Inkling-2-Mini/chat_template.jinja` no Hugging Face,
`examples/server/api.md` de `leejet/stable-diffusion.cpp`, tipos de máquina com
GPU do Google Cloud e FAQ do Colab, `unsloth/Inkling-2-Mini-GGUF` no Hugging
Face (`Mainline_Support/README.md`, `HOW_TO_RUN.md` e discussão #8), listas de
assets de release de `ggml-org/llama.cpp` `b12611` e `b12649`, PRs #32694 e
#32704 do llama.cpp e `leejet/stable-diffusion.cpp` `master-1106-532d2c1`.

## Uso com agentes de codificação

| Agente | Configuração |
| :--- | :--- |
| OpenCode | Automática: `byom up` adiciona o provedor e define o modelo padrão |
| dsh (DeepSeek harness) | Automática: `byom up` adiciona o provedor `byom` e define o modelo padrão |
| Aider | `OPENAI_API_BASE=http://127.0.0.1:4000/v1` e `OPENAI_API_KEY=dummy` |
| Cline | Provedor compatível com OpenAI usando URL base `http://127.0.0.1:4000/v1` |

**OpenCode:** `byom up` mescla o provedor, o plugin de descoberta e os aliases
32k/256k na configuração global (`$XDG_CONFIG_HOME/opencode` ou
`~/.config/opencode`), preservando o restante. Use `--opencode-config <arquivo>`
para indicar outro caminho e reinicie o OpenCode após as alterações. Arquivos
JSONC (com comentários ou vírgulas finais) não são reescritos, fazendo o `up`
abortar antes de alocar uma GPU.

**dsh:** se `~/.dsh/profiles/desktop/cordis.patch.yml` existir (ou
`--dsh-config <arquivo>` for informado), `byom up` adiciona `providers.byom` à
entrada `llm-pi-ai` (compatível com OpenAI, `baseURL` apontando para o proxy
local, o modelo com o `contextWindow` do perfil ativo e `apiKeyEnv: BYOM_API_KEY`
caso ainda não haja uma chave definida) e define `agent-default-model` como
`byom`. Nada mais é alterado ou removido: outros provedores, modelos, variáveis
de chave de API, entradas e comentários permanecem intactos. Exporte qualquer
valor, por exemplo `export BYOM_API_KEY=dummy` (o proxy não valida a chave),
antes de iniciar o dsh. O arquivo é editado como texto; arquivos inválidos (não
lista, indentados com tab ou com bloco `byom` fora do padrão) abortam o `up`
antes de alocar uma GPU (mantendo o arquivo intocado). Reinicie o dsh após as
alterações.

**Não suportados:** Cursor (as requisições partem do backend do Cursor e não
alcançam seu `localhost`) e Claude Code (usa a API Messages da Anthropic em
`/v1/messages`).

**Raciocínio (Reasoning):** cada modelo traz seu padrão documentado no catálogo;
o perfil o herda e o cliente pode sobrescrevê-lo por requisição. Se o cliente
omitir `chat_template_kwargs.reasoning_effort`, o gateway injeta esse valor —
`high` para Inkling 2 Mini, `xhigh` para Qwen3.8, `high` para Muse Glimmer. Para
Gemma 4 o valor no catálogo é `null` (a página não define o campo) e **nada é
injetado**. Os valores válidos são exatamente `none`, `minimal`, `low`,
`medium`, `high`, `xhigh`, `max`, validados em `config.toml`, `session.json` e na
fronteira do gateway; um valor inválido aborta a inicialização. Um cliente que
envie um `chat_template_kwargs` que não seja objeto mantém o payload original.
As fontes e a data de conferência (2026-11-23) estão na tabela acima e em
`ModelEntry.reasoning_source`.

## Como funciona

```text
Agente --HTTP/SSE--> Proxy :4000 --> Túnel :4001 ==SSH (via colab ssh)==> VM Colab
                                                                           Gateway :8001 --> runtime :8000
                                                                             (vLLM | llama-server | sd-server)
                                                                              ^
                                                       Watchdog (célula) -----+--> runtime.unassign()
```

* **Gateway:** todas as requisições de inferência passam por ele, tornando-o a
  fonte de verdade de atividade (`/health` e `/v1/models` não contam como
  trabalho ativo). Ele encaminha para o runtime em `vllm_port`, a porta que vem
  na `session.json`; não existe URL fixa para a porta do runtime.
* **Watchdog:** célula de longa duração no kernel que verifica o gateway a cada
  10 segundos e chama `unassign()` após o tempo limite de ociosidade. O daemon
  local a reinicia se a conexão cair.
* **Desligamento seguro:** a VM só é encerrada após o gateway confirmar que não
  há trabalho em andamento. Se o watchdog ficar sem enviar heartbeats por mais
  de 60 segundos, o gateway interrompe o vLLM.
* **SSH:** uma conexão mestre do OpenSSH (`ControlMaster`, usando
  `colab ssh` como `ProxyCommand`) por vez. O `up` usa a dele e a encerra ao
  terminar; o daemon abre a do túnel num socket de controle próprio
  (`cm-<sessão>-tunnel.sock`), então o `ssh -O exit` do `up` não derruba a
  conexão que o túnel está usando, e as duas nunca disputam o slot único do
  `colab ssh`.

## Configuração

Arquivo de configuração: `~/.config/byom/config.toml` (criado por `init`; os
perfis definem os campos ligados ao modelo):

| Chave | Padrão | Descrição |
| :--- | :--- | :--- |
| `profile` | vazio | Perfil ativo |
| `runtime` | `vllm` | Runtime selecionado pelo perfil: `vllm`, `llama-cpp` ou `image` |
| `model` | `philbert440/Qwen3.8-27B-W4A16-AWQ` | Checkpoint servido pelo vLLM |
| `served_model_name` | `qwen3.8-27b` | ID do modelo exposto pela API |
| `max_model_len` | `16384` | Tamanho máximo de contexto |
| `reasoning_effort` | herdado | Padrão documentado do modelo no catálogo (Qwen3.8 `xhigh`, Inkling `high`, Muse `high`; Gemma/imagem omitido). Um perfil ou o cliente pode sobrescrevê-lo. |
| `gpu` | `L4` | Aceleradora solicitada |
| `gpu_high_mem` | `false` | Solicita runtime de memória estendida (A100; exige plano elegível) |
| `idle_timeout_min` | `20` | Tempo de ociosidade antes do desligamento |
| `max_job_min` | `40` | Tempo limite para job travado |
| `bootstrap_timeout_min` | `30` | Tempo máximo permitido para o worker ficar pronto |
| `watchdog_timeout_s` | `60` | Timeout do watchdog que interrompe o vLLM |
| `proxy_port` / `tunnel_port` | `4000` / `4001` | Portas locais |
| `gateway_port` / `vllm_port` | `8001` / `8000` | Portas na VM |
| `gpu_memory_utilization` | `0.92` | Fração de memória da GPU usada pelo vLLM (deve estar em `(0, 1]`) |
| `kv_cache_dtype` | `fp8` | Tipo de dado do cache KV |
| `vllm_spec` | `vllm` | Pacote vLLM a instalar (versão não fixada) |
| `vllm_enforce_eager` | `false` | Desativa CUDA graphs se houver falta de memória |
| `vllm_language_model_only` | `true` | Não reserva memória para a torre de visão |
| `ssh_key` | `~/.ssh/byom_ed25519` | Chave do túnel |

O estado da sessão é salvo em `~/.config/byom/state.json` (gravado com permissão
0600, serializado por arquivo de trava e contendo o nonce de identidade do
worker: não é uma credencial do usuário, mas não deve ser compartilhado). As
sessões do `byom` usam um arquivo isolado
(`~/.config/byom/colab-sessions.json`), separado das suas outras sessões do
Colab. `session.json` também carrega `thinking_policy` (chave explícita do
Gemma 4 vinda do catálogo) e `generation_defaults` (`steps`/`guidance` para o
Qwen-Image), ambos validados pelo gateway na inicialização.

## Problemas conhecidos

**Uma conexão SSH por sessão.** O runtime aceita apenas uma conexão `colab ssh`
por vez. Executar outro `colab ssh` desconecta o `byom` (HTTP 429). Use
`byom status` para inspecionar o worker.

**Sessões órfãs.** Se a CLI do Colab alocar uma VM mas falhar ao gravar o estado
local, a sessão aparece como `[?]` em `colab sessions`, continua consumindo
unidades de computação e não pode ser encerrada com `colab stop`. `byom status`
e `byom up` emitem um aviso quando isso ocorre. Encerre a sessão pelo site do
Colab ou execute:

```bash
~/.local/share/uv/tools/google-colab-cli/bin/python -c \
  "from colab_cli.common import state; state.client.unassign('<endpoint>')"
```

onde `<endpoint>` é o segundo campo na saída de `colab sessions`.

## Decisões de design e comportamento externo não verificado

Registrado em 2026-11-23 sobre o clone local no commit `cba79b8`.

* **Valores documentados são dados do catálogo.** `reasoning_effort`,
  `thinking_policy` do Gemma 4, `generation_defaults` do Qwen-Image e ambos os
  números de contexto ficam em `src/byom/catalog.py`; os perfis os herdam e o
  gateway os recebe em `session.json`. Fontes e datas de cada valor estão
  registradas no catálogo e nas tabelas acima. Nenhum perfil repete `medium`,
  `25` ou um máximo de contexto documentado.
* **O contexto é otimizado por GPU, não reduzido em silêncio.**
  `PROFILE_CONTEXTS` lista os 27 perfis com contexto doc, contexto de referência,
  GPU e motivo. `byom models` e a TUI calculam a maior janela que cabe na GPU
  até o máximo documentado (`catalog.max_fitting_context`), e a seleção grava
  esse valor em `max_model_len`; `check_fit` valida o par antes de qualquer
  alocação.
* **Thinking é guiado por comportamento, não pelo nome do modelo.** Apenas um
  modelo com `ModelEntry.thinking` recebe a transformação de prompt do Gemma 4.
  O token do próprio cliente, histórico já limpo ou `enable_thinking: false`
  prevalecem.
* **Padrões de imagem usam os nomes do servidor.** A Unsloth fala em
  "steps/guidance"; o sd-server documenta `sample_params.sample_steps` e
  `sample_params.guidance.txt_cfg` via `sd_cpp_extra_args`. A equivalência está
  registrada em `ModelEntry.generation_source`; nenhum `guidance` no nível raiz
  é enviado.
* **`state.json` corrompido bloqueia `up`.** Arquivo ausente significa "sem
  sessão"; arquivo ilegível ou inválido é erro. `up` também verifica
  `orphan_assignments()` tanto no caminho normal quanto com `--force` e
  **bloqueia** (sem alocar) enquanto houver sessões órfãs. Motivo: uma órfã
  continua consumindo unidades de computação e não pode ser encerrada por
  `colab stop`; pedir confirmação deixaria a decisão para quem pode não estar
  vendo o estado do backend. Corrija no Colab e tente novamente.
* **Desalocação de fallback no gateway.** A célula de watchdog continua sendo a
  chamadora principal de `runtime.unassign()`. Se ela (ou seu supervisor local)
  falhar, o gateway agenda um fallback pelo mesmo módulo `google.colab.runtime`,
  30 segundos após um desligamento por ociosidade aceito ou após a parada do
  vLLM. Ele ignora a desalocação enquanto `active_jobs > 0` e nunca chama
  endpoints externos.
* **Limite de jobs.** O gateway aceita no máximo `BYOM_MAX_CONCURRENT_JOBS`
  jobs concorrentes (padrão `4`; valores inválidos voltam para `4`); requisições
  excedentes recebem HTTP 503. Defina a variável de ambiente no ambiente da VM
  para alterá-la; ela não faz parte de `session.json`.
* **Endpoints internos.** `/health`, `/internal/state` e
  `/internal/begin-shutdown` exigem `X-BYOM-Nonce` igual ao `nonce` da sessão
  (comparação em tempo constante, HTTP 403 caso contrário) e continuam aceitando
  apenas clientes loopback. O header é obrigatório porque o forward é
  `ssh -L 127.0.0.1:<porta>:127.0.0.1:<porta>`: todo cliente do túnel aparece
  como loopback. O nonce nunca vai em resposta de status nem em log.
* **SSH.** `init` define a chave privada com permissão `0600` após a geração. A
  chave de host é fixada **por sessão** em `<config_dir>/known_hosts_<sessão>`
  com `HostKeyAlias=byom-<sessão>`: `accept-new` na primeira conexão da sessão e
  `yes` nas seguintes, sem `/dev/null`. Se a chave mudar, o comando falha com
  "chave de host da sessão mudou; possível MITM — rode `byom down` e `byom up`"
  em vez de cair para `no`. O arquivo é removido **quando a desalocação é
  confirmada pelo Colab** (no `down` com confirmação ou no encerramento da
  sessão pelo `up --force`); se o `down` não confirma, o pin fica para a
  próxima tentativa.
* **Túnel.** `Address already in use`, `cannot listen`, `Permission denied` e
  `Host key verification failed` são fatais: o status fica `failed` com o motivo
  e o daemon segue de pé para `byom status` mostrar a causa. Falhas transitórias
  usam backoff exponencial e param depois de 10 ciclos sem health OK; o contador
  de falhas **consecutivas** só zera na reconexão que chega a health OK (um
  health isolado no `_monitor` não zera), enquanto `reconnects` é cumulativo.
* **OpenCode V1/V2.** A configuração gerada permanece no formato V1 (`plugin`,
  `provider`, `npm`, `options`, `tool_call`), que o guia de migração V2 indica
  ser normalizado em memória. O plugin de descoberta é fixado em
  `opencode-models-discovery@2.2.1` (`latest` no npm em 2026-11-23) em vez de
  `@latest`. O README do próprio plugin descreve um adaptador V1 separado, mas
  a execução do plugin V1 sob V2 **não foi verificada** neste ambiente.
  Revalide com a versão instalada do OpenCode antes de depender da descoberta
  automática de modelos.
* **Divergência da Colab CLI.** O README upstream documenta comandos pontuais
  como `colab ssh -s NOME -- CMD`; `src/byom/remote.py` registra que a CLI
  0.7.5 não aceita essa forma e utiliza um master multiplexado
  `colab ssh --proxy-mode` em vez disso. O comando **não foi alterado** e o
  comportamento/versão real da CLI **não foi verificado** aqui. Revalide contra
  `colab version` e o README upstream correspondente antes de portar esse
  comando.

## Desenvolvimento

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Os testes não acessam o Colab: utilizam um servidor vLLM simulado e um stub da
Colab CLI.

Para validar contra o Colab real (o script cria e encerra suas próprias sessões;
use `--keep` para mantê-las para inspeção):

```bash
uv run python scripts/spike.py --phase cheap   # CPU: túnel, tmux, watchdog, unassign()
uv run python scripts/spike.py --phase model   # GPU (T4 por padrão; --gpu escolhe outra): vLLM, decode, tool calling
```

A fase `model` baixa cerca de 20 GB de pesos de modelo e consome unidades de
computação. Use `--enforce-eager` se faltar memória.

### Estrutura do projeto

```text
src/byom/
  cli.py                  init, profiles, gpus, models, configure, up, status, down, setup, chat, web, uninstall
  llm_chat.py             setup idempotente, execução isolada de `llm chat` e uninstall seguro
  webui.py                frontend Open WebUI isolado (`byom web`)
  tui.py                  interface interativa (TUI) com Rich e questionary
  mensagens.py            textos, traduções, glossário e formatadores PT-BR da TUI
  catalog.py              catálogo validado de GPUs/modelos e checagem de compatibilidade
  profiles/               perfis de runtime (vLLM, GGUF, imagem, TTS) e aliases do OpenCode
  opencode.py             mescla a configuração do OpenCode
  dsh.py                  mescla a configuração do dsh (DeepSeek harness)
  colab.py                chamadas da Colab CLI e células de kernel
  tunnel.py, remote.py    conexão mestre SSH, túnel e comandos na VM
  daemon.py, proxy.py     processo de proxy local e supervisão do túnel
  watchdog_supervisor.py  mantém ativa a célula de watchdog no Colab
worker/                   arquivos enviados para a VM
  bootstrap.sh            instala dependências e inicia vLLM e gateway no tmux
  build_vllm_command.py   monta os argumentos do vLLM a partir do perfil ativo
  gateway.py              proxy de inferência, rastreamento de atividade e coordenação de desligamento
  watchdog.py             heartbeat e unassign()
scripts/spike.py          validação manual contra o Colab real
```
