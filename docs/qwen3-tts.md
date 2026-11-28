---
type: Feature
title: Qwen3-TTS Runtime
description: Feasibility and implementation of Qwen3-TTS on the Colab worker through vLLM-Omni, including the CustomVoice variants, the VRAM budget for the two-stage pipeline, and the /v1/audio/speech and /v1/audio/voices routes.
tags: [tts, qwen3-tts, vllm-omni, audio, colab]
status: draft
generated: { by: opencode/deepseek-v4.1-flash, at: 2026-11-27T22:30:00Z }
sources:
  - id: qwen3-tts-card
    resource: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    title: Qwen3-TTS 12Hz 1.7B CustomVoice card and file tree (consulted 2026-11-27)
  - id: qwen3-tts-small
    resource: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
    title: Qwen3-TTS 12Hz 0.6B CustomVoice card and file tree (consulted 2026-11-27)
  - id: qwen3-tts-repo
    resource: https://github.com/QwenLM/Qwen3-TTS
    title: Qwen3-TTS README, model list and vLLM-Omni pointer
  - id: vllm-omni-recipe
    resource: https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md
    title: vLLM-Omni Qwen3-TTS recipe, command and measured GPU notes
  - id: vllm-omni-speech
    resource: https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/
    title: vLLM-Omni speech API, /v1/audio/speech and /v1/audio/voices
  - id: catalog-module
    resource: /src/byom/catalog.py
    title: qwen3-tts model entry and the overhead_gib budget
  - id: build-module
    resource: /worker/build_vllm_command.py
    title: build_tts_args and the vLLM-Omni command
  - id: gateway-module
    resource: /worker/gateway.py
    title: TTS routes, readiness path and synthetic /v1/models
---

# Qwen3-TTS Runtime

Qwen3-TTS roda no Colab pelo **vLLM-Omni**, o runtime multi-stage da vLLM. A
receita oficial sobe o servidor com `vllm serve <modelo> --omni` e expõe a API
compatível com a OpenAI em `POST /v1/audio/speech` e `GET /v1/audio/voices`. O
byom ganhou o runtime `tts`, dois perfis CustomVoice (0.6B e 1.7B) e as rotas
correspondentes no gateway.

## O que foi verificado

* O repositório oficial lista cinco checkpoints: `1.7B` e `0.6B` em
  `VoiceDesign`, `CustomVoice` e `Base`, mais o tokenizer de 12 Hz. O byom
  adiciona as duas variantes **CustomVoice**, que têm 9 timbres embutidos e não
  precisam de áudio de referência.
* A receita do vLLM-Omni foi validada pelo projeto em uma **RTX 4090 de 24 GiB**
  (0.6B, deploy config padrão) e em uma **A100 de 40 GiB** (0.6B, pico de
  16.283 MiB com `gpu_memory_utilization: 0.3` por stage). Não há medição
  publicada para a **T4** nem para o **1.7B em L4**; o orçamento abaixo é uma
  estimativa de engenharia, marcada como tal.
* O pacote `vllm-omni` está no PyPI em **0.34.0**, exige Python `>=3.11,<3.14`
  e instala o vLLM correspondente. O worker usa Python 3.12, então o pin cabe.
* O vLLM-Omni **não publica `/v1/models`**. O gateway anuncia o checkpoint
  carregado por conta própria e usa `GET /v1/audio/voices` como sinal de
  prontidão.

## Bytes e orçamento de VRAM

Os pesos saíram da árvore da API do HF (`?blobs=true`, consultada em
2026-11-27, `lastModified` 2026-01-29):

| Variante | Arquivos | Peso | Pipeline | Total | GPU mínima |
| :--- | :--- | ---: | ---: | ---: | :--- |
| 0.6B CustomVoice | `model.safetensors` 1.811.626.576 B + `speech_tokenizer/model.safetensors` 682.293.092 B | 2,32 GiB | 6 GiB | 10,32 GiB | **T4** |
| 1.7B CustomVoice | `model.safetensors` 3.833.402.552 B + `speech_tokenizer/model.safetensors` 682.293.092 B | 4,21 GiB | 9 GiB | 15,21 GiB | **L4** |

A conta do byom soma pesos + 2 GiB de runtime + `Variant.overhead_gib` (o
pipeline talker + code2wav) + a folga da GPU. Na T4 (15 GiB, folga 1,0):
0.6B = 10,32 + 1,0 = **11,32 <= 15**; 1.7B = 15,21 + 1,0 = **16,21 > 15**. Na
L4 (24 GiB, folga 1,2): 1.7B = 15,21 + 1,2 = **16,41 <= 24**. O overhead é
deliberadamente conservador; o `gpu_memory_utilization` padrão da deploy config
é 0.3 por stage e o próprio recipe recomenda 0.15 para GPUs apertadas.

## Como o worker serve

1. `bootstrap.sh` instala, para `runtime = "tts"`, o `vllm_spec` (`vllm==0.34.0`)
   **e** o `[worker].vllm_omni_spec` (`vllm-omni==0.34.0`); o wheel do
   vllm-omni não declara o vLLM como dependência, então os dois pins são
   explícitos. O launcher é o `run_vllm.sh`, o mesmo do vLLM.
2. `prepare_runtime.py` não baixa pesos: como no vLLM, o `vllm serve` puxa o
   checkpoint do HF na primeira subida.
3. `build_tts_args()` monta:

   ```text
   /content/byom/venv/bin/vllm serve <model> --omni \
     --deploy-config <site-packages>/vllm_omni/deploy/qwen3_tts.yaml \
     --host 127.0.0.1 --port <vllm_port> \
     --served-model-name <served_model_name> --trust-remote-code
   ```

4. A prontidão do gateway usa `GET /v1/audio/voices`; `GET /v1/models` devolve
   um modelo sintético (`served_model_name`) e `POST /v1/chat/completions`
   responde 501, como no runtime de imagem.
5. O proxy local já encaminha qualquer `/v1/*`; a resposta de áudio é binária
   e passa pelo `StreamingResponse` sem parsing de JSON.

## Uso

```bash
uv run byom configure --profile qwen3-tts-0.6b-customvoice
uv run byom up
curl http://127.0.0.1:4000/v1/audio/voices
curl -X POST http://127.0.0.1:4000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Olá, isto é um teste.","voice":"vivian","language":"Portuguese"}' \
  --output fala.wav
```

O perfil 1.7B usa `qwen3-tts-1.7b-customvoice`. A saída padrão é WAV; a API
também aceita `stream`, `stream_format` e `response_format` do vLLM-Omni. Um
checkpoint por sessão: trocar de 0.6B para 1.7B exige `byom down` e `byom up`.

## Limites

* **CustomVoice apenas.** `VoiceDesign` (descrição em linguagem natural) e
  `Base` (clonagem com áudio de referência) usam o mesmo runtime e podem virar
  variantes depois; não estão no catálogo ainda.
* **Não medido no Colab.** O orçamento de 0.6B/1.7B é uma estimativa a partir
  da receita e dos bytes do HF. A primeira execução real deve ser tratada como
  validação: se o servidor OOM, reduza os dois `gpu_memory_utilization` da
  deploy config para 0.15.
* **Um runtime por sessão.** TTS não conversa; o frontend Open WebUI precisaria
  de uma conexão de áudio separada (`AUDIO_TTS_OPENAI_API_BASE_URL`), que o
  `byom web` ainda não configura.
* **Pins do vLLM e do vLLM-Omni.** O perfil fixa `vllm==0.34.0` e
  `vllm-omni==0.34.0`; a receita exige `>=0.20.0`. Um bump precisa conferir a
  deploy config e o suporte do wheel à CUDA do Colab. O wheel do vllm-omni
  0.34.0 foi baixado e inspecionado em 2026-11-27: contém
  `vllm_omni/deploy/qwen3_tts.yaml` e registra o entry point
  `vllm.general_plugins`, que é o que habilita `--omni` no `vllm serve`.
