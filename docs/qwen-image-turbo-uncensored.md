---
type: Model
title: Qwen-Image-2.1 Turbo Uncensored
description: The Qwen-Image-2.1 Turbo image pipeline with AtomicChat's abliterated Qwen3-VL text encoder, served by sd-server in two quantization pairs, including the fixed 8-step sigma schedule the Turbo checkpoint requires.
tags: [imagem, qwen-image, turbo, uncensored, gguf, sd-server, sigmas, vram]
status: draft
generated: { by: opencode/deepseek-v4.1-flash, at: 2026-11-28T12:00:00Z }
sources:
  - id: qwen-turbo-repo
    resource: https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo
    title: Official Qwen-Image-2.1 Turbo checkpoint (Qwen Research License)
  - id: turbo-card
    resource: https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF
    title: AtomicChat Turbo denoiser GGUFs and the fixed 8-step schedule (2026-11-28)
  - id: uncensored-card
    resource: https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF
    title: Abliterated Qwen3-VL encoder, file ladder and what it changes (2026-11-28)
  - id: sdcpp-api
    resource: https://github.com/leejet/stable-diffusion.cpp/blob/0518b34/examples/server/api.md
    title: sd-server API, sample_params.custom_sigmas and sd_cpp_extra_args
  - id: sdcpp-qwen
    resource: https://github.com/leejet/stable-diffusion.cpp/blob/0518b34/docs/qwen_image_2.1.md
    title: Qwen-Image-2.1 Turbo preset qwen-image-2.1-turbo and --sampling-method euler
  - id: catalog-module
    resource: /src/byom/catalog.py
    title: ModelEntry qwen-image-2.1-turbo-uncensored, variants and generation defaults
  - id: gateway
    resource: /worker/gateway.py
    title: Injection of sample_method and custom_sigmas into sd_cpp_extra_args
---

# Qwen-Image-2.1 Turbo Uncensored

`qwen-image-2.1-turbo-uncensored` é o pipeline de imagem do **Qwen-Image-2.1
Turbo** com o encoder de texto **Qwen3-VL-8B abliterado** da AtomicChat. O
denoiser de 7B é o Turbo da própria Qwen, sem alteração; o encoder é o único
componente trocado.[^uncensored-card][^qwen-turbo-repo]

O byom serve os quatro papéis do sd-server, cada um com uma flag confirmada na
documentação da build fixada:[^catalog-module]

| Papel | Flag | Arquivo nos dois perfis |
| :--- | :--- | :--- |
| `diffusion` | `--diffusion-model` | `Qwen-Image-2.1-Turbo-{AD-Q4_K,Q8_0}.gguf` |
| `vae` | `--vae` | `vae/qwen_image_2.1_vae_bf16.safetensors` (Comfy-Org) |
| `text_encoder` | `--llm` | `Qwen-Image-2.1-Turbo-Uncensored-{AD-Q4_K,Q8_0}.gguf` |
| `vision` | `--llm_vision` | `mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf` |

## Agenda fixa: 8 passos, CFG 1, euler, 9 sigmas

O Turbo é um checkpoint destilado: o card do denoiser diz que sem a agenda
própria o stable-diffusion.cpp usa a agenda do Qwen-Image-2.1 (25 passos) e a
imagem sai diferente.[^turbo-card] O mínimo que o perfil precisa entregar é:

```text
steps = 8
guidance (txt_cfg) = 1.0
sample_method = euler
custom_sigmas = 1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0
```

A lista é o `sample_sigmas` do `model_index.json` do checkpoint mais o `0.0`
final; o shift do scheduler é 1.0, então os valores são usados como estão. A
doc da build fixada publica o mesmo conjunto como preset
`qwen-image-2.1-turbo`.[^turbo-card][^sdcpp-qwen]

`ModelEntry.generation_defaults` carrega os quatro valores. O gateway mapeia
`steps`/`guidance` para `sample_params.sample_steps`/`sample_params.guidance.txt_cfg`
e acrescenta `sample_params.sample_method` e `sample_params.custom_sigmas` no
bloco `sd_cpp_extra_args` dentro de `prompt`, em
`POST /v1/images/generations`. O servidor documenta que, com
`custom_sigmas`, a **quantidade de sigmas menos um** determina os passos
reais.[^sdcpp-api][^gateway] Um cliente que envie o próprio bloco
`sd_cpp_extra_args` mantém o controle e recebe o corpo intacto.

`POST /v1/images/edits` é multipart e o gateway repassa o corpo sem reescrever:
para editar com o Turbo, o cliente precisa enviar o bloco
`sd_cpp_extra_args` no campo `prompt`, ou aceita a agenda padrão do servidor
(25 passos), que não é a agenda para a qual o checkpoint foi destilado.

A lista de sigmas exige stable-diffusion.cpp com o fix `c150a6b` (2026-10-06)
ou mais novo. A build Unsloth fixada no perfil,
`master-982-0518b34-u27b9ddc`, é de 2026-11-15 e já inclui o fix (e o preset);
nada é baixado de `latest`.

## Perfis e orçamento

Os dois perfis usam o binário Unsloth `cuda13` já fixado nos perfis de
Qwen-Image e o mesmo VAE. Cada linha é a soma dos bytes publicados na API do HF
em 2026-11-28, em GiB, mais a reserva de runtime de 2 GiB e a folga obrigatória
de `max(1 GiB, 5% da VRAM)`:[^catalog-module]

| Perfil | Encoder | Denoiser | Pesos | Visão | Runtime | Required | Folga | GPU | Total |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| `qwen-image-2.1-turbo-uncensored-q4k` | Unc AD-Q4_K 4,92 | AD-Q4_K 3,91 | 9,47 | 1,08 | 2,00 | 12,55 | 1,00 | **T4** | 13,55 ≤ 15 |
| `qwen-image-2.1-turbo-uncensored-q8` | Unc Q8_0 8,11 | Q8_0 7,07 | 15,81 | 1,08 | 2,00 | 18,89 | 1,20 | L4 | 20,09 ≤ 24 |

O VAE (`675.509.688 B` = 0,63 GiB) entra em `Pesos` nas duas linhas. O projetor
de visão F16 (`1.159.029.984 B` = 1,08 GiB) é opcional para gerar e obrigatório
para editar. Modelo de difusão não tem cache KV autoregressivo: o contexto é a
sentinela 1 e `required_vram_gib` não soma KV.

A escolha do par segue os dois cards: `AD-Q4_K` é o degrau de 4 bits
recomendado e `Q8_0` o mais próximo do BF16 quando cabe. O encoder AD-Q4_K
move as imagens em LPIPS 0,087 contra o BF16; o Q8_0 do denoiser em 0,037 e o
Q8_0 do encoder em 0,031. Nenhum número de velocidade foi medido no Colab para
os 8 passos.

## O que "uncensored" muda

Só o encoder. A AtomicChat projeta fora dos pesos a direção de recusa do
Qwen3-VL e mantém a torre de visão intacta; o denoiser não tem mecanismo de
recusa e continua byte a byte o do Turbo.[^uncensored-card]

No próprio card, 45 prompts sensíveis (adultos) e 48 neutros foram comparados
com o encoder original no mesmo seed: **0 de 45** prompts mudaram de resultado
e as imagens mudaram tanto em prompts neutros quanto nos sensíveis (LPIPS
0,084-0,090, contra 0,499 de um seed novo). O efeito medido está no encoder
como modelo de chat: as recusas caem de 88,9% para 1,2%, com MMLU estável em
77,35%. O byom serve o encoder no papel `--llm`, onde ele não conversa; o
`mmproj` habilita edição com imagem de referência.

Licenças: o encoder abliterado é Apache-2.0; o denoiser Turbo é Qwen Research
License. O uso responsável está no card e não é alterado pelo catálogo.

## Limites conhecidos

- **Não executado no Colab nesta integração.** O argv, os papéis e o orçamento
  são verificados por testes offline; a primeira execução real deve confirmar
  download, carga e a imagem de 8 passos.
- **Sem medição de tempo.** O guia da Unsloth mede 1024×1024 em 25 passos na
  T4; o perfil Turbo usa 8 passos, mas a diferença não foi medida.
- **O 1024×1024 é o tamanho medido pelos cards.** O card do Turbo recomenda
  cerca de 2 megapixels (2048×2048 em 1:1) para qualidade; nenhuma resolução
  além de 1024×1024 foi medida.
- **`custom_sigmas` não é sobrescrito pelo cliente por campo.** Quem quiser
  outra agenda envia o próprio `sd_cpp_extra_args`; aí o gateway não injeta
  nada.
- **Edição não recebe a agenda fixa por padrão.** `/v1/images/edits` é
  multipart e o gateway não reescreve o campo `prompt`; o perfil declara
  `vision`, então o sd-server aceita a edição, mas a agenda é a do servidor
  (25 passos) a menos que o cliente embuta `sd_cpp_extra_args` no `prompt`. O
  card do Turbo não mediu edição com imagem de referência.

[^qwen-turbo-repo]: https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo (2026-11-28)
[^turbo-card]: https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF (2026-11-28)
[^uncensored-card]: https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF (2026-11-28)
[^sdcpp-api]: https://github.com/leejet/stable-diffusion.cpp/blob/0518b34/examples/server/api.md (2026-11-28)
[^sdcpp-qwen]: https://github.com/leejet/stable-diffusion.cpp/blob/0518b34/docs/qwen_image_2.1.md (2026-11-28)
[^catalog-module]: `/src/byom/catalog.py`
[^gateway]: `/worker/gateway.py`
