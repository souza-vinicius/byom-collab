"""Textos, traduções, glossário e formatadores da interface interativa (TUI) do byom.

Toda a redação exibida pela TUI mora aqui para facilitar revisão e manutenção.
As mensagens da CLI permanecem em ``cli.py`` e ``catalog.py``; esta camada nunca
altera o catálogo nem a saída da CLI.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

TITULO = "byom"
SUBTITULO = "Modelos de código no Google Colab com API local em 127.0.0.1:4000/v1"

SOMENTE_LEITURA = "Somente leitura: não altera arquivos locais nem aloca GPU."
NADA_GRAVADO_ATE_CONFIRMAR = "Nada é gravado até você confirmar."
#: Trecho da mensagem da CLI (`cli._check_orphans`) que indica sessão alocada sem
#: registro local. A TUI compara com esta constante em vez de repetir o texto.
MARCADOR_SEM_REGISTRO = "no record in this byom"
AVISO_CANCELAR = "Pressione Ctrl+C a qualquer momento para cancelar."
ENTER_VOLTAR = "Pressione Enter para voltar ao menu..."
DICA_PAGINACAO = (
    "Terminal compacto ({largura}×{altura}): "
    "role com ↑↓/PageUp/PageDown ou use os números para abrir um item."
)

TECLAS_MENU = "↑↓ navegar • atalhos 1–5 / c / i / a / h / q • Enter selecionar • Ctrl+C sair"
TECLAS_SELECAO = "↑↓ navegar • digite para filtrar • Enter selecionar • Ctrl+C voltar"
TECLAS_LISTA = "↑↓ navegar • Enter abrir • Ctrl+C voltar"
TECLAS_CONFIRMACAO = "↑↓ ou s/n escolher • Enter confirmar • Ctrl+C cancelar"
TECLAS_DETALHE = "↑↓ navegar • Enter selecionar • Ctrl+C voltar"

MENU_PERGUNTA = "Escolha uma ação:"

#: Tela inicial única, na ordem de uso. A saída (`q. Sair`) é fixa na TUI (D3).
MENU_INICIAL: tuple[str, ...] = (
    "init",
    "selecionar_gpu",
    "up",
    "status",
    "down",
    "chat",
    "imagem",
    "audio",
    "harness",
)

ATALHOS_MENU: dict[str, str] = {
    "init": "1",
    "selecionar_gpu": "2",
    "up": "3",
    "status": "4",
    "down": "5",
    "chat": "c",
    "imagem": "i",
    "audio": "a",
    "harness": "h",
}

ACOES: dict[str, tuple[str, str]] = {
    "init": (
        "Preparar ambiente local",
        "Verifica ssh e Colab CLI, cria a chave SSH e salva config.toml.",
    ),
    "configure": (
        "Aplicar modelo e GPU",
        "Grava o modelo, a GPU e a janela de contexto em config.toml.",
    ),
    "selecionar_gpu": (
        "Selecionar GPU",
        "Escolhe a GPU e o modelo; a janela de contexto é a maior que cabe nela.",
    ),
    "harness": (
        "Configurar harness (OpenCode / dsh)",
        "Mostra instruções manuais e permite configurar a conexão automaticamente.",
    ),
    "up": (
        "Iniciar sessão no Colab",
        "Aloca a GPU, inicia o worker remoto e publica a API local.",
    ),
    "status": (
        "Verificar estado da sessão",
        "Consulta sessão no Colab, worker, túnel SSH e uso da conta.",
    ),
    "down": (
        "Encerrar sessão no Colab",
        "Desliga a VM remota e confirma a liberação da GPU no Colab.",
    ),
    "chat": (
        "Conversar com o modelo (llm chat)",
        "Abre um chat no terminal com o modelo servido no Colab; histórico fica só no byom.",
    ),
    "imagem": (
        "Gerar imagem",
        "Envia um prompt para o runtime de imagem e salva o PNG no diretório do byom.",
    ),
    "audio": (
        "Gerar áudio",
        "Envia um texto para o runtime TTS e salva o WAV no diretório do byom.",
    ),
}

#: Listas objetivas do que cada ação com efeito colateral/custo executa (D4).
O_QUE_VAI_ACONTECER: dict[str, tuple[str, ...]] = {
    "init": (
        "Verifica se `ssh` e a CLI do Colab estão instalados e autenticados.",
        "Cria a chave SSH ed25519 caso ainda não exista.",
        "Grava as configurações padrão locais em config.toml.",
    ),
    "configure": (
        "Atualiza o modelo, a GPU padrão e a janela de contexto em config.toml.",
        "Não aloca GPU nem inicia sessão no Colab.",
    ),
    "selecionar_gpu": (
        "Mostra as GPUs e os modelos que cabem em cada uma.",
        "A janela exibida é a maior que a VRAM da GPU permite, até o máximo do modelo.",
        "Nada é gravado em config.toml até você confirmar GPU + modelo.",
    ),
    "harness": (
        "Exibe as instruções para configurar manualmente o harness escolhido.",
        "Permite aplicar automaticamente a conexão em OpenCode ou dsh.",
    ),
    "up": (
        "Aloca uma máquina virtual com GPU no Colab (consome unidades de computação).",
        "Envia e inicializa o servidor do modelo e abre o túnel em 127.0.0.1:4000/v1.",
    ),
    "down": (
        "Encerra a sessão ativa e libera a GPU no Colab.",
        "Finaliza o daemon local e o túnel SSH.",
        "Arquivos temporários na VM remota serão perdidos; config.toml é mantido.",
    ),
    "chat": (
        "Ajusta o chat para a API local atual (grava só dentro da pasta do byom).",
        "Abre a conversa no terminal; digite `exit` ou `quit` para voltar ao menu.",
    ),
    "imagem": (
        "Envia o prompt para POST /v1/images/generations no proxy local.",
        "Salva o PNG em byom_home()/images; requer uma sessão com runtime de imagem.",
    ),
    "audio": (
        "Envia o texto para POST /v1/audio/speech no proxy local.",
        "Salva o WAV em byom_home()/audio; requer uma sessão com runtime TTS.",
    ),
}

CONFIRMACOES = {
    "init": "Deseja preparar o ambiente local agora?",
    "configure": "Deseja aplicar este modelo e esta GPU em config.toml?",
    "harness": "Deseja gravar automaticamente a conexão no arquivo do harness?",
    "up": "Deseja alocar a GPU e iniciar a sessão no Colab agora?",
    "up_force": ("Já existe uma sessão registrada. Deseja encerrá-la antes e criar uma nova?"),
    "down": "Deseja encerrar a sessão ativa e liberar a GPU agora?",
}

SELECAO_GPU = "Escolha a GPU que será usada na próxima sessão:"
SELECAO_HARNESS = "Qual harness de codificação você deseja configurar?"
SELECAO_PROXIMO = "O que fazer agora?"

SELECAO_MODELO_GPU = "Escolha o modelo desta GPU para gravar em config.toml:"
USAR_PAR_GPU_MODELO = "Usar este modelo nesta GPU"

PROMPT_IMAGEM = "Descreva a imagem que você quer gerar:"
SELECAO_TAMANHO_IMAGEM = "Escolha o tamanho da imagem:"
PASSOS_IMAGEM = "Passos (padrão 25):"
GUIDANCE_IMAGEM = "Guidance (padrão 1.0):"
PROMPT_AUDIO = "Digite o texto que será falado:"
SELECAO_VOZ_AUDIO = "Escolha a voz:"
SELECAO_IDIOMA_AUDIO = "Escolha o idioma:"
GERANDO_IMAGEM = "Gerando imagem..."
GERANDO_AUDIO = "Gerando áudio..."

TAMANHOS_IMAGEM: tuple[tuple[str, str], ...] = (
    ("1024x1024", "1024x1024 (quadrada)"),
    ("1152x864", "1152x864 (4:3)"),
    ("864x1152", "864x1152 (3:4)"),
    ("512x512", "512x512 (teste rápido)"),
)

VOZES_AUDIO: tuple[str, ...] = (
    "vivian",
    "serena",
    "uncle_fu",
    "dylan",
    "eric",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
)

IDIOMAS_AUDIO: tuple[str, ...] = (
    "Portuguese",
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Spanish",
    "Italian",
)

HARNESS_OPCOES: tuple[tuple[str, str, str, bool], ...] = (
    (
        "opencode",
        "OpenCode",
        "Configura provedor `byom`, plugin de descoberta e modelo padrão.",
        True,
    ),
    (
        "dsh",
        "dsh (DeepSeek harness)",
        "Configura `providers.byom` e modelo padrão em `cordis.patch.yml`.",
        True,
    ),
    (
        "aider",
        "Aider",
        "Exibe variáveis de ambiente e comando para usar a API local.",
        False,
    ),
    (
        "cline",
        "Cline (ou outro cliente OpenAI)",
        "Exibe URL base, chave e ID do modelo para preencher na extensão.",
        False,
    ),
)

COMANDO_BASE = "byom"
EQUIVALENTE_CLI = "Equivalente na CLI: {comando}"

PROXIMO_ACAO: dict[str, tuple[str, str]] = {
    "init": ("selecionar_gpu", "Selecionar GPU"),
    "selecionar_gpu": ("up", "Iniciar sessão no Colab"),
    "configure": ("up", "Iniciar sessão no Colab"),
    "harness": ("up", "Iniciar sessão no Colab"),
    "up": ("status", "Verificar estado da sessão"),
    "status": ("down", "Encerrar sessão no Colab"),
    "down": ("selecionar_gpu", "Selecionar GPU"),
    "chat": ("down", "Encerrar sessão no Colab"),
    "imagem": ("imagem", "Gerar outra imagem"),
    "audio": ("audio", "Gerar outro áudio"),
}

#: Mantido para compatibilidade e cobertura textual.
AJUDA_ACOES = {k: v[1] for k, v in ACOES.items()}
ANTES = {k: " ".join(v) for k, v in O_QUE_VAI_ACONTECER.items()}
ANTES["status"] = SOMENTE_LEITURA
PROXIMO = {k: f"{v[1]} (`byom {v[0]}`)" for k, v in PROXIMO_ACAO.items()}

ERRO_TITULO = "Não foi possível concluir"
ERRO = "O que aconteceu: {falha}\nCausa provável: {causa}\nO que fazer: {acao}"
DETALHES_TECNICOS_TITULO = "Detalhes técnicos"

ERROS_COMUNS = {
    "tty": (
        "Esta interface precisa de um terminal interativo.",
        "stdin ou stdout não são um TTY (saída redirecionada, pipe ou modo batch).",
        "Use um terminal real, execute um subcomando como `byom status` ou `byom --help`.",
    ),
    "dependencia": (
        "A biblioteca de menu interativo não está instalada.",
        "A dependência questionary não está disponível neste ambiente.",
        "Instale as dependências do projeto e tente novamente, ou use a CLI tradicional.",
    ),
    "cancelado": (
        "Operação cancelada pelo usuário.",
        "Você pressionou Ctrl+C, Ctrl+D/EOF, Esc ou recusou a confirmação.",
        "Nenhuma nova ação foi iniciada. Verifique o estado da sessão se já havia uma ativa.",
    ),
    "inesperado": (
        "Ocorreu um erro inesperado na interface.",
        "Falha interna na TUI ou em uma dependência.",
        "O detalhe técnico foi salvo em {log}. Consulte o arquivo antes de tentar novamente.",
    ),
    "vazio": (
        "Nenhum item encontrado em {origem}.",
        "O catálogo está vazio, o caminho mudou ou o filtro não corresponde a nada.",
        "Verifique a instalação do projeto ou tente outro filtro.",
    ),
    "caminho": (
        "Caminho inexistente ou inacessível: {caminho}.",
        "O arquivo/diretório não existe, foi movido ou o usuário não tem permissão.",
        "Confira o caminho e as permissões, depois tente novamente.",
    ),
}

TERMINAL_PEQUENO = (
    "Terminal compacto ({largura}×{altura}); usando visualização adaptada para caber na tela."
)

SUCESSO = "[OK]"
FALHA = "[ERRO]"
INFO = "[INFO]"

# ---------------------------------------------------------------------------
# Traduções e metadados de apresentação para o catálogo (D7, D8, D9)
# ---------------------------------------------------------------------------

FAMILIAS_MODELOS: dict[str, str] = {
    "qwen3.8-27b-awq": "Qwen3.8 27B",
    "qwen3.8-27b-gguf": "Qwen3.8 27B",
    "inkling-2-mini": "Inkling 2 Mini",
    "inkling-2-mini-strata": "Inkling 2 Mini (Strata)",
    "gemma-4": "Gemma 4",
    "muse-glimmer-30b": "Muse Glimmer 30B",
    "qwen-image-2.1": "Qwen-Image-2.1",
}

MODELOS_COM_VISAO: frozenset[str] = frozenset(
    {"inkling-2-mini", "qwen3.8-27b-awq", "qwen3.8-27b-gguf", "muse-glimmer-30b"}
)

ROTULOS_VARIANTES: dict[tuple[str, str], str] = {
    ("qwen3.8-27b-awq", "q4-awq"): "4 bits (AWQ)",
    ("qwen3.8-27b-gguf", "q3"): "3 bits (GGUF)",
    ("qwen3.8-27b-gguf", "q4"): "4 bits (GGUF)",
    ("inkling-2-mini", "q2"): "2 bits (GGUF)",
    ("inkling-2-mini", "q4"): "4 bits (GGUF)",
    ("inkling-2-mini", "q8"): "8 bits (GGUF)",
    # Receita AD da AtomicChat: o prefixo diz de quem é a quantização, porque o
    # mesmo modelo também tem GGUFs da Unsloth.
    ("inkling-2-mini", "ad-q6k"): "AD-Q6_K • 6 bits",
    ("inkling-2-mini", "ad-q4k"): "AD-Q4_K • 4 bits",
    ("inkling-2-mini", "ad-iq2xxs"): "AD-IQ2_XXS • 2 bits",
    ("gemma-4", "e2b-q4"): "E2B • 4 bits",
    ("gemma-4", "e4b-q4"): "E4B • 4 bits",
    ("gemma-4", "12b-q4"): "12B • 4 bits",
    ("gemma-4", "26b-a4b-q4"): "26B-A4B • 4 bits",
    ("gemma-4", "31b-q4"): "31B • 4 bits",
    ("muse-glimmer-30b", "q2"): "2 bits (GGUF)",
    ("muse-glimmer-30b", "q4"): "4 bits (GGUF)",
    ("muse-glimmer-30b", "q6"): "6 bits (GGUF)",
    ("qwen-image-2.1", "q4km"): "4 bits (GGUF)",
    ("qwen-image-2.1", "fp8-int8"): "8 bits (FP8)",
    # Packs `.strata` do engine Strata: o número do pack diz o tamanho, não o
    # número de bits; o KV é de 8 bits nos três. O rótulo longo (com o card de
    # destino) fica no catálogo, onde há espaço.
    ("inkling-2-mini-strata", "sq10"): "SQ-10 • 8 bits",
    ("inkling-2-mini-strata", "sq15"): "SQ-15 • 8 bits",
    ("inkling-2-mini-strata", "sq20"): "SQ-20 v2 • 8 bits",
}

NOMES_RUNTIMES: dict[str, str] = {
    "vllm": "vLLM",
    "llama-cpp": "llama.cpp",
    "image": "sd-server (geração de imagens)",
    "strata": "Strata (engine nativo, packs SQ)",
    "tts": "vLLM-Omni (texto para fala)",
}

ESFORCO_RACIOCINIO_PT: dict[str, str] = {
    "none": "desativado",
    "minimal": "mínimo",
    "low": "baixo",
    "medium": "médio",
    "high": "alto",
    "xhigh": "muito alto",
    "max": "máximo",
}

DESCRICOES_PERFIS: dict[str, str] = {
    "gemma-4-12b-q4-gguf": (
        "Modelo de 12B com quantização de 4 bits; ativa modo de raciocínio automaticamente."
    ),
    "gemma-4-26b-a4b-q4-gguf": ("Modelo MoE de 26B com quantização de 4 bits e suporte a visão."),
    "gemma-4-31b-q4-gguf": (
        "Modelo de 31B com quantização de 4 bits e suporte a visão; requer GPU A100."
    ),
    "gemma-4-e4b-q4-gguf": (
        "Modelo compacto E4B de 4 bits; ativa modo de raciocínio automaticamente."
    ),
    "inkling-2-mini-iq2-gguf": (
        "Quantização de 2 bits com suporte a visão; cabe na T4 com 32K de contexto."
    ),
    "inkling-2-mini-q4-gguf": (
        "Quantização recomendada de 4 bits com suporte a visão; cabe na L4 com 64K."
    ),
    "inkling-2-mini-q8-gguf": (
        "Quantização de 8 bits de alta fidelidade com visão; requer A100 de 40 GiB ou mais."
    ),
    "muse-glimmer-30b-q2-gguf": (
        "Modelo de 30B em 2 bits com suporte a visão; com o projetor, começa em L4."
    ),
    "muse-glimmer-30b-q4-gguf": (
        "Modelo de 30B em 4 bits com suporte a visão; requer A100 de 40 GiB."
    ),
    "qwen-image-2.1-fp8": ("Geração de imagens em 8 bits (FP8/INT8) para GPUs com 24 GiB ou mais."),
    "qwen-image-2.1-q4km": ("Geração de imagens compacta (4 bits) compatível com T4 ou L4."),
    "qwen3.8-27b-256k": ("Contexto completo de 256K para GPU A100 de 80 GiB (memória estendida)."),
    "qwen3.8-27b-32k": ("Perfil com contexto de 32K para GPU A100 padrão (40 GiB) ou L4."),
    "qwen3.8-27b-q3-gguf": ("Variante GGUF de 3 bits para L4; alternativa leve ao perfil AWQ."),
    "qwen3.8-27b-q4-gguf": ("Variante GGUF recomendada de 4 bits; requer L4 ou superior."),
}

MOTIVOS_CONTEXTO_PERFIS: dict[str, str] = {
    "qwen3.8-27b-32k": (
        "Na A100 de 40 GiB (sem memória estendida), "
        "limitar o contexto a 32K mantém o cache KV em 2 GiB."
    ),
    "qwen3.8-27b-256k": (
        "Na A100 de 80 GiB (memória estendida), "
        "há VRAM suficiente para servir todo o contexto de 256K."
    ),
    "inkling-2-mini-iq2-gguf": (
        "7,90 GiB de pesos + 0,65 de visão + 2 GiB do motor + 0,58 GiB de cache KV em "
        "32K (medido pela arquitetura do modelo) somam 11,13 GiB: com a folga de 1,0 GiB, "
        "12,13 GiB cabem nos 15 GiB da T4."
    ),
    "inkling-2-mini-q4-gguf": (
        "18 GiB dos pesos + 0,65 de visão + 2 GiB do motor + 1,08 GiB de cache KV em 64K "
        "(medido pela arquitetura) somam 21,73 GiB: com a folga de 1,2 GiB, 22,93 GiB "
        "cabem nos 24 GiB da L4."
    ),
    "inkling-2-mini-q8-gguf": (
        "Na A100 de 40 GiB, 24 GiB de pesos + 0,65 de visão + 2 GiB do motor + 2,08 GiB "
        "de cache KV em 128K (medido pela arquitetura) somam 28,73 GiB."
    ),
    "qwen3.8-27b-q3-gguf": (
        "12 GiB de pesos + 2 GiB do motor + 2 GiB de cache KV somam 16 GiB, acima dos "
        "15 GiB utilizáveis da T4; 32K roda em L4."
    ),
    "qwen3.8-27b-q4-gguf": (
        "Na L4 (24 GiB), 17 GiB dos pesos deixam espaço para "
        "2 GiB do motor e 2 GiB de cache KV (32K)."
    ),
    "gemma-4-e4b-q4-gguf": (
        "Na T4 (15 GiB utilizáveis), 6 GiB dos pesos permitem servir 64K de contexto "
        "(4 GiB de cache KV)."
    ),
    "gemma-4-12b-q4-gguf": (
        "Na T4 (15 GiB utilizáveis), 8 GiB dos pesos mantêm 32K de contexto com folga."
    ),
    "gemma-4-26b-a4b-q4-gguf": (
        "18 GiB dos pesos + 0,65 de visão + 2 GiB do motor deixam 1,2 GiB de margem em "
        "L4 e limitam o cache KV a 32K."
    ),
    "gemma-4-31b-q4-gguf": (
        "Na A100 de 40 GiB, 20 GiB dos pesos + 0,65 de visão + 2 GiB do motor + "
        "8 GiB de cache KV somam 30,65 GiB em 128K."
    ),
    "muse-glimmer-30b-q2-gguf": (
        "12 GiB de pesos + 0,65 de visão + 2 GiB do motor + 2 GiB de cache KV passam dos "
        "15 GiB da T4; 32K roda em L4."
    ),
    "muse-glimmer-30b-q4-gguf": (
        "17 GiB dos pesos + 0,65 de visão + 2 GiB do motor + 4 GiB de cache KV somam "
        "23,65 GiB: com a folga de 1,2 GiB, 64K exige A100 de 40 GiB."
    ),
    "qwen-image-2.1-q4km": (
        "Modelo de difusão para geração de imagens; não utiliza janela de contexto de texto."
    ),
    "qwen-image-2.1-fp8": (
        "Modelo de difusão para geração de imagens; não utiliza janela de contexto de texto."
    ),
}

NOTAS_MODELOS: dict[str, tuple[str, ...]] = {
    "inkling-2-mini": (
        "Arquitetura MoE de 26,4B parâmetros totais (5,4B ativos por token) com visão.",
        "Usa build mainline do llama.cpp com suporte a camadas loopadas.",
    ),
    "qwen3.8-27b-awq": ("Checkpoint AWQ servido via vLLM (philbert440/Qwen3.8-27B-W4A16-AWQ).",),
    "qwen3.8-27b-gguf": ("Modelo denso de 27B parâmetros com raciocínio e suporte a visão.",),
    "gemma-4": (
        "O gateway ativa o modo de pensamento automaticamente "
        "e remove blocos internos do histórico.",
    ),
    "muse-glimmer-30b": ("Modelo denso de 30B com visão; utiliza release fixada do llama.cpp.",),
    "qwen-image-2.1": (
        "Modelo de difusão atendido em POST /v1/images/generations (não usa /v1/chat/completions).",
        "Padrão da API: 25 passos de geração e guidance 1,0.",
    ),
}

NOTAS_GPUS: dict[str, str] = {
    "T4": "15 GiB utilizáveis no Colab; sem bf16, geração de imagem usa fp16.",
    "L4": "Equilíbrio entre custo e memória (24 GiB).",
    "A100-40": "No Colab: A100 com memória padrão (high_mem desativado).",
    "A100-80": "No Colab: A100 com memória estendida (high_mem ativado).",
    "H100-80": "Maior desempenho disponível no catálogo.",
}

FONTE_GPUS_RESUMO = (
    "Fonte de VRAM: Google Cloud (Compute Engine > GPUs, consultado em {data}). "
    "No Colab, a disponibilidade e os modelos exatos de GPU podem variar."
)

#: Glossário curto (1 frase por termo) — exibido apenas com termos presentes na tela (D9).
GLOSSARIO: dict[str, str] = {
    "Contexto": "Quantidade máxima de tokens (entrada + saída) que o modelo processa de uma vez.",
    "Cache KV": (
        "Memória reservada durante a conversa para guardar "
        "o histórico de tokens sem recalcular tudo."
    ),
    "GGUF": "Formato de arquivo quantizado usado pelo llama.cpp para rodar modelos com menos VRAM.",
    "AWQ": (
        "Técnica de quantização de 4 bits otimizada para rodar rápido em placas NVIDIA via vLLM."
    ),
    "VRAM (~)": "Memória dedicada da placa de vídeo; o símbolo ~ indica valor aproximado.",
    "bf16": (
        "Formato numérico de 16 bits acelerado em GPUs recentes "
        "(L4, A100 e H100) para mais estabilidade."
    ),
    "Raciocínio": "Intensidade com que o modelo pensa passo a passo antes de responder.",
    "Visão": "Capacidade de interpretar imagens enviadas na conversa além de texto.",
    "Fornecedor": "Time ou empresa que desenvolveu o modelo; o catálogo guarda a fonte do nome.",
    "Publicado por": (
        "Quem publicou o arquivo quantizado de onde o byom baixa os pesos "
        "(pode não ser quem desenvolveu o modelo)."
    ),
}


def glossario_para_tela(termos: Sequence[str]) -> list[tuple[str, str]]:
    """Retorna pares (termo, definição) apenas para os termos presentes na tela."""
    vistos: set[str] = set()
    itens: list[tuple[str, str]] = []
    for termo in termos:
        if termo in GLOSSARIO and termo not in vistos:
            vistos.add(termo)
            itens.append((termo, GLOSSARIO[termo]))
    return itens


# ---------------------------------------------------------------------------
# Formatadores puros em pt-BR (D7, D8, D10)
# ---------------------------------------------------------------------------


def formatar_numero(valor: int | float) -> str:
    """Formata inteiros e decimais no padrão pt-BR (milhar '.', decimal ',')."""
    if isinstance(valor, bool):
        raise TypeError("Booleano não é um número formatável")
    if isinstance(valor, int) or float(valor).is_integer():
        inteiro = int(valor)
        sinal = "-" if inteiro < 0 else ""
        digitos = f"{abs(inteiro):,}".replace(",", ".")
        return f"{sinal}{digitos}"
    texto = f"{valor:.2f}".rstrip("0").rstrip(".")
    inteira, _, fracao = texto.partition(".")
    inteira_fmt = f"{int(inteira):,}".replace(",", ".")
    return f"{inteira_fmt},{fracao}" if fracao else inteira_fmt


def formatar_gib(valor: float, *, aproximado: bool = False) -> str:
    """Formata tamanho de memória em GiB no padrão pt-BR."""
    prefixo = "~" if aproximado else ""
    return f"{prefixo}{formatar_numero(round(valor, 1))} GiB"


def formatar_contexto(tokens: int, runtime: str = "") -> str:
    """Converte contagem de tokens em K/M legível ou 'não se aplica' para imagem/TTS."""
    if runtime == "tts":
        return "fala"
    if runtime == "image" or tokens <= 1:
        return "não se aplica"
    if tokens >= 1_000_000:
        if tokens in (1_000_000, 1_048_576):
            return "1M"
        m = tokens / 1_048_576
        if abs(m - round(m)) < 0.05:
            return f"{int(round(m))}M"
        return f"{formatar_numero(round(tokens / 1_000_000, 1))}M"
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    if tokens >= 1000:
        return f"{formatar_numero(round(tokens / 1000, 1))}K"
    return formatar_numero(tokens)


def formatar_contexto_detalhe(tokens: int, runtime: str = "") -> str:
    """Formato detalhado do contexto: '32K (32.768 tokens)' ou 'não se aplica'."""
    if runtime == "tts":
        return "não se aplica (modelo de fala)"
    if runtime == "image" or tokens <= 1:
        return "não se aplica (modelo de geração de imagens)"
    return f"{formatar_contexto(tokens, runtime)} ({formatar_numero(tokens)} tokens)"


def formatar_data_ptbr(texto: str) -> str:
    """Converte data ISO (YYYY-MM-DD ou ISO 8601) para dd/mm/aaaa."""
    limpo = texto.strip()
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}:\d{2}))?", limpo)
    if not match:
        return limpo
    ano, mes, dia, hora = match.groups()
    try:
        datetime(int(ano), int(mes), int(dia))
    except ValueError:
        return limpo
    data_fmt = f"{dia}/{mes}/{ano}"
    return f"{data_fmt} às {hora}" if hora else data_fmt


def encurtar_caminho(caminho: Path | str) -> str:
    """Substitui o diretório home do usuário por '~' quando possível."""
    texto = str(caminho)
    home = str(Path.home())
    if home and (texto == home or texto.startswith(home + "/")):
        return "~" + texto[len(home) :]
    return texto


def traduzir_raciocinio(
    esforco: str | None, *, tem_thinking_prompt: bool = False, runtime: str = ""
) -> str:
    """Traduz o nível de raciocínio do catálogo para português simples."""
    if runtime == "image":
        return "não se aplica"
    if esforco is None:
        return "automático" if tem_thinking_prompt else "não informado"
    chave = esforco.strip().lower()
    return ESFORCO_RACIOCINIO_PT.get(chave, chave)


def nome_familia(model_id: str, fallback: str = "") -> str:
    """Nome amigável da família do modelo para agrupamento na TUI."""
    return FAMILIAS_MODELOS.get(model_id, fallback or model_id)


#: Rótulos do fornecedor (quem desenvolve) e do publicador dos pesos.
#: Fonte única: TUI, CLI e testes importam daqui.
ROTULO_FORNECEDOR = "Fornecedor"
ROTULO_PUBLICADO_POR = "Publicado por"
COLUNA_FORNECEDOR = "Fornecedor"
COLUNA_PACOTE = "Pacote"


def publicador_diverge(fornecedor: str, pacote: str) -> bool:
    """O pacote de pesos vem de outra org? Só nesse caso vale exibir o segundo."""
    fornecedor = (fornecedor or "").strip()
    pacote = (pacote or "").strip()
    return bool(pacote) and pacote != fornecedor


def fornecedor_resumo(fornecedor: str, pacote: str = "") -> str:
    """Linha curta: fornecedor e, apenas quando divergir, quem publicou os pesos."""
    fornecedor = (fornecedor or "").strip()
    pacote = (pacote or "").strip()
    if not publicador_diverge(fornecedor, pacote):
        return fornecedor or pacote
    if not fornecedor:
        return f"pesos {pacote}"
    return f"{fornecedor} · pesos {pacote}"


def fornecedor_curto(fornecedor: str) -> str:
    """Nome do fornecedor sem o parêntese explicativo, para tabela estreita.

    "Qwen (Alibaba Cloud)" -> "Qwen". A forma completa continua no detalhe, na
    lista de seleção e na coluna dedicada de terminal largo.
    """
    nome = (fornecedor or "").strip()
    return nome.split(" (", 1)[0].strip() or nome


def linhas_fornecedor(fornecedor: str, pacote: str) -> list[tuple[str, str]]:
    """Pares (rótulo, valor) para o detalhe: uma linha quando não divergem."""
    fornecedor = (fornecedor or "").strip()
    pacote = (pacote or "").strip()
    linhas: list[tuple[str, str]] = []
    if fornecedor:
        linhas.append((ROTULO_FORNECEDOR, fornecedor))
    if publicador_diverge(fornecedor, pacote) or (not fornecedor and pacote):
        linhas.append((ROTULO_PUBLICADO_POR, pacote))
    return linhas


def rotulo_variante(model_id: str, variant_id: str, fallback: str = "") -> str:
    """Rótulo curto de quantização/variante para exibição em tabela."""
    return ROTULOS_VARIANTES.get((model_id, variant_id), fallback or variant_id)


def descricao_perfil(profile_id: str, fallback: str = "") -> str:
    """Descrição em português claro de um perfil."""
    return DESCRICOES_PERFIS.get(profile_id, fallback)


def motivo_contexto_perfil(profile_id: str, fallback: str = "") -> str:
    """Explicação em português claro do limite de contexto do perfil."""
    return MOTIVOS_CONTEXTO_PERFIS.get(profile_id, fallback)


def traduzir_estado_worker(estado: str | None) -> str:
    """Traduz estado interno do worker/túnel para português."""
    mapa = {
        "ready": "pronto",
        "starting": "inicializando",
        "loading": "carregando modelo",
        "installing": "instalando dependências",
        "unavailable": "indisponível",
        "connected": "conectado",
        "connecting": "conectando",
        "reconnecting": "reconectando",
        "disconnected": "desconectado",
        "error": "com erro",
        "waiting": "aguardando",
    }
    if not estado:
        return "desconhecido"
    return mapa.get(estado.strip().lower(), estado)
