"""Textos da interface interativa do byom.

Toda a redação exibida pela TUI mora aqui para facilitar revisão e tradução.
As mensagens da CLI permanecem em ``cli.py``; esta camada nunca as reescreve.
"""

from __future__ import annotations

TITULO = "byom — modelo de código no Colab"
SUBTITULO = "Escolha uma ação. A CLI continua disponível para scripts e pipes."

TECLAS = "↑↓ navegar • Enter selecionar • Esc voltar • h ajuda • q sair • Ctrl+C cancelar"

MENU_PERGUNTA = "O que você quer fazer?"
MENU_AJUDA = (
    "Este menu reúne as mesmas ações da CLI. Use ↑ e ↓ para navegar, Enter para "
    "executar, h para ver esta ajuda e q ou Ctrl+C para sair. Nenhuma ação é "
    "executada apenas por navegar."
)

ACOES = {
    "init": ("Configurar o ambiente", "Cria a chave SSH e o arquivo de configuração local."),
    "profiles": (
        "Ver perfis de modelo",
        "Lista modelos, contextos e GPUs disponíveis no catálogo.",
    ),
    "gpus": ("Ver GPUs", "Mostra GPUs, VRAM aproximada e perfis que cabem em cada uma."),
    "models": ("Explorar modelos", "Filtra o catálogo por GPU e mostra o que cabe na memória."),
    "configure": ("Escolher perfil", "Grava o modelo e a GPU que serão usados no próximo `up`."),
    "up": (
        "Subir modelo no Colab",
        "Aloca uma GPU, inicia o worker e publica a API em 127.0.0.1:4000.",
    ),
    "status": ("Ver status", "Mostra sessão, worker, túnel, watchdog e uso da conta."),
    "down": ("Desligar sessão", "Encerra a VM e confirma que ela foi liberada no Colab."),
}

AJUDA_ACOES = {
    "init": "Verifica ssh e a CLI do Colab, gera a chave SSH e salva config.toml.",
    "profiles": "Leitura do catálogo local; não altera arquivos nem aloca GPU.",
    "gpus": "Leitura do catálogo local; a VRAM é aproximada e não consulta sua conta.",
    "models": "Filtra o catálogo por GPU. Não consulta a conta nem grava configuração.",
    "configure": (
        "Só funciona sem sessão ativa. Para trocar de perfil com sessão ativa, "
        "use `down` primeiro."
    ),
    "up": "Aloca GPU e gasta unidades de computação. Pode levar vários minutos no primeiro uso.",
    "status": "Leitura de estado local, túnel e uso da conta; não altera nada.",
    "down": "Interrompe a sessão ativa e libera a GPU. Esta ação é irreversível.",
}

SELECAO_GPU = "Qual GPU você quer usar como filtro?"
SELECAO_PERFIL = "Qual perfil você quer aplicar?"
_busca = " (digite para filtrar)"

IGUAL_ARQUIVO = "Para repetir sem o menu: {comando}"

ANTES = {
    "init": "Vou verificar ssh e a CLI do Colab, depois criar a chave e a configuração local.",
    "profiles": "Vou ler o catálogo de perfis e mostrar modelo, contexto e GPU de cada um.",
    "gpus": "Vou ler o catálogo de GPUs e calcular quais perfis cabem em cada uma.",
    "models": "Vou filtrar o catálogo pela GPU escolhida e mostrar os modelos compatíveis.",
    "configure": "Vou gravar o perfil escolhido em config.toml. Nenhuma GPU será alocada agora.",
    "up": "Vou criar uma sessão no Colab, enviar o worker e iniciar a API. Isso gasta créditos.",
    "status": "Vou consultar o estado local, o túnel e o uso da conta.",
    "down": "Vou encerrar a sessão atual e liberar a GPU no Colab.",
}

DURANTE = "Trabalhando agora: {detalhe}"
AGUARDANDO = "Aguardando o Colab: {detalhe}"

RESUMO_TITULO = "Concluído"
RESUMO = (
    "Ação: {acao}\n"
    "Resultado: {resultado}\n"
    "Duração: {duracao:.1f}s\n"
    "Próximo passo sugerido: {proximo}"
)

PROXIMO = {
    "init": "`byom configure` para escolher um modelo.",
    "profiles": "`byom configure` para aplicar um perfil.",
    "gpus": "`byom models` para ver modelos que cabem na GPU.",
    "models": "`byom configure` para gravar sua escolha.",
    "configure": "`byom up` para iniciar a sessão.",
    "up": "`byom status` para acompanhar e `byom down` para encerrar.",
    "status": "`byom down` se a sessão não for mais necessária.",
    "down": "`byom gpus` para planejar a próxima sessão.",
}

CONFIRMACOES = {
    "up": (
        "Isso vai alocar uma GPU no Colab, baixar o modelo e iniciar a API. "
        "Sua conta será cobrada em unidades de computação até o desligamento. Continuar?"
    ),
    "up_force": (
        "Já existe uma sessão registrada. `--force` vai desligá-la antes de criar outra. "
        "O modelo será recarregado do zero. Continuar?"
    ),
    "down": (
        "Isso vai encerrar a sessão e liberar a GPU. Modelos e arquivos temporários da VM "
        "serão perdidos; a configuração local será mantida. Continuar?"
    ),
}

ERRO_TITULO = "Não foi possível concluir"
ERRO = "O que aconteceu: {falha}\nCausa provável: {causa}\nO que fazer: {acao}"

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
        "Você pressionou Ctrl+C, Esc ou recusou a confirmação.",
        "Nenhuma nova ação foi iniciada. Verifique `byom status` se uma sessão já existia.",
    ),
    "inesperado": (
        "Ocorreu um erro inesperado na interface.",
        "Falha interna na TUI ou em uma dependência.",
        "O detalhe técnico foi salvo em {log}. Abra um issue com esse arquivo.",
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
    "Aviso: o terminal parece pequeno ({largura}x{altura}). "
    "Tabelas podem ser cortadas; amplie a janela ou use a CLI para ver a saída completa."
)

SUCESSO = "[OK]"
FALHA = "[ERRO]"
INFO = "[INFO]"

COMANDO_BASE = "byom"
