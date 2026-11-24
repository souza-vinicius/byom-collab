"""Interface interativa (TUI) do byom, construída sobre Typer, Rich e questionary.

Esta camada renderiza dados estruturados com componentes Rich em português simples,
sem reescrever a CLI tradicional nem alterar o catálogo.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from . import catalog, config, dsh, llm_chat, opencode
from . import mensagens as M

# ---------------------------------------------------------------------------
# Console, símbolos e largura/altura responsiva (D10, D11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Simbolos:
    ok: str
    erro: str
    aviso: str
    pendente: str
    ativo: str
    seta: str
    estrela: str
    barra: str
    bullet: str


UNICODE_SIMBOLOS = Simbolos(
    ok="✔",
    erro="✘",
    aviso="⚠",
    pendente="○",
    ativo="●",
    seta="→",
    estrela="★",
    barra="│",
    bullet="•",
)

ASCII_SIMBOLOS = Simbolos(
    ok="[OK]",
    erro="[ERRO]",
    aviso="[AVISO]",
    pendente="[ ]",
    ativo="[*]",
    seta="->",
    estrela="*",
    barra="|",
    bullet="-",
)


def _suporta_unicode(con: Console | None = None) -> bool:
    encoding = (
        getattr(con, "encoding", None) or getattr(sys.stdout, "encoding", None) or ""
    ).lower()
    if not encoding:
        return False
    return "utf" in encoding


def obter_simbolos(con: Console | None = None) -> Simbolos:
    return UNICODE_SIMBOLOS if _suporta_unicode(con) else ASCII_SIMBOLOS


def criar_console(
    *,
    width: int | None = None,
    height: int | None = None,
    record: bool = False,
) -> Console:
    no_color = bool(os.environ.get("NO_COLOR"))
    return Console(
        highlight=False,
        no_color=no_color,
        width=width,
        height=height,
        record=record,
    )


console: Console = criar_console()


def is_interactive() -> bool:
    """Verdadeiro apenas quando entrada e saída são um terminal real."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def limpar_tela(con: Console) -> None:
    """Limpa a tela apenas quando a saída é um terminal interativo real."""
    if con.is_terminal:
        con.clear()


def _largura_terminal(con: Console) -> int:
    return int(con.size.width or 100)


def _altura_terminal(con: Console) -> int:
    return int(con.size.height or 30)


def _modo_layout(con: Console) -> str:
    """Retorna 'completo' (>=100), 'compacto' (80..99) ou 'cartoes' (<80)."""
    largura = _largura_terminal(con)
    if largura >= 100:
        return "completo"
    if largura >= 80:
        return "compacto"
    return "cartoes"


# ---------------------------------------------------------------------------
# Estado local barato e barra de estado (D2, D3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EstadoLocal:
    config_existe: bool
    chave_ssh_existe: bool
    ambiente_configurado: bool
    perfil_id: str
    perfil_nome: str
    gpu_id: str
    gpu_rotulo: str
    sessao_ativa: bool
    sessao_nome: str
    sessao_criada_em: str
    erro_estado: str = ""
    erro_config: str = ""


def ler_estado_local() -> EstadoLocal:
    """Lê apenas arquivos locais (sem rede ou subprocessos) para a barra de estado."""
    cfg_path = config.config_path()
    config_existe = cfg_path.exists()
    erro_config = ""
    try:
        cfg = config.load_config()
    except config.ConfigError as exc:
        cfg = config.Config()
        erro_config = str(exc)

    chave_ssh_existe = cfg.ssh_key_path.exists()
    ambiente_configurado = config_existe and chave_ssh_existe

    perfil_id = cfg.profile
    perfil_nome = perfil_id
    if perfil_id:
        try:
            perfil_nome = config.load_profile(perfil_id).name
        except ValueError:
            perfil_nome = perfil_id

    try:
        gpu = catalog.resolve_gpu(cfg.gpu, cfg.gpu_high_mem)
        gpu_id = gpu.id
        gpu_rotulo = gpu.label
    except ValueError:
        gpu_id = cfg.gpu
        gpu_rotulo = cfg.gpu

    erro_estado = ""
    sessao_ativa = False
    sessao_nome = ""
    sessao_criada_em = ""
    try:
        state = config.load_state()
        if state is not None:
            sessao_ativa = True
            sessao_nome = state.session
            sessao_criada_em = state.created_at
    except config.StateFileError as exc:
        erro_estado = str(exc)

    return EstadoLocal(
        config_existe=config_existe,
        chave_ssh_existe=chave_ssh_existe,
        ambiente_configurado=ambiente_configurado,
        perfil_id=perfil_id,
        perfil_nome=perfil_nome,
        gpu_id=gpu_id,
        gpu_rotulo=gpu_rotulo,
        sessao_ativa=sessao_ativa,
        sessao_nome=sessao_nome,
        sessao_criada_em=sessao_criada_em,
        erro_estado=erro_estado,
        erro_config=erro_config,
    )


def proximo_passo_recomendado(estado: EstadoLocal) -> str:
    """Determina o próximo passo recomendado dado o estado local."""
    if not estado.ambiente_configurado:
        return "init"
    if estado.sessao_ativa:
        return "status"
    if not estado.perfil_id:
        return "selecionar_gpu"
    return "up"


def _texto_barra_estado(estado: EstadoLocal, sim: Simbolos, compacto: bool) -> Text:
    texto = Text()
    texto.append("Ambiente: ", style="dim")
    if estado.erro_config:
        texto.append(f"{sim.erro} erro na config", style="red")
    elif estado.ambiente_configurado:
        texto.append(f"{sim.ok} configurado", style="green")
    else:
        texto.append(f"{sim.aviso} pendente", style="yellow")

    sep = "\n" if compacto else f"  {sim.barra}  "
    texto.append(sep, style="dim")

    texto.append("Perfil: ", style="dim")
    if estado.perfil_id:
        texto.append(f"{estado.perfil_nome}", style="cyan")
        texto.append(f" ({estado.gpu_id})", style="dim")
    else:
        texto.append(f"pendente (GPU {estado.gpu_id})", style="yellow")

    texto.append(sep, style="dim")

    texto.append("Sessão: ", style="dim")
    if estado.erro_estado:
        texto.append(f"{sim.aviso} desconhecida", style="yellow")
    elif estado.sessao_ativa:
        texto.append(f"{sim.ativo} ativa ({estado.sessao_nome})", style="green")
    else:
        texto.append(f"{sim.pendente} inativa", style="dim")
    return texto


def renderizar_cabecalho(con: Console, estado: EstadoLocal, subtela: str = "") -> None:
    """Limpa a tela (em TTY) e desenha cabeçalho compacto com barra de estado."""
    limpar_tela(con)
    sim = obter_simbolos(con)
    compacto = _modo_layout(con) == "cartoes"

    cabecalho = Text()
    cabecalho.append(M.TITULO, style="bold cyan")
    if subtela:
        cabecalho.append(f"  {sim.seta}  ", style="dim")
        cabecalho.append(subtela, style="bold")
    else:
        cabecalho.append(f" — {M.SUBTITULO}", style="dim")

    con.print(cabecalho)
    con.print(_texto_barra_estado(estado, sim, compacto=compacto))
    con.rule(style="dim")


# ---------------------------------------------------------------------------
# Helpers de UI: mensagens, erros, glossário, confirmações e etapas (D4, D5, D6, D9)
# ---------------------------------------------------------------------------


def _painel(
    con: Console,
    titulo: str,
    conteudo: Any,
    estilo: str = "cyan",
    *,
    expand: bool = False,
) -> None:
    con.print(Panel(conteudo, title=titulo, border_style=estilo, expand=expand))


def _erro(chave: str, **kwargs: str) -> str:
    falha, causa, acao = M.ERROS_COMUNS[chave]
    return M.ERRO.format(
        falha=falha.format(**kwargs), causa=causa.format(**kwargs), acao=acao.format(**kwargs)
    )


def _partes_erro_inesperado(caminho: Path) -> dict[str, str]:
    falha, causa, acao = M.ERROS_COMUNS["inesperado"]
    return {
        "falha": falha,
        "causa": causa,
        "acao": acao.format(log=M.encurtar_caminho(caminho)),
    }


def _log_erro(exc: Exception) -> Path:
    diretorio = config.config_dir() / "logs"
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / "tui-erros.log"
    with caminho.open("a", encoding="utf-8") as handle:
        handle.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return caminho


def _renderizar_erro(
    con: Console,
    falha: str,
    causa: str,
    acao: str,
    *,
    detalhes_tecnicos: str = "",
) -> None:
    sim = obter_simbolos(con)
    corpo = M.ERRO.format(falha=falha, causa=causa, acao=acao)
    _painel(con, f"{sim.erro} {M.ERRO_TITULO}", corpo, estilo="red")
    if detalhes_tecnicos.strip():
        _painel(
            con,
            M.DETALHES_TECNICOS_TITULO,
            Text(detalhes_tecnicos.strip(), style="dim"),
            estilo="dim",
        )


def _carregar_config(con: Console) -> config.Config:
    """Lê config.toml; arquivo inválido vira painel de erro, nunca exceção genérica.

    É a única forma de o usuário consertar o arquivo pela interface: o painel
    diz o caminho e a ação, e o menu volta a ser desenhado.
    """
    try:
        return config.load_config()
    except config.ConfigError as exc:
        raise _ErroEtapa(
            "O arquivo config.toml não pôde ser lido.",
            str(exc),
            f"Corrija ou mova {M.encurtar_caminho(config.config_path())} e volte ao menu. "
            "Nada foi alocado nem gravado nesta tentativa.",
            detalhes_tecnicos=str(exc),
        ) from exc


def _ler_estado_ou_erro() -> tuple[config.State | None, str]:
    """Lê state.json devolvendo o erro em vez de levantar `StateFileError`."""
    try:
        return config.load_state(), ""
    except config.StateFileError as exc:
        return None, str(exc)


def _partes_erro_estado(erro: str) -> tuple[str, str, str]:
    """Texto padrão do painel de state.json ilegível (falha, causa, ação)."""
    return (
        "O arquivo de estado local (state.json) está ilegível.",
        erro,
        "Confira o painel do Google Colab: pode haver uma GPU alocada sem registro. "
        "Depois rode `byom status` e corrija ou remova o state.json.",
    )


def _renderizar_erro_estado(con: Console, erro: str) -> None:
    """Painel padrão de state.json ilegível: pode haver GPU alocada sem registro."""
    _renderizar_erro(con, *_partes_erro_estado(erro), detalhes_tecnicos=erro)


def _renderizar_glossario(con: Console, termos: Sequence[str]) -> None:
    itens = M.glossario_para_tela(termos)
    if not itens:
        return
    sim = obter_simbolos(con)
    con.print()
    con.print("Legenda:", style="bold cyan")
    for termo, definicao in itens:
        linha = Text("  ")
        linha.append(f"{sim.bullet} {termo}: ", style="bold")
        linha.append(definicao, style="dim")
        con.print(linha)


def _comando_equivalente(acao_id: str, kwargs: dict[str, Any] | None = None) -> str:
    parts = [M.COMANDO_BASE, acao_id]
    if not isinstance(kwargs, dict):
        kwargs = {}
    aliases = {
        "profile_id": "profile",
        "gpu": "gpu",
        "opencode_config": "opencode-config",
        "dsh_config": "dsh-config",
    }
    for chave in ("gpu", "profile_id", "opencode_config", "dsh_config"):
        if kwargs.get(chave) is not None:
            parts += ["--" + aliases[chave], str(kwargs[chave])]
    if kwargs.get("force"):
        parts.append("--force")
    return " ".join(parts)


def _estilo_select() -> Any:
    """Evita fundo invertido (`reverse`) aplicado pelo prompt_toolkit ao item default."""
    import questionary

    return questionary.Style([("selected", "noreverse")])


def _confirmar(chave: str, *, pergunta: str | None = None, default: bool | None = None) -> bool:
    import questionary

    padrao = default if default is not None else chave in ("init", "configure")
    texto = pergunta or M.CONFIRMACOES[chave]
    sim_choice = questionary.Choice(title="Sim, continuar", value=True, shortcut_key="s")
    nao_choice = questionary.Choice(title="Não, voltar ao menu", value=False, shortcut_key="n")
    opcoes = [sim_choice, nao_choice] if padrao else [nao_choice, sim_choice]
    escolha = questionary.select(
        texto,
        choices=opcoes,
        use_shortcuts=True,
        use_jk_keys=False,
        style=_estilo_select(),
        instruction=M.TECLAS_CONFIRMACAO,
    ).ask()
    return bool(escolha)


def _opcoes_down() -> dict[str, Any]:
    if not _confirmar("down", default=False):
        raise KeyboardInterrupt
    return {}


@contextlib.contextmanager
def _guarda_transacao(holder: str) -> Iterator[None]:
    """Segura o lock exclusivo de `up`/`down` e traduz a recusa em painel.

    Dois `up` simultâneos alocariam duas GPUs; um `down` lento apagaria o state
    de uma sessão nova. O lock da CLI é o mesmo arquivo, então a TUI não abre
    uma janela própria de alocação concorrente.
    """
    try:
        with config.up_lock(holder):
            yield
    except config.ConfigError as exc:
        motivo = (
            "Já existe um `byom up` em andamento."
            if holder == "up"
            else "Um `up`/`down` está em andamento."
        )
        raise _ErroEtapa(
            motivo,
            str(exc),
            "Aguarde a outra execução terminar; esta tentativa não alterou nada.",
        ) from exc


@contextlib.contextmanager
def _guarda_up() -> Iterator[None]:
    """Compatibilidade: `up` usa `_guarda_transacao("up")`."""
    with _guarda_transacao("up"):
        yield



def _opcoes_up() -> dict[str, Any]:
    try:
        state = config.load_state()
    except config.StateFileError as exc:
        # `state.json` ilegível pode significar GPU alocada sem registro:
        # o painel de estado ilegível diz o que fazer, o de caminho não.
        raise _ErroEtapa(*_partes_erro_estado(str(exc)), detalhes_tecnicos=str(exc)) from exc
    chave = "up_force" if state is not None else "up"
    if not _confirmar(chave, default=False):
        raise KeyboardInterrupt
    return {"force": chave == "up_force", "opencode_config": None, "dsh_config": None}


def _mostrar_pre_acao(con: Console, acao_id: str) -> None:
    sim = obter_simbolos(con)
    if acao_id == "status":
        con.print(M.SOMENTE_LEITURA, style="dim")
        con.print()
        return
    itens = M.O_QUE_VAI_ACONTECER.get(acao_id, ())
    if not itens:
        return
    texto = "\n".join(f"{sim.bullet} {item}" for item in itens)
    titulo = M.ACOES[acao_id][0]
    estilo = "yellow" if acao_id in ("up", "down") else "cyan"
    _painel(con, f"O que vai acontecer — {titulo}", texto, estilo=estilo)


def _aguardar_enter(con: Console) -> None:
    if not is_interactive():
        return
    con.print()
    con.input(f"[dim]{M.ENTER_VOLTAR}[/dim]")


def _perguntar_proximo_passo(
    acao_id: str,
    estado: EstadoLocal,
    *,
    extras: Sequence[tuple[str, str]] = (),
) -> str:
    """Pergunta ao usuário qual o próximo passo após concluir uma tela/ação."""
    import questionary

    escolhas: list[Any] = []
    if acao_id == "up":
        escolhas.append(questionary.Choice(title=M.ACOES["chat"][0], value="acao:chat"))
    for valor, rotulo in extras:
        escolhas.append(questionary.Choice(title=rotulo, value=valor))

    prox_id, prox_rotulo = M.PROXIMO_ACAO[acao_id]
    # Ajusta sugestão se o estado indicar passo mais coerente
    if acao_id == "configure" and not estado.ambiente_configurado:
        prox_id, prox_rotulo = "init", M.ACOES["init"][0]
    elif acao_id == "status" and not estado.sessao_ativa:
        prox_id, prox_rotulo = (
            ("up", M.ACOES["up"][0])
            if estado.ambiente_configurado
            else ("init", M.ACOES["init"][0])
        )
    elif acao_id == "harness":
        prox_id, prox_rotulo = (
            ("status", M.ACOES["status"][0]) if estado.sessao_ativa else ("up", M.ACOES["up"][0])
        )
    escolhas.append(
        questionary.Choice(
            title=f"{prox_rotulo} (próximo passo recomendado)",
            value=f"acao:{prox_id}",
        )
    )
    escolhas.append(questionary.Choice(title="Voltar ao menu principal", value="menu"))

    escolha = questionary.select(
        M.SELECAO_PROXIMO,
        choices=escolhas,
        use_jk_keys=False,
        style=_estilo_select(),
        instruction=M.TECLAS_LISTA,
    ).ask()
    return str(escolha or "menu")


@dataclass
class EtapaExecucao:
    titulo: str
    estado: str = "pendente"  # pendente | andamento | ok | aviso | erro
    detalhe: str = ""
    inicio: float = 0.0
    duracao: float = 0.0
    dica: str = ""


class ExecutorEtapas:
    """Checklist ao vivo de etapas com tempo decorrido e captura de logs (D5, P11)."""

    def __init__(self, con: Console, titulos: Sequence[str]) -> None:
        self.con = con
        self.sim = obter_simbolos(con)
        self.etapas = [EtapaExecucao(titulo=t) for t in titulos]
        self.detalhes_tecnicos: list[str] = []
        self._inicio_total = time.monotonic()
        self._live: Live | None = None
        unicode_ok = "utf" in (getattr(self.con, "encoding", "") or "").lower()
        self._spinner_nome = "dots" if unicode_ok else "line"
        self._spinners: dict[int, Spinner] = {}

    def __enter__(self) -> ExecutorEtapas:
        # `Live.refresh` só desenha quando `Console.is_terminal` é verdadeiro, e
        # esse teste segue `sys.stdout` enquanto `Console.file` não estiver
        # fixado. As etapas redirecionam stdout para capturar a saída bruta;
        # congelar o terminal aqui evita que a animação pare no redirect.
        terminal = self.con.file
        self._arquivo_original = self.con._file
        self.con.file = terminal
        if self.con.is_terminal:
            # `get_renderable` refaz o desenho a cada atualização: o spinner anima
            # e o tempo decorrido avança mesmo sem nenhuma chamada nova.
            self._live = Live(
                console=self.con,
                get_renderable=self._render,
                refresh_per_second=8,
                transient=False,
            )
            self._live.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self._live is not None:
                self._live.update(self._render(), refresh=True)
                self._live.__exit__(exc_type, exc, tb)
                self._live = None
            else:
                self.con.print(self._render())
        finally:
            self.con._file = self._arquivo_original

    def registrar_log(self, texto: str) -> None:
        limpo = texto.strip()
        if limpo:
            self.detalhes_tecnicos.append(limpo)

    def iniciar(self, indice: int, detalhe: str = "", *, dica: str = "") -> None:
        etapa = self.etapas[indice]
        etapa.estado = "andamento"
        etapa.detalhe = detalhe
        etapa.dica = dica
        etapa.inicio = time.monotonic()
        self._atualizar()

    def concluir(self, indice: int, detalhe: str = "", *, aviso: bool = False) -> None:
        etapa = self.etapas[indice]
        etapa.estado = "aviso" if aviso else "ok"
        if detalhe:
            etapa.detalhe = detalhe
        if etapa.inicio:
            etapa.duracao = time.monotonic() - etapa.inicio
        self._atualizar()

    def falhar(self, indice: int, detalhe: str = "") -> None:
        etapa = self.etapas[indice]
        etapa.estado = "erro"
        if detalhe:
            etapa.detalhe = detalhe
        if etapa.inicio:
            etapa.duracao = time.monotonic() - etapa.inicio
        self._atualizar()

    def atualizar_detalhe(self, indice: int, detalhe: str) -> None:
        self.etapas[indice].detalhe = detalhe
        self._atualizar()

    def duracao_total(self) -> float:
        return time.monotonic() - self._inicio_total

    def _atualizar(self) -> None:
        if self._live is not None:
            self._live.update(self._render(), refresh=True)

    def _render(self) -> Table:
        tabela = Table.grid(padding=(0, 1))
        tabela.add_column(no_wrap=True)
        tabela.add_column()
        for indice, etapa in enumerate(self.etapas):
            icone: Text | Spinner
            if etapa.estado == "ok":
                icone = Text(self.sim.ok, style="green")
            elif etapa.estado == "aviso":
                icone = Text(self.sim.aviso, style="yellow")
            elif etapa.estado == "erro":
                icone = Text(self.sim.erro, style="red")
            elif etapa.estado == "andamento":
                icone = self._spinners.setdefault(indice, Spinner(self._spinner_nome, style="cyan"))
            else:
                icone = Text(self.sim.pendente, style="dim")

            linha = Text(etapa.titulo)
            if etapa.detalhe:
                estilo_det = "red" if etapa.estado == "erro" else "dim"
                linha.append(f" {self.sim.barra} {etapa.detalhe}", style=estilo_det)
            if etapa.estado == "andamento" and etapa.inicio:
                decorrido = time.monotonic() - etapa.inicio
                if decorrido >= 1.0:
                    linha.append(f" ({decorrido:.0f}s)", style="dim")
                if etapa.dica:
                    linha.append(f"\n  {etapa.dica}", style="dim italic")
            elif etapa.duracao >= 2.0:
                linha.append(f" ({etapa.duracao:.1f}s)", style="dim")
            tabela.add_row(icone, linha)

        decorrido_total = time.monotonic() - self._inicio_total
        rodape = Text(M.AVISO_CANCELAR, style="dim")
        if decorrido_total >= 2.0:
            rodape.append(f"  {self.sim.barra}  tempo: {decorrido_total:.1f}s", style="dim")
        tabela.add_row("", rodape)
        return tabela


def _extrair_versao_colab(texto: str) -> str:
    limpo = texto.strip()
    if not limpo:
        return "versão não informada"
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", limpo)
    if match:
        return f"versão {match.group(1)}"
    return limpo


# ---------------------------------------------------------------------------
# Dados estruturados de Perfis e GPUs (D7, D8, D9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerfilView:
    indice: int
    profile: config.Profile
    familia: str
    variante_curta: str
    tipo: str
    runtime_nome: str
    gpu_minima: str
    contexto_servido: int
    contexto_doc: int
    contexto_curto: str
    raciocinio: str
    pesos_gib: float
    visao_gib: float
    runtime_gib: float
    kv_gib: float
    kv_dtype: str
    total_gib: float
    cabe_gpu_atual: bool
    motivo_contexto: str
    descricao: str
    notas: tuple[str, ...]
    endpoint_aviso: str


def _tipo_modelo(model_id: str, runtime: str, tem_mmproj: bool) -> str:
    if runtime == "image":
        return "Imagem"
    if tem_mmproj or model_id in M.MODELOS_COM_VISAO:
        return "Chat/visão"
    return "Chat"


def listar_perfis_estruturados(estado: EstadoLocal) -> list[PerfilView]:
    """Constrói visão estruturada de todos os perfis do catálogo."""
    perfis = config.available_profiles()
    gpu_ref = estado.gpu_id
    itens_brutos: list[PerfilView] = []

    ordem_familias = list(M.FAMILIAS_MODELOS.values())

    def chave_ordenacao(p: config.Profile) -> tuple[int, str, int]:
        fam = M.nome_familia(p.catalog_model, p.name)
        idx_fam = ordem_familias.index(fam) if fam in ordem_familias else 99
        ctx = int(p.settings.get("max_model_len", 0))
        return (idx_fam, fam, ctx)

    for p in sorted(perfis, key=chave_ordenacao):
        entry = catalog.model_by_id(p.catalog_model) if p.catalog_model else None
        variant = (
            catalog.variant_by_id(p.catalog_model, p.catalog_variant)
            if p.catalog_model and p.catalog_variant
            else None
        )
        tem_mmproj = bool((p.worker or {}).get("mmproj_pattern"))
        tipo = _tipo_modelo(p.catalog_model, p.runtime, tem_mmproj)
        familia = M.nome_familia(p.catalog_model, p.name)
        variante_curta = M.rotulo_variante(
            p.catalog_model, p.catalog_variant, variant.label if variant else p.name
        )

        ctx_servido = int(p.settings.get("max_model_len", 0))
        ctx_doc = variant.context_doc if variant else ctx_servido
        if variant and p.catalog_model:
            # No contexto DESTE perfil, não no mínimo calibrado da variante.
            gpu_min = catalog.min_gpu_for(p.catalog_model, p.catalog_variant, ctx_servido)
        else:
            gpu_min = str(p.settings.get("gpu", "L4"))
        pesos = variant.memory_gib if variant else 0.0
        visao_gib = variant.vision_gib if variant else 0.0
        runtime_gib = catalog.RUNTIME_RESERVE_GIB
        kv_dtype = catalog.effective_kv_dtype(p.runtime, p.settings.get("kv_cache_dtype"))
        kv_gib = (
            0.0
            if p.runtime == "image"
            else catalog.kv_cache_gib(
                ctx_servido, kv_cache_dtype=p.settings.get("kv_cache_dtype"), runtime=p.runtime
            )
        )
        total_gib = (
            catalog.required_vram_gib(
                variant, ctx_servido, p.runtime, kv_cache_dtype=p.settings.get("kv_cache_dtype")
            )
            if variant
            else pesos + runtime_gib + kv_gib
        )

        fit_atual = (
            catalog.check_fit(
                p.catalog_model,
                p.catalog_variant,
                gpu_ref,
                ctx_servido,
                kv_cache_dtype=p.settings.get("kv_cache_dtype"),
            )
            if p.catalog_model and p.catalog_variant
            else None
        )
        cabe = bool(fit_atual.fits) if fit_atual else True

        esforco = p.settings.get("reasoning_effort") or (entry.reasoning_effort if entry else None)
        tem_think = bool(entry and entry.thinking is not None)
        raciocinio = M.traduzir_raciocinio(
            esforco, tem_thinking_prompt=tem_think, runtime=p.runtime
        )

        motivo = ""
        if p.catalog_model:
            try:
                ref = catalog.profile_context(p.id)
                motivo = M.motivo_contexto_perfil(p.id, ref.reason)
            except ValueError:
                motivo = ""

        notas = M.NOTAS_MODELOS.get(p.catalog_model, ())
        endpoint_aviso = (
            "/v1/images/generations" if p.runtime == "image" else "/v1/chat/completions"
        )

        itens_brutos.append(
            PerfilView(
                indice=0,
                profile=p,
                familia=familia,
                variante_curta=variante_curta,
                tipo=tipo,
                runtime_nome=M.NOMES_RUNTIMES.get(p.runtime, p.runtime),
                gpu_minima=gpu_min,
                contexto_servido=ctx_servido,
                contexto_doc=ctx_doc,
                contexto_curto=M.formatar_contexto(ctx_servido, p.runtime),
                raciocinio=raciocinio,
                pesos_gib=pesos,
                visao_gib=visao_gib,
                runtime_gib=runtime_gib,
                kv_gib=kv_gib,
                kv_dtype=kv_dtype,
                total_gib=total_gib,
                cabe_gpu_atual=cabe,
                motivo_contexto=motivo,
                descricao=M.descricao_perfil(p.id, p.description),
                notas=notas,
                endpoint_aviso=endpoint_aviso,
            )
        )

    return [
        PerfilView(**{**item.__dict__, "indice": idx}) for idx, item in enumerate(itens_brutos, 1)
    ]


@dataclass(frozen=True)
class GpuView:
    indice: int
    gpu: catalog.Gpu
    compativeis: int
    total_perfis: int
    nota: str


def listar_gpus_estruturadas() -> list[GpuView]:
    perfis = config.available_profiles()
    total = len(perfis)
    views: list[GpuView] = []
    for idx, gpu in enumerate(catalog.GPUS, 1):
        compativeis = sum(
            1
            for p in perfis
            if p.catalog_model
            and catalog.check_fit(
                p.catalog_model,
                p.catalog_variant,
                gpu.id,
                int(p.settings["max_model_len"]),
            ).fits
        )
        views.append(
            GpuView(
                indice=idx,
                gpu=gpu,
                compativeis=compativeis,
                total_perfis=total,
                nota=M.NOTAS_GPUS.get(gpu.id, ""),
            )
        )
    return views


# ---------------------------------------------------------------------------
# Renderizadores estruturados (D7, D8, D9, D10, D11)
# ---------------------------------------------------------------------------


def renderizar_perfis(
    con: Console,
    estado: EstadoLocal,
    *,
    itens: Sequence[PerfilView] | None = None,
) -> list[PerfilView]:
    """Renderiza os perfis em tabela (completa/compacta) ou cartões conforme a largura."""
    lista = list(itens) if itens is not None else listar_perfis_estruturados(estado)
    sim = obter_simbolos(con)
    modo = _modo_layout(con)

    if _altura_terminal(con) <= 26:
        con.print(
            M.DICA_PAGINACAO.format(largura=_largura_terminal(con), altura=_altura_terminal(con)),
            style="dim",
        )

    titulo_lista = f"Perfis de modelo disponíveis ({len(lista)})"
    con.print(titulo_lista, style="bold cyan")
    if not lista:
        con.print(_erro("vazio", origem="catálogo de perfis"), style="yellow")
        return lista

    gpu_alvo = estado.gpu_id

    if modo == "cartoes":
        familia_atual = ""
        for item in lista:
            if item.familia != familia_atual:
                familia_atual = item.familia
                con.print(f"\n[bold cyan]{familia_atual}[/bold cyan]")
            marca_atual = f" {sim.estrela} em uso" if item.profile.id == estado.perfil_id else ""
            cabe_txt = (
                f"{sim.ok} cabe em {gpu_alvo}"
                if item.cabe_gpu_atual
                else f"{sim.erro} exige {item.gpu_minima}"
            )
            corpo = (
                f"Variante: {item.variante_curta}  |  Tipo: {item.tipo}\n"
                f"GPU mínima: {item.gpu_minima}  |  Contexto: {item.contexto_curto}\n"
                f"Raciocínio: {item.raciocinio}  |  {cabe_txt}"
            )
            _painel(
                con,
                f"{item.indice}. {item.familia} — {item.variante_curta}{marca_atual}",
                corpo,
                estilo="green" if item.profile.id == estado.perfil_id else "cyan",
                expand=True,
            )
    else:
        tabela = Table(
            box=box.SIMPLE,
            padding=(0, 1),
            header_style="bold cyan",
            show_edge=False,
            expand=False,
        )
        tabela.add_column("#", justify="right", style="dim", min_width=2, no_wrap=True)
        tabela.add_column("Modelo", no_wrap=True)
        tabela.add_column("Variante", no_wrap=True)
        tabela.add_column("Tipo", no_wrap=True)
        tabela.add_column("GPU mín.", no_wrap=True)
        tabela.add_column("Contexto", no_wrap=True)
        if modo == "completo":
            tabela.add_column("Raciocínio", no_wrap=True)
            cabe_header = f"Em {gpu_alvo}" if _largura_terminal(con) >= 110 else "Cabe"
            tabela.add_column(cabe_header, justify="center", no_wrap=True)

        familia_atual = ""
        apertado_100 = modo == "completo" and _largura_terminal(con) < 110
        for idx, item in enumerate(lista):
            if idx > 0 and item.familia != familia_atual:
                tabela.add_section()
            familia_atual = item.familia

            if modo == "compacto" or apertado_100:
                tipo_cell = "Visão" if item.tipo == "Chat/visão" else item.tipo
                ctx_cell = item.contexto_curto
                var_cell = (
                    item.variante_curta.replace(" (GGUF)", " GGUF")
                    .replace(" (AWQ)", " AWQ")
                    .replace(" (FP8)", " FP8")
                    .replace(" • ", " ")
                )
            else:
                tipo_cell = item.tipo
                ctx_cell = item.contexto_curto
                var_cell = item.variante_curta

            modelo_cell = Text(item.familia)
            if item.profile.id == estado.perfil_id:
                modelo_cell.append(f" {sim.estrela}", style="green")

            linha: list[Any] = [
                str(item.indice),
                modelo_cell,
                var_cell,
                tipo_cell,
                item.gpu_minima,
                ctx_cell,
            ]
            if modo == "completo":
                linha.append(item.raciocinio)
                cabe_text = (
                    Text(sim.ok, style="green")
                    if item.cabe_gpu_atual
                    else Text(sim.erro, style="red")
                )
                linha.append(cabe_text)
            tabela.add_row(*linha)
        con.print(tabela)

    termos = ["Contexto", "GGUF", "AWQ"]
    if any("visão" in item.tipo for item in lista):
        termos.append("Visão")
    if modo == "completo" or modo == "cartoes":
        termos.append("Raciocínio")
    _renderizar_glossario(con, termos)
    return lista


def renderizar_detalhe_perfil(con: Console, item: PerfilView, estado: EstadoLocal) -> None:
    """Exibe os detalhes completos de um perfil em português simples."""
    sim = obter_simbolos(con)
    p = item.profile

    linhas: list[Text] = []

    def add(rotulo: str, valor: str, estilo_valor: str = "") -> None:
        t = Text()
        t.append(f"{rotulo}: ", style="bold cyan")
        t.append(valor, style=estilo_valor)
        linhas.append(t)

    add("Modelo", f"{item.familia} ({item.variante_curta})")
    add("Tipo", f"{item.tipo}  {sim.barra}  Motor: {item.runtime_nome}")
    add("Descrição", item.descricao)

    if p.runtime == "image":
        add(
            "Memória estimada",
            f"{M.formatar_gib(item.total_gib, aproximado=True)} "
            f"({M.formatar_gib(item.pesos_gib)} para os pesos do modelo + "
            f"{M.formatar_gib(item.runtime_gib)} para o motor; "
            "modelos de imagem não usam cache KV).",
        )
        add("Contexto", f"Não se aplica (atendido em POST {item.endpoint_aviso}).")
    else:
        visao = (
            f" + {M.formatar_gib(item.visao_gib)} para o projetor de visão"
            if item.visao_gib
            else ""
        )
        add(
            "Memória estimada",
            f"{M.formatar_gib(item.total_gib, aproximado=True)} "
            f"({M.formatar_gib(item.pesos_gib)} para os pesos{visao} + "
            f"{M.formatar_gib(item.runtime_gib)} para o motor + "
            f"{M.formatar_gib(item.kv_gib)} para o cache KV).",
        )
        ctx_serv = M.formatar_contexto_detalhe(item.contexto_servido, p.runtime)
        ctx_max = M.formatar_contexto_detalhe(item.contexto_doc, p.runtime)
        add("Contexto servido", f"{ctx_serv}  {sim.barra}  Máximo do modelo: {ctx_max}")
        if item.motivo_contexto:
            add("Por que esse contexto", item.motivo_contexto)
        add("Raciocínio padrão", item.raciocinio)

    if item.cabe_gpu_atual:
        add(
            "Compatibilidade",
            f"{sim.ok} Cabe na sua GPU configurada ({estado.gpu_id}) "
            f"• GPU mínima: {item.gpu_minima}",
            "green",
        )
    else:
        add(
            "Compatibilidade",
            f"{sim.erro} Exige pelo menos {item.gpu_minima}; não cabe em {estado.gpu_id}",
            "red",
        )

    for nota in item.notas:
        add("Observação", nota)

    linhas.append(Text())
    add("ID do perfil", p.id, "dim")
    if p.catalog_model:
        add("Catálogo", f"{p.catalog_model}:{p.catalog_variant}", "dim")
    add("OpenCode", p.opencode_model, "dim")
    add("Equivalente na CLI", _comando_equivalente("configure", {"profile_id": p.id}), "dim")

    corpo = Text("\n").join(linhas)
    _painel(
        con,
        f"Detalhe do perfil — {item.familia} ({item.variante_curta})",
        corpo,
        estilo="cyan",
        expand=True,
    )

    termos = (
        ["GGUF"]
        if "GGUF" in item.variante_curta
        else ["AWQ"]
        if "AWQ" in item.variante_curta
        else []
    )
    if p.runtime != "image":
        termos.extend(["Contexto", "Cache KV", "Raciocínio"])
    if "visão" in item.tipo:
        termos.append("Visão")
    _renderizar_glossario(con, termos)


def renderizar_gpus(con: Console, estado: EstadoLocal) -> list[GpuView]:
    """Renderiza tabela real de GPUs com no máximo 5 colunas (ou cartões em <80)."""
    views = listar_gpus_estruturadas()
    sim = obter_simbolos(con)
    modo = _modo_layout(con)

    con.print(f"GPUs disponíveis no catálogo ({len(views)})", style="bold cyan")

    if modo == "cartoes":
        for gv in views:
            g = gv.gpu
            bf16_txt = "Sim" if g.bf16 else "Não"
            vram_txt = M.formatar_gib(g.vram_gib, aproximado=g.vram_approximate)
            marca = f" {sim.estrela} configurada" if g.id == estado.gpu_id else ""
            corpo = (
                f"VRAM: {vram_txt}  |  Suporta bf16: {bf16_txt}\n"
                f"Perfis compatíveis: {gv.compativeis} de {gv.total_perfis}"
            )
            if gv.nota:
                corpo += f"\n{gv.nota}"
            _painel(
                con,
                f"{gv.indice}. {g.id} ({g.label}){marca}",
                corpo,
                estilo="green" if g.id == estado.gpu_id else "cyan",
                expand=True,
            )
    else:
        tabela = Table(
            box=box.SIMPLE,
            padding=(0, 1),
            header_style="bold cyan",
            show_edge=False,
            expand=False,
        )
        tabela.add_column("GPU", no_wrap=True)
        tabela.add_column("VRAM", justify="right", no_wrap=True)
        tabela.add_column("Suporta bf16", justify="center", no_wrap=True)
        tabela.add_column("Perfis compatíveis", no_wrap=True)
        if modo == "completo":
            tabela.add_column("Observação")

        for gv in views:
            g = gv.gpu
            nome_gpu = Text(g.label if g.label == g.id else f"{g.id} ({g.label})")
            if g.id == estado.gpu_id:
                nome_gpu.append(f" {sim.estrela}", style="green")
            vram_txt = M.formatar_gib(g.vram_gib, aproximado=g.vram_approximate)
            bf16_cell = Text("Sim", style="green") if g.bf16 else Text("Não", style="yellow")
            comp_txt = f"{gv.compativeis} de {gv.total_perfis}"
            row: list[Any] = [nome_gpu, vram_txt, bf16_cell, comp_txt]
            if modo == "completo":
                row.append(gv.nota or "—")
            tabela.add_row(*row)
        con.print(tabela)

    _renderizar_glossario(con, ["VRAM (~)", "bf16"])
    data_ptbr = M.formatar_data_ptbr(catalog.DOCS_CHECKED_ON)
    con.print()
    con.print(M.FONTE_GPUS_RESUMO.format(data=data_ptbr), style="dim")
    return views


def renderizar_modelos_por_gpu(
    con: Console, gpu_id: str, estado: EstadoLocal
) -> tuple[catalog.CompatibleSelection, dict[str, PerfilView]]:
    """Renderiza variantes compatíveis com uma GPU usando Rich em português."""
    gpu = catalog.gpu_by_id(gpu_id)
    selection = catalog.compatible_variants(gpu.id)
    modo = _modo_layout(con)

    perfis_por_id = {pv.profile.id: pv for pv in listar_perfis_estruturados(estado)}
    vram_txt = M.formatar_gib(gpu.vram_gib, aproximado=gpu.vram_approximate)
    con.print(f"Modelos compatíveis com {gpu.id} ({gpu.label} — {vram_txt})", style="bold cyan")

    if modo == "cartoes":
        for idx, row in enumerate(selection.rows, 1):
            fam = M.nome_familia(row.model_id, row.model_id)
            var = M.rotulo_variante(row.model_id, row.variant_id, row.variant_label)
            ctx = M.formatar_contexto(row.context_served, row.runtime)
            mem = M.formatar_gib(row.required_gib, aproximado=True)
            perfil_txt = (
                f"{obter_simbolos(con).ok} disponível"
                if row.profile_id in perfis_por_id
                else "sem perfil pronto"
            )
            _painel(
                con,
                f"{idx}. {fam} — {var}",
                f"Memória: {mem}  |  Contexto: {ctx}\nPerfil: {perfil_txt}",
                estilo="cyan",
                expand=True,
            )
    else:
        tabela = Table(
            box=box.SIMPLE,
            padding=(0, 1),
            header_style="bold cyan",
            show_edge=False,
            expand=False,
        )
        tabela.add_column("#", justify="right", style="dim", min_width=2, no_wrap=True)
        tabela.add_column("Modelo", no_wrap=True)
        tabela.add_column("Variante", no_wrap=True)
        if modo == "completo":
            tabela.add_column("Tipo", no_wrap=True)
        tabela.add_column("Memória", justify="right", no_wrap=True)
        tabela.add_column("Contexto", no_wrap=True)
        tabela.add_column("Perfil", no_wrap=True)

        for idx, row in enumerate(selection.rows, 1):
            fam = M.nome_familia(row.model_id, row.model_id)
            var = M.rotulo_variante(row.model_id, row.variant_id, row.variant_label)
            if modo == "compacto":
                var = (
                    var.replace(" (GGUF)", " GGUF")
                    .replace(" (AWQ)", " AWQ")
                    .replace(" (FP8)", " FP8")
                    .replace(" • ", " ")
                )
            ctx = M.formatar_contexto(row.context_served, row.runtime)
            mem = M.formatar_gib(row.required_gib, aproximado=True)
            sim = obter_simbolos(con)
            if modo == "compacto":
                perfil_txt = f"{sim.ok} sim" if row.profile_id in perfis_por_id else "não"
            else:
                perfil_txt = (
                    f"{sim.ok} disponível" if row.profile_id in perfis_por_id else "sem perfil"
                )
            cells: list[Any] = [str(idx), fam, var]
            if modo == "completo":
                entry = catalog.model_by_id(row.model_id)
                tipo = _tipo_modelo(row.model_id, entry.runtime, entry.id in M.MODELOS_COM_VISAO)
                cells.append(tipo)
            cells.extend([mem, ctx, perfil_txt])
            tabela.add_row(*cells)
        con.print(tabela)

    con.print(
        f"Resumo: {M.formatar_numero(selection.compatible_count)} variante(s) compatível(eis); "
        f"{M.formatar_numero(selection.dropped_for_memory)} descartada(s) por falta de VRAM.",
        style="dim",
    )
    _renderizar_glossario(con, ["VRAM (~)", "Contexto", "GGUF", "AWQ"])
    return selection, perfis_por_id


# ---------------------------------------------------------------------------
# Fluxos interativos de cada ação (P4, P5, P6, P10, P11, D4, D5, D6)
# ---------------------------------------------------------------------------


def _mostrar_resultado(
    con: Console,
    acao_id: str,
    *,
    titulo_ok: str,
    linhas_detalhe: Sequence[str] = (),
    duracao_s: float = 0.0,
    kwargs_cli: dict[str, Any] | None = None,
    detalhes_tecnicos: str = "",
) -> None:
    sim = obter_simbolos(con)
    con.print()
    cab = Text()
    cab.append(f"{sim.ok} {titulo_ok}", style="bold green")
    if duracao_s >= 2.0:
        cab.append(f"  ({duracao_s:.1f}s)", style="dim")
    con.print(cab)
    for linha in linhas_detalhe:
        con.print(f"  {sim.bullet} {linha}")
    if detalhes_tecnicos.strip():
        con.print()
        _painel(
            con,
            M.DETALHES_TECNICOS_TITULO,
            Text(detalhes_tecnicos.strip(), style="dim"),
            estilo="dim",
        )
    con.print()
    if acao_id == "harness":
        con.print("Dica na CLI: byom up (sincroniza OpenCode/dsh ao iniciar)", style="dim")
    else:
        con.print(
            M.EQUIVALENTE_CLI.format(comando=_comando_equivalente(acao_id, kwargs_cli)),
            style="dim",
        )


def _escolher_gpu(con: Console, estado: EstadoLocal) -> str:
    import questionary

    views = listar_gpus_estruturadas()
    if not views:
        raise ValueError(_erro("vazio", origem="catálogo de GPUs"))
    escolhas = [
        questionary.Choice(
            title=(
                f"{gv.gpu.id} ({gv.gpu.label}) — "
                f"{M.formatar_gib(gv.gpu.vram_gib, aproximado=gv.gpu.vram_approximate)} • "
                f"{gv.compativeis} de {gv.total_perfis} perfis"
            ),
            value=gv.gpu.id,
        )
        for gv in views
    ]
    escolhas.append(questionary.Choice(title="Voltar ao menu principal", value="__voltar__"))
    escolha = questionary.select(
        M.SELECAO_GPU,
        choices=escolhas,
        use_search_filter=True,
        use_jk_keys=False,
        style=_estilo_select(),
        instruction=M.TECLAS_SELECAO,
    ).ask()
    if escolha is None:
        raise KeyboardInterrupt
    return str(escolha)


def _escolher_perfil(
    con: Console, estado: EstadoLocal, *, itens: Sequence[PerfilView] | None = None
) -> PerfilView:
    import questionary

    lista = list(itens) if itens is not None else listar_perfis_estruturados(estado)
    if not lista:
        raise ValueError(_erro("vazio", origem="catálogo de perfis"))

    escolhas = []
    for item in lista:
        atual = " • atual" if item.profile.id == estado.perfil_id else ""
        escolhas.append(
            questionary.Choice(
                title=(
                    f"{item.indice:2d}. {item.familia} — {item.variante_curta} "
                    f"({item.gpu_minima}, {item.contexto_curto}){atual}"
                ),
                value=item,
            )
        )
    escolhas.append(questionary.Choice(title="Voltar ao menu principal", value="__voltar__"))

    escolha = questionary.select(
        M.SELECAO_PERFIL,
        choices=escolhas,
        use_search_filter=True,
        use_jk_keys=False,
        style=_estilo_select(),
        instruction=M.TECLAS_SELECAO,
    ).ask()
    if escolha is None or escolha == "__voltar__":
        raise KeyboardInterrupt
    return escolha


def _aplicar_perfil_interativo(
    con: Console,
    item: PerfilView,
    *,
    pedir_confirmacao: bool = True,
    gpu_id: str | None = None,
) -> tuple[bool, float]:
    """Aplica um perfil em config.toml; com `gpu_id`, fixa essa GPU por último.

    `apply_profile` copia `profile.settings`, que inclui `gpu`/`gpu_high_mem`.
    Quando o usuário escolheu a GPU no caminho GPU + modelo, essa escolha
    sobrescreve o que o perfil declara, e o par precisa passar por
    `config.check_compatibility` antes de ser gravado.
    """
    p = item.profile
    gpu = catalog.gpu_by_id(gpu_id) if gpu_id else None
    alvo = f"'{item.familia} — {item.variante_curta}'"
    if pedir_confirmacao:
        _mostrar_pre_acao(con, "configure")
        origem = f"{gpu.id} + {alvo}" if gpu else alvo
        if not _confirmar("configure", pergunta=f"Aplicar {origem} em config.toml?", default=True):
            return False, 0.0

    inicio = time.monotonic()
    state, erro_estado = _ler_estado_ou_erro()
    if erro_estado:
        # Pode haver GPU alocada sem registro: nada é gravado antes de o
        # usuário conferir o painel do Colab.
        _renderizar_erro_estado(con, erro_estado)
        return False, time.monotonic() - inicio
    if state is not None:
        _renderizar_erro(
            con,
            f"Existe uma sessão ativa ({state.session}).",
            "Não é permitido trocar de perfil enquanto uma VM está alocada no Colab.",
            "Encerre a sessão ativa antes de selecionar outro perfil.",
        )
        return False, time.monotonic() - inicio

    try:
        cfg = _carregar_config(con)
    except _ErroEtapa as exc:
        _renderizar_erro(
            con, exc.falha, exc.causa, exc.acao, detalhes_tecnicos=exc.detalhes_tecnicos
        )
        return False, time.monotonic() - inicio
    config.apply_profile(cfg, p.id)
    fit: catalog.CompatibilityResult | None = None
    if gpu is not None:
        cfg.gpu, cfg.gpu_high_mem = catalog.colab_settings_for_gpu(gpu.id)
        try:
            fit = config.check_compatibility(cfg)
        except ValueError as exc:
            _renderizar_erro(
                con,
                f"{gpu.id} + '{item.familia} — {item.variante_curta}' não é um par válido.",
                str(exc),
                "Escolha outro modelo ou outra GPU.",
                detalhes_tecnicos=str(exc),
            )
            return False, time.monotonic() - inicio
        if not fit.fits:
            _renderizar_erro(
                con,
                f"'{item.familia} — {item.variante_curta}' não cabe em {gpu.id}.",
                fit.reason,
                "Escolha um modelo compatível com esta GPU ou volte e escolha outra GPU.",
            )
            return False, time.monotonic() - inicio

    caminho = config.save_config(cfg)
    duracao = time.monotonic() - inicio
    memoria = "memória estendida" if cfg.gpu_high_mem else "memória padrão"

    detalhes = [
        f"Perfil ativo: {item.familia} — {item.variante_curta}",
        f"GPU padrão: {cfg.gpu} ({memoria})",
        f"Contexto servido: {M.formatar_contexto_detalhe(cfg.max_model_len, cfg.runtime)}",
    ]
    if fit is not None:
        detalhes.append(
            f"Compatível: {M.formatar_gib(fit.required_gib, aproximado=True)} necessários "
            f"de {M.formatar_gib(fit.available_gib)} disponíveis em {fit.gpu_id}"
        )
    detalhes += [
        f"Arquivo salvo: {M.encurtar_caminho(caminho)}",
        f"No OpenCode, use o modelo: {p.opencode_model}",
    ]
    _mostrar_resultado(
        con,
        "configure",
        titulo_ok="Perfil selecionado",
        linhas_detalhe=detalhes,
        duracao_s=duracao,
        kwargs_cli={"profile_id": p.id},
    )
    return True, duracao


def _fluxo_detalhe_perfil(con: Console, item: PerfilView, estado: EstadoLocal) -> str:
    """Abre tela de detalhe do perfil e retorna 'lista', 'menu' ou 'acao:<id>'."""
    import questionary

    renderizar_cabecalho(con, estado, subtela="Detalhe do perfil")
    renderizar_detalhe_perfil(con, item, estado)
    con.print()
    escolha = questionary.select(
        "O que você quer fazer com este perfil?",
        choices=[
            questionary.Choice(title="Usar este perfil", value="usar"),
            questionary.Choice(title="Voltar para a lista", value="lista"),
            questionary.Choice(title="Voltar ao menu principal", value="menu"),
        ],
        use_jk_keys=False,
        style=_estilo_select(),
        instruction=M.TECLAS_DETALHE,
    ).ask()

    if escolha is None or escolha == "menu":
        return "menu"
    if escolha == "lista":
        return "lista"
    if escolha == "usar":
        ok, _ = _aplicar_perfil_interativo(con, item, pedir_confirmacao=True)
        if not ok:
            _aguardar_enter(con)
            return "lista"
        novo_estado = ler_estado_local()
        return _perguntar_proximo_passo(
            "configure",
            novo_estado,
            extras=[("acao:harness", M.ACOES["harness"][0])],
        )
    return "menu"


def _executar_init(con: Console, cli: Any, estado: EstadoLocal) -> str:
    renderizar_cabecalho(con, estado, subtela=M.ACOES["init"][0])
    _mostrar_pre_acao(con, "init")
    if not _confirmar("init", default=True):
        return "menu"

    etapas = [
        "Verificando cliente ssh",
        "Verificando CLI do Colab",
        "Verificando login no Colab",
        "Preparando chave SSH",
        "Salvando configuração local",
    ]
    with ExecutorEtapas(con, etapas) as exec_etapas:
        # 1. ssh
        exec_etapas.iniciar(0)
        ssh_path = cli.shutil.which("ssh")
        if ssh_path is None:
            exec_etapas.falhar(0, "não encontrado no PATH")
            exec_etapas.registrar_log("shutil.which('ssh') -> None")
            det = "\n".join(exec_etapas.detalhes_tecnicos)
            raise _ErroEtapa(
                "O cliente `ssh` não foi encontrado no PATH.",
                "O OpenSSH client não está instalado ou não está disponível no PATH do sistema.",
                "Instale o cliente OpenSSH (ex.: `sudo apt install openssh-client`) "
                "e tente novamente.",
                detalhes_tecnicos=det,
            )
        exec_etapas.concluir(0, M.encurtar_caminho(ssh_path))

        # 2. colab version
        exec_etapas.iniciar(1)
        try:
            versao_bruta = cli.colab.version()
            exec_etapas.registrar_log(f"colab --version -> {versao_bruta}")
        except cli.colab.ColabNotInstalled as exc:
            exec_etapas.falhar(1, "não instalada")
            exec_etapas.registrar_log(str(exc))
            raise _ErroEtapa(
                "A CLI oficial do Colab (`colab`) não está instalada.",
                "O pacote `google-colab-cli` não foi encontrado no PATH.",
                "Instale com `uv tool install google-colab-cli` e tente novamente.",
                detalhes_tecnicos="\n".join(exec_etapas.detalhes_tecnicos),
            ) from exc
        except cli.colab.ColabCliError as exc:
            exec_etapas.falhar(1, "falha ao consultar")
            exec_etapas.registrar_log(str(exc))
            raise _ErroEtapa(
                "Não foi possível consultar a CLI do Colab.",
                "A execução de `colab --version` retornou erro.",
                "Verifique a instalação do `google-colab-cli`.",
                detalhes_tecnicos="\n".join(exec_etapas.detalhes_tecnicos),
            ) from exc
        exec_etapas.concluir(1, _extrair_versao_colab(versao_bruta))

        # 3. colab login
        exec_etapas.iniciar(2)
        proc = cli.colab.run(["sessions"], check=False)
        for texto in (getattr(proc, "stdout", ""), getattr(proc, "stderr", "")):
            if texto:
                exec_etapas.registrar_log(texto)
        if proc.returncode != 0:
            exec_etapas.falhar(2, "não autenticado")
            raise _ErroEtapa(
                "A CLI do Colab ainda não está autenticada.",
                "Nenhuma credencial válida foi encontrada ao executar `colab sessions`.",
                "Execute `colab sessions` no terminal, conclua o login no navegador "
                "e tente novamente.",
                detalhes_tecnicos="\n".join(exec_etapas.detalhes_tecnicos),
            )
        exec_etapas.concluir(2, "autenticado")

        # 4. ssh key
        exec_etapas.iniciar(3)
        try:
            cfg = _carregar_config(con)
        except _ErroEtapa:
            exec_etapas.falhar(3, "config.toml inválido")
            raise
        criou_chave = False
        if not cfg.ssh_key_path.exists():
            cfg.ssh_key_path.parent.mkdir(parents=True, exist_ok=True)
            res = cli.subprocess.run(
                [
                    "ssh-keygen",
                    "-t",
                    "ed25519",
                    "-N",
                    "",
                    "-C",
                    "byom",
                    "-f",
                    str(cfg.ssh_key_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for texto in (getattr(res, "stdout", ""), getattr(res, "stderr", "")):
                if texto:
                    exec_etapas.registrar_log(texto)
            if cfg.ssh_key_path.exists():
                cfg.ssh_key_path.chmod(0o600)
            criou_chave = True
            exec_etapas.concluir(3, f"criada em {M.encurtar_caminho(cfg.ssh_key_path)}")
        else:
            problema = cli.remote.validate_ssh_key(cfg.ssh_key_path)
            if problema:
                exec_etapas.falhar(3, "chave inválida")
                raise _ErroEtapa(
                    "A chave SSH existente não é válida.",
                    problema[0].upper() + problema[1:] + ".",
                    f"Renomeie ou apague {M.encurtar_caminho(cfg.ssh_key_path)} e rode "
                    "'Preparar ambiente local' de novo para gerar uma chave nova.",
                    detalhes_tecnicos="\n".join(exec_etapas.detalhes_tecnicos),
                )
            exec_etapas.registrar_log(f"Chave existente reutilizada: {cfg.ssh_key_path}")
            exec_etapas.concluir(3, f"já existente em {M.encurtar_caminho(cfg.ssh_key_path)}")

        # 5. save config
        exec_etapas.iniciar(4)
        caminho_cfg = config.save_config(cfg)
        exec_etapas.concluir(4, M.encurtar_caminho(caminho_cfg))

    duracao = exec_etapas.duracao_total()
    detalhes = [
        f"Chave SSH {'criada' if criou_chave else 'pronta'}: "
        f"{M.encurtar_caminho(cfg.ssh_key_path)}",
        f"Configuração salva em: {M.encurtar_caminho(caminho_cfg)}",
    ]
    _mostrar_resultado(
        con,
        "init",
        titulo_ok="Ambiente local pronto",
        linhas_detalhe=detalhes,
        duracao_s=duracao,
    )
    novo_estado = ler_estado_local()
    return _perguntar_proximo_passo("init", novo_estado)


def _troca_bloqueada_por_sessao(con: Console) -> str | None:
    """Devolve o destino quando a troca é bloqueada; None quando é permitida.

    Lê o `state.json` de verdade: a barra de estado marca `sessao_ativa=False`
    quando o arquivo está ilegível, e liberar a troca nesse caso esconderia
    uma VM possivelmente alocada.
    """
    state, erro_estado = _ler_estado_ou_erro()
    if erro_estado:
        _renderizar_erro_estado(con, erro_estado)
        return "menu"
    if state is None:
        return None
    _painel(
        con,
        "Sessão ativa encontrada",
        f"A sessão '{state.session}' está ativa no momento.\n"
        "É necessário encerrá-la antes de trocar o perfil de modelo.",
        estilo="yellow",
    )
    if _confirmar(
        "down", pergunta="Deseja ir para 'Encerrar sessão no Colab' agora?", default=False
    ):
        return "acao:down"
    return "menu"


def _executar_configure(con: Console, estado: EstadoLocal) -> str:
    renderizar_cabecalho(con, estado, subtela=M.ACOES["configure"][0])
    bloqueio = _troca_bloqueada_por_sessao(con)
    if bloqueio:
        return bloqueio

    while True:
        renderizar_perfis(con, estado)
        con.print()
        try:
            selecionado = _escolher_perfil(con, estado)
        except KeyboardInterrupt:
            return "menu"
        destino = _fluxo_detalhe_perfil(con, selecionado, estado)
        if destino != "lista":
            return destino
        # "Voltar para a lista" volta para ESTA lista, não para o menu principal.
        estado = ler_estado_local()
        renderizar_cabecalho(con, estado, subtela=M.ACOES["configure"][0])


def _renderizar_par_gpu_modelo(con: Console, gpu: catalog.Gpu, item: PerfilView) -> None:
    """Mostra o par que será gravado, incluindo o mapeamento do catálogo para o config.toml."""
    sim = obter_simbolos(con)
    gpu_cfg, high_mem = catalog.colab_settings_for_gpu(gpu.id)
    memoria = f"pesos {M.formatar_gib(item.pesos_gib)}"
    if item.visao_gib:
        memoria += f" + visão {M.formatar_gib(item.visao_gib)}"
    memoria += f" + motor {M.formatar_gib(item.runtime_gib)}"
    if item.profile.runtime != "image":
        memoria += f" + cache KV {M.formatar_gib(item.kv_gib)} ({item.kv_dtype})"

    corpo = Text()
    corpo.append("GPU: ", style="bold cyan")
    corpo.append(
        f"{gpu.id} ({gpu.label}) {sim.bullet} "
        f"{M.formatar_gib(gpu.vram_gib, aproximado=gpu.vram_approximate)}\n"
    )
    corpo.append("Modelo: ", style="bold cyan")
    corpo.append(f"{item.familia} — {item.variante_curta}\n")
    corpo.append("Contexto servido: ", style="bold cyan")
    corpo.append(f"{M.formatar_contexto_detalhe(item.contexto_servido, item.profile.runtime)}\n")
    corpo.append("Memória estimada: ", style="bold cyan")
    corpo.append(f"{M.formatar_gib(item.total_gib, aproximado=True)} ({memoria})\n")
    corpo.append("config.toml: ", style="bold cyan")
    corpo.append(f'gpu = "{gpu_cfg}"  {sim.barra}  gpu_high_mem = {str(high_mem).lower()}')
    _painel(con, "GPU + modelo escolhidos", corpo, estilo="cyan", expand=True)


def _escolher_modelo_para_gpu(con: Console, estado: EstadoLocal, gpu_id: str) -> str:
    """Lista os modelos compatíveis e grava o par escolhido; devolve o destino da tela."""
    import questionary

    gpu = catalog.gpu_by_id(gpu_id)
    while True:
        renderizar_cabecalho(con, estado, subtela=f"Modelos compatíveis com {gpu.id}")
        con.print(M.NADA_GRAVADO_ATE_CONFIRMAR, style="dim")
        con.print()
        selection, perfis_por_id = renderizar_modelos_por_gpu(con, gpu.id, estado)
        con.print()

        escolhas: list[Any] = []
        for row in selection.rows:
            pv = perfis_por_id.get(row.profile_id) if row.profile_id else None
            if pv is None:
                continue
            escolhas.append(
                questionary.Choice(
                    title=f"{pv.familia} — {pv.variante_curta} ({pv.contexto_curto})",
                    value=pv,
                )
            )
        if not escolhas:
            _renderizar_erro(
                con,
                f"Nenhum modelo compatível com {gpu.id} tem perfil pronto.",
                "As variantes que cabem nesta GPU não têm perfil em config.available_profiles().",
                "Escolha outra GPU ou selecione um perfil de modelo.",
            )
            con.print()
            escolha_vazio = questionary.select(
                "E agora?",
                choices=[
                    questionary.Choice(title="Escolher outra GPU", value="outra_gpu"),
                    questionary.Choice(title="Voltar ao menu principal", value="menu"),
                ],
                use_jk_keys=False,
                style=_estilo_select(),
                instruction=M.TECLAS_DETALHE,
            ).ask()
            if escolha_vazio is None or escolha_vazio == "menu":
                return "menu"
            return "outra_gpu"

        escolhas.append(questionary.Choice(title="Escolher outra GPU", value="__outra_gpu__"))
        escolhas.append(questionary.Choice(title="Voltar ao menu principal", value="__voltar__"))
        escolha = questionary.select(
            M.SELECAO_MODELO_GPU,
            choices=escolhas,
            use_search_filter=True,
            use_jk_keys=False,
            style=_estilo_select(),
            instruction=M.TECLAS_SELECAO,
        ).ask()
        if escolha is None or escolha == "__voltar__":
            return "menu"
        if escolha == "__outra_gpu__":
            return "outra_gpu"
        item: PerfilView = escolha

        renderizar_cabecalho(con, estado, subtela="Confirmar GPU e modelo")
        _renderizar_par_gpu_modelo(con, gpu, item)
        con.print()
        escolha_par = questionary.select(
            "Gravar este par em config.toml?",
            choices=[
                questionary.Choice(title=M.USAR_PAR_GPU_MODELO, value="usar"),
                questionary.Choice(title="Escolher outro modelo", value="modelo"),
                questionary.Choice(title="Voltar ao menu principal", value="menu"),
            ],
            use_jk_keys=False,
            style=_estilo_select(),
            instruction=M.TECLAS_DETALHE,
        ).ask()
        if escolha_par is None or escolha_par == "menu":
            return "menu"
        if escolha_par == "modelo":
            continue

        ok, _ = _aplicar_perfil_interativo(con, item, pedir_confirmacao=True, gpu_id=gpu.id)
        if not ok:
            _aguardar_enter(con)
            continue
        return _perguntar_proximo_passo(
            "configure",
            ler_estado_local(),
            extras=[("acao:harness", M.ACOES["harness"][0])],
        )


def _executar_gpu_modelo(con: Console, estado: EstadoLocal) -> str:
    """Caminho GPU + modelo: escolhe a GPU, o modelo compatível e confirma o par."""
    renderizar_cabecalho(con, estado, subtela=M.ACOES["selecionar_gpu"][0])
    bloqueio = _troca_bloqueada_por_sessao(con)
    if bloqueio:
        return bloqueio

    while True:
        renderizar_cabecalho(con, estado, subtela=M.ACOES["selecionar_gpu"][0])
        con.print(M.SOMENTE_LEITURA, style="dim")
        con.print()
        renderizar_gpus(con, estado)
        con.print()
        try:
            gpu_id = _escolher_gpu(con, estado)
        except KeyboardInterrupt:
            return "menu"
        if gpu_id == "__voltar__":
            return "menu"
        destino = _escolher_modelo_para_gpu(con, ler_estado_local(), gpu_id)
        if destino == "outra_gpu":
            continue
        if destino == "lista":
            # "Voltar para a lista" no detalhe do perfil recua para a lista de GPUs.
            continue
        return destino


def _executar_selecionar_gpu(con: Console, estado: EstadoLocal) -> str:
    """Submenu de GPU: reusa o fluxo de perfil pronto ou o par GPU + modelo."""
    import questionary

    renderizar_cabecalho(con, estado, subtela=M.ACOES["selecionar_gpu"][0])
    _mostrar_pre_acao(con, "selecionar_gpu")
    con.print()
    escolha = questionary.select(
        M.SELECAO_SUBMENU_GPU,
        choices=[
            questionary.Choice(title=M.SUBMENU_PERFIL, value="perfil"),
            questionary.Choice(title=M.SUBMENU_GPU_MODELOS, value="gpu"),
            questionary.Choice(title="Voltar ao menu principal", value="menu"),
        ],
        use_jk_keys=False,
        style=_estilo_select(),
        instruction=M.TECLAS_LISTA,
    ).ask()
    if escolha is None or escolha == "menu":
        return "menu"
    if escolha == "perfil":
        return _executar_configure(con, ler_estado_local())
    return _executar_gpu_modelo(con, ler_estado_local())


def renderizar_instrucoes_harness(
    con: Console, harness_id: str, cfg: config.Config, estado: EstadoLocal
) -> None:
    """Renderiza instruções manuais e dados da API para o harness escolhido."""
    import json

    sim = obter_simbolos(con)
    base_url = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    perfil_obj = None
    if cfg.profile:
        try:
            perfil_obj = config.load_profile(cfg.profile)
        except ValueError:
            perfil_obj = None
    perfil_id = perfil_obj.id if perfil_obj else cfg.served_model_name
    perfil_nome = perfil_obj.name if perfil_obj else cfg.served_model_name
    output_limit = perfil_obj.opencode_output_limit if perfil_obj else 8192
    opencode_model = f"byom/{perfil_id}"
    tool_call = bool(cfg.enable_auto_tool_choice)
    estado_sessao = (
        f"{sim.ok} ativa ({estado.sessao_nome})"
        if estado.sessao_ativa
        else f"{sim.pendente} inativa (inicie pela opção {M.ATALHOS_MENU['up']} antes de usar)"
    )

    resumo = Text()
    resumo.append("Endpoint da API: ", style="bold cyan")
    resumo.append(f"{base_url}\n")
    resumo.append("Modelo servido: ", style="bold cyan")
    resumo.append(f"{cfg.served_model_name}  {sim.barra}  Contexto: {cfg.max_model_len}\n")
    resumo.append("Sessão no Colab: ", style="bold cyan")
    resumo.append(estado_sessao)
    _painel(con, "Dados da conexão local", resumo, estilo="cyan", expand=True)

    if harness_id == "opencode":
        caminho = M.encurtar_caminho(opencode.resolve_config_path())
        bloco = json.dumps(
            {
                "model": opencode_model,
                "provider": {
                    "byom": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "BYOM Colab",
                        "options": {
                            "baseURL": base_url,
                            "modelsDiscovery": {
                                "enabled": True,
                                "modelInfoFormat": "vllm",
                            },
                        },
                        "models": {
                            perfil_id: {
                                "id": cfg.served_model_name,
                                "name": perfil_nome,
                                "limit": {
                                    "context": cfg.max_model_len,
                                    "output": output_limit,
                                },
                                "tool_call": tool_call,
                            }
                        },
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        instrucoes = (
            f"Arquivo de configuração: {caminho}\n\n"
            "Passo a passo para configurar manualmente:\n"
            f"  1. Abra o arquivo {caminho}.\n"
            f'  2. Adicione o provedor "byom" e defina "model": "{opencode_model}".\n'
            "     (Se o arquivo for .jsonc com comentários, mantenha seus comentários\n"
            "     e insira apenas o bloco abaixo.)\n"
            "  3. Reinicie o OpenCode após salvar.\n\n"
            f"{bloco}"
        )
        _painel(con, "Configuração manual — OpenCode", instrucoes, estilo="cyan", expand=True)
    elif harness_id == "dsh":
        caminho = M.encurtar_caminho(dsh.resolve_config_path())
        bloco = (
            "- id: llm-pi-ai\n"
            '  name: "@deepseek-ai/dsh-llm-pi-ai"\n'
            "  config:\n"
            "    providers:\n"
            "      byom:\n"
            "        apiKeyEnv: BYOM_API_KEY\n"
            "        api: openai-completions\n"
            f"        baseURL: {base_url}\n"
            "        models:\n"
            f'          - id: "{cfg.served_model_name}"\n'
            f'            name: "{perfil_nome}"\n'
            f"            contextWindow: {cfg.max_model_len}\n"
            "- id: agent-default-model\n"
            '  name: "@deepseek-ai/dsh-agent-default-model"\n'
            "  config:\n"
            '    provider: "byom"\n'
            f'    model: "{cfg.served_model_name}"'
        )
        instrucoes = (
            f"Arquivo de configuração: {caminho}\n\n"
            "Passo a passo para configurar manualmente:\n"
            "  1. No shell, defina qualquer valor para a variável de chave:\n"
            "       export BYOM_API_KEY=dummy\n"
            f"  2. Adicione os blocos abaixo ao arquivo {caminho}.\n"
            "  3. Reinicie o dsh após salvar.\n\n"
            f"{bloco}"
        )
        _painel(
            con,
            "Configuração manual — dsh (DeepSeek harness)",
            instrucoes,
            estilo="cyan",
            expand=True,
        )
    elif harness_id == "aider":
        instrucoes = (
            "Passo a passo para configurar manualmente:\n"
            "  1. Exporte as variáveis apontando para o proxy local do byom:\n"
            f"       export OPENAI_API_BASE={base_url}\n"
            "       export OPENAI_API_KEY=dummy\n"
            "  2. Execute o Aider informando o modelo servido:\n"
            f"       aider --model openai/{cfg.served_model_name}"
        )
        _painel(con, "Configuração manual — Aider", instrucoes, estilo="cyan", expand=True)
    else:
        instrucoes = (
            "Passo a passo para configurar manualmente:\n"
            "  1. Abra as configurações do Cline (ou outro cliente compatível com OpenAI).\n"
            "  2. Selecione o provedor: OpenAI Compatible.\n"
            f"  3. Base URL: {base_url}\n"
            "  4. API Key: dummy (o proxy local não exige chave externa).\n"
            f"  5. Model ID: {cfg.served_model_name}"
        )
        _painel(
            con,
            "Configuração manual — Cline / Cliente OpenAI",
            instrucoes,
            estilo="cyan",
            expand=True,
        )


def _aplicar_harness_automatico(con: Console, harness_id: str, cfg: config.Config) -> bool:
    """Configura automaticamente OpenCode ou dsh para usar o endpoint do Colab."""
    if harness_id == "opencode":
        try:
            res = opencode.sync_config(cfg)
        except opencode.OpenCodeConfigError as exc:
            err_txt = str(exc)
            if "valid JSON" in err_txt or "opencode.jsonc" in err_txt:
                _renderizar_erro(
                    con,
                    "Não foi possível atualizar o arquivo do OpenCode automaticamente.",
                    "O arquivo opencode.jsonc contém comentários ou vírgulas finais (JSONC), "
                    "e o byom não reescreve arquivos JSONC para evitar apagar comentários.",
                    "Copie o bloco JSON exibido acima e adicione-o manualmente no arquivo.",
                    detalhes_tecnicos=err_txt,
                )
            else:
                _renderizar_erro(
                    con,
                    "Falha ao configurar o OpenCode automaticamente.",
                    err_txt,
                    "Use as instruções manuais exibidas acima ou corrija o arquivo informado.",
                    detalhes_tecnicos=err_txt,
                )
            return False

        detalhes = [
            f"Arquivo: {M.encurtar_caminho(res.path)}",
            f"Modelo padrão: {res.model}",
            "Reinicie o OpenCode para carregar a configuração atualizada."
            if res.changed
            else "Nenhuma alteração necessária (a configuração já estava atualizada).",
        ]
        _mostrar_resultado(
            con,
            "harness",
            titulo_ok="OpenCode configurado para a API do Colab",
            linhas_detalhe=detalhes,
        )
        return True

    if harness_id == "dsh":
        try:
            res_dsh = dsh.sync_config(cfg)
        except dsh.DshConfigError as exc:
            _renderizar_erro(
                con,
                "Falha ao configurar o dsh automaticamente.",
                str(exc),
                "Use as instruções manuais exibidas acima ou ajuste o arquivo YAML.",
                detalhes_tecnicos=str(exc),
            )
            return False

        detalhes = [
            f"Arquivo: {M.encurtar_caminho(res_dsh.path)}",
            f"Modelo padrão: {res_dsh.model}",
            "Defina `export BYOM_API_KEY=dummy` e reinicie o dsh.",
        ]
        _mostrar_resultado(
            con,
            "harness",
            titulo_ok="dsh configurado para a API do Colab",
            linhas_detalhe=detalhes,
        )
        return True

    return False


def _executar_harness(con: Console, cli: Any, estado: EstadoLocal) -> str:
    import questionary

    # A config é lida antes de qualquer pergunta: com config.toml inválido o
    # painel precisa aparecer já na entrada da tela.
    cfg = _carregar_config(con)

    while True:
        renderizar_cabecalho(con, estado, subtela=M.ACOES["harness"][0])
        _mostrar_pre_acao(con, "harness")

        escolhas = [
            questionary.Choice(
                title=f"{nome} — {desc}",
                value=hid,
            )
            for hid, nome, desc, _auto in M.HARNESS_OPCOES
        ]
        escolhas.append(questionary.Choice(title="Voltar ao menu principal", value="__voltar__"))

        harness_id = questionary.select(
            M.SELECAO_HARNESS,
            choices=escolhas,
            use_jk_keys=False,
            style=_estilo_select(),
            instruction=M.TECLAS_LISTA,
        ).ask()
        if harness_id is None or harness_id == "__voltar__":
            return "menu"

        suporta_auto = next(
            auto for hid, _n, _d, auto in M.HARNESS_OPCOES if hid == str(harness_id)
        )

        renderizar_cabecalho(con, estado, subtela=M.ACOES["harness"][0])
        renderizar_instrucoes_harness(con, str(harness_id), cfg, estado)
        con.print()

        opcoes_tela: list[Any] = []
        if suporta_auto:
            opcoes_tela.append(
                questionary.Choice(
                    title="Configurar automaticamente a conexão com a API do Colab",
                    value="auto",
                )
            )
        opcoes_tela.append(questionary.Choice(title="Escolher outro harness", value="outro"))
        opcoes_tela.append(questionary.Choice(title="Voltar ao menu principal", value="menu"))

        acao_tela = questionary.select(
            "Como deseja prosseguir?",
            choices=opcoes_tela,
            use_jk_keys=False,
            style=_estilo_select(),
            instruction=M.TECLAS_DETALHE,
        ).ask()

        if acao_tela is None or acao_tela == "menu":
            return "menu"
        if acao_tela == "outro":
            continue
        if acao_tela == "auto":
            ok = _aplicar_harness_automatico(con, str(harness_id), cfg)
            if not ok:
                _aguardar_enter(con)
                continue
            return _perguntar_proximo_passo("harness", ler_estado_local())


def _executar_status(con: Console, cli: Any, estado: EstadoLocal) -> str:
    renderizar_cabecalho(con, estado, subtela=M.ACOES["status"][0])
    con.print(M.SOMENTE_LEITURA, style="dim")
    con.print()

    cfg = _carregar_config(con)
    detalhes_tecnicos: list[str] = []

    with ExecutorEtapas(
        con,
        [
            "Verificando estado local",
            "Verificando sessões no Colab",
            "Consultando worker e túnel local",
            "Consultando uso da conta",
        ],
    ) as exec_etapas:
        exec_etapas.iniciar(0)
        state, state_error = _ler_estado_ou_erro()
        if state_error:
            etapa_local = "estado ilegível"
        elif state is None:
            etapa_local = "sem sessão registrada"
        else:
            etapa_local = f"sessão {state.session}"
        exec_etapas.concluir(0, etapa_local, aviso=bool(state_error))

        exec_etapas.iniciar(1)
        try:
            orphans = cli.colab.orphan_assignments()
        except cli.colab.ColabCliError as exc:
            orphans = []
            exec_etapas.registrar_log(f"orphan_assignments -> {exc}")

        colab_status_txt = "sem sessão"
        if state is not None:
            try:
                line = cli.colab.status_line(state.session)
                if line is not None:
                    colab_status_txt = f"ativa ({line.hardware})"
                else:
                    allocated = cli.colab.is_allocated(state.session, state.endpoint)
                    colab_status_txt = "ativa" if allocated else "encerrada"
            except cli.colab.ColabCliError as exc:
                colab_status_txt = "desconhecida"
                exec_etapas.registrar_log(f"colab status -> {exc}")
        exec_etapas.concluir(1, colab_status_txt)

        exec_etapas.iniciar(2)
        payload = None
        if state is not None:
            payload = cli._daemon_status(cfg, expected_session=state.session)
            if payload is None:
                exec_etapas.concluir(2, "daemon local não responde", aviso=True)
            else:
                w_state = M.traduzir_estado_worker(payload.get("worker", {}).get("state"))
                exec_etapas.concluir(2, f"worker {w_state}")
        else:
            exec_etapas.concluir(2, "não aplicável")

        exec_etapas.iniciar(3)
        try:
            usage = cli.colab.usage()
        except cli.colab.ColabCliError as exc:
            usage = ""
            exec_etapas.registrar_log(f"colab usage -> {exc}")
        if usage:
            detalhes_tecnicos.append(usage)
            exec_etapas.concluir(3, "obtido")
        else:
            exec_etapas.concluir(3, "não disponível")

        detalhes_tecnicos.extend(exec_etapas.detalhes_tecnicos)

    if orphans:
        linhas_orf = "\n".join(f"• {o.endpoint} ({o.hardware})" for o in orphans)
        _painel(
            con,
            "Atenção: sessões sem registro local",
            f"Foram encontradas {len(orphans)} sessão(ões) no Colab sem registro local:\n"
            f"{linhas_orf}\n\nEncerre-as manualmente no Colab para evitar consumo de créditos.",
            estilo="yellow",
        )

    if state is None:
        if state_error:
            _renderizar_erro_estado(con, state_error)
        else:
            _mostrar_resultado(
                con,
                "status",
                titulo_ok="Nenhuma sessão ativa no momento",
                linhas_detalhe=[
                    f"Perfil configurado: {estado.perfil_nome or 'nenhum'} (GPU {estado.gpu_id})",
                ],
                duracao_s=exec_etapas.duracao_total(),
                detalhes_tecnicos="\n".join(detalhes_tecnicos),
            )
        return _perguntar_proximo_passo("status", ler_estado_local())

    tabela = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tabela.add_column("Campo", style="bold cyan", no_wrap=True)
    tabela.add_column("Valor")
    tabela.add_row("Sessão", state.session)
    tabela.add_row("Criada em", M.formatar_data_ptbr(state.created_at))
    if state.endpoint:
        tabela.add_row("Endpoint remoto", state.endpoint)
    tabela.add_row("Sessão no Colab", colab_status_txt)
    if payload is None:
        tabela.add_row("Worker local", "indisponível (daemon não está respondendo)")
    else:
        worker = payload.get("worker", {})
        tunnel = payload.get("tunnel", {})
        w_estado = M.traduzir_estado_worker(worker.get("state"))
        tabela.add_row("Worker", w_estado)
        if worker.get("error"):
            tabela.add_row("Motivo (worker)", str(worker["error"]))
        if worker.get("watchdog_silence_s") is not None:
            tabela.add_row("Watchdog", f"último sinal há {worker['watchdog_silence_s']:.0f}s")
        if worker.get("idle_s") is not None and worker.get("idle_timeout_min"):
            restante = max(0, worker["idle_timeout_min"] * 60 - worker["idle_s"])
            tabela.add_row(
                "Ociosidade",
                f"{worker['idle_s']:.0f}s (desliga em {restante:.0f}s)",
            )
        t_estado = M.traduzir_estado_worker(tunnel.get("state"))
        lat = tunnel.get("latency_ms") or "?"
        rec = tunnel.get("reconnects", 0)
        tabela.add_row("Túnel SSH", f"{t_estado} (reconexões: {rec}, latência: {lat} ms)")
        if tunnel.get("last_error"):
            tabela.add_row("Motivo (túnel)", str(tunnel["last_error"]))

    con.print()
    _painel(con, "Estado da sessão ativa", tabela, estilo="cyan", expand=True)
    _mostrar_resultado(
        con,
        "status",
        titulo_ok="Consulta de estado concluída",
        duracao_s=exec_etapas.duracao_total(),
        detalhes_tecnicos="\n".join(detalhes_tecnicos),
    )
    return _perguntar_proximo_passo("status", ler_estado_local())


def _executar_up(con: Console, cli: Any, estado: EstadoLocal) -> str:
    renderizar_cabecalho(con, estado, subtela=M.ACOES["up"][0])
    if not estado.ambiente_configurado:
        _painel(
            con,
            "Ambiente ainda não preparado",
            "É necessário verificar o SSH/Colab e gerar a chave local antes de iniciar uma sessão.",
            estilo="yellow",
        )
        if _confirmar(
            "init", pergunta="Deseja ir para 'Preparar ambiente local' agora?", default=True
        ):
            return "acao:init"
        return "menu"

    cfg = _carregar_config(con)
    try:
        fit = config.check_compatibility(cfg)
    except ValueError as exc:
        raise _ErroEtapa(
            "Configuração de GPU ou modelo inválida.",
            str(exc),
            "Selecione um perfil válido antes de iniciar a sessão.",
        ) from exc

    if not fit.fits:
        raise _ErroEtapa(
            f"O perfil atual não cabe na GPU configurada ({fit.gpu_id}).",
            fit.reason,
            "Selecione um perfil compatível ou altere a GPU em config.toml.",
        )

    problema_chave = cli.remote.validate_ssh_key(cfg.ssh_key_path)
    if problema_chave:
        raise _ErroEtapa(
            "A chave SSH do byom não é válida.",
            problema_chave[0].upper() + problema_chave[1:] + ".",
            f"Renomeie ou apague {M.encurtar_caminho(cfg.ssh_key_path)} e rode "
            "'Preparar ambiente local' para gerar uma chave nova. Nenhuma GPU foi alocada.",
        )

    _mostrar_pre_acao(con, "up")
    try:
        opcoes = _opcoes_up()
    except KeyboardInterrupt:
        return "menu"
    force = bool(opcoes["force"])

    etapas = [
        "Verificando sessões existentes",
        "Alocando GPU no Colab",
        "Conectando e enviando worker",
        "Inicializando servidor do modelo",
        "Aguardando API ficar pronta",
    ]

    allocated = False
    state: config.State | None = None
    name = cli._new_session_name()
    client = cli.remote.Remote(session=name, cfg=cfg)
    buf = io.StringIO()

    with _guarda_up(), ExecutorEtapas(con, etapas) as exec_etapas:
        try:
            # 1. Preparar sessão
            exec_etapas.iniciar(0)
            existing, erro_estado = _ler_estado_ou_erro()
            if erro_estado:
                exec_etapas.falhar(0, "estado ilegível")
                raise _ErroEtapa(
                    *_partes_erro_estado(erro_estado), detalhes_tecnicos=erro_estado
                )
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                cli._prepare_session(cfg, existing, force)
            exec_etapas.concluir(0, "sem conflitos")

            # 2. Alocar GPU
            exec_etapas.iniciar(
                1,
                f"GPU {cfg.gpu}",
                dica="Pedindo a máquina ao Colab. Costuma levar de 1 a 3 minutos.",
            )
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                cli._create_colab_session(cfg, name)
                allocated = True
                state = cli._record_allocation(name)
            exec_etapas.concluir(1, f"sessão {name}")

            # 3. SSH + upload
            exec_etapas.iniciar(
                2,
                dica="Abrindo o túnel SSH com a VM. Se falhar, tenta de novo sozinho (até ~2 min).",
            )
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                client.start(on_progress=lambda msg: exec_etapas.atualizar_detalhe(2, msg))
            exec_etapas.concluir(2, "conectado")

            # 4. Bootstrap
            exec_etapas.iniciar(
                3,
                cfg.runtime,
                dica="Instalando dependências e baixando o modelo. Pode levar vários minutos.",
            )
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                cli._bootstrap_worker(cfg, state, client)
                client.stop()
            exec_etapas.concluir(3, "worker iniciado")

            # 5. Daemon + wait ready
            exec_etapas.iniciar(
                4,
                dica="Carregando o modelo na GPU. Quando terminar, a API responde sozinha.",
            )
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                cli._register_daemon(cfg, state)
                cli._wait_ready(
                    cfg,
                    cfg.bootstrap_timeout_min * 60,
                    expected_session=state.session,
                    state=state,
                )
            exec_etapas.concluir(4, "API respondendo")
        except KeyboardInterrupt:
            exec_etapas.registrar_log(buf.getvalue())
            if allocated:
                # Mesmo teardown da CLI: `colab stop -s <nome>` não depende do
                # state.json e é idempotente para uma sessão inexistente.
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    cli._stop_allocation(name)
                    if state is not None:
                        cli._terminate_local(state)
            else:
                config.clear_state()
            raise
        except cli.UP_FAILURE_TYPES as exc:
            exec_etapas.registrar_log(buf.getvalue())
            exec_etapas.registrar_log(str(exc))
            for idx, et in enumerate(exec_etapas.etapas):
                if et.estado == "andamento":
                    exec_etapas.falhar(idx, "falhou")
            if allocated:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    cli._stop_allocation(name)
                    if state is not None:
                        cli._terminate_local(state)
            else:
                config.clear_state()
            raise _ErroEtapa(
                "Não foi possível iniciar a sessão no Colab.",
                str(exc),
                "Confira os detalhes técnicos abaixo, verifique suas unidades de computação "
                "e tente novamente.",
                detalhes_tecnicos=buf.getvalue() + f"\n{exc}",
            ) from exc
        except typer.Exit as exc:
            det_bruto = buf.getvalue().strip()
            exec_etapas.registrar_log(det_bruto)
            for idx, et in enumerate(exec_etapas.etapas):
                if et.estado == "andamento":
                    exec_etapas.falhar(idx, "interrompido")
            causa = "Uma verificação prévia impediu a alocação da GPU."
            acao = "Confira os detalhes técnicos ou execute `byom status`."
            if M.MARCADOR_SEM_REGISTRO in det_bruto:
                causa = "Foram encontradas sessões órfãs no Colab sem registro local."
                acao = "Encerre as sessões órfãs no Colab antes de iniciar uma nova."
            elif "already registered" in det_bruto:
                causa = "Já existe uma sessão registrada localmente."
                acao = "Use 'Verificar estado da sessão' ou 'Encerrar sessão no Colab' antes."
            elif "Could not check for orphaned" in det_bruto:
                causa = "Falha ao consultar sessões existentes na CLI do Colab."
                acao = "Verifique se `colab sessions` funciona e está autenticado no terminal."
            elif det_bruto:
                causa = det_bruto.splitlines()[-1].removeprefix("Error: ").strip()
            raise _ErroEtapa(
                "Não foi possível preparar a inicialização da sessão.",
                causa,
                acao,
                detalhes_tecnicos=det_bruto,
            ) from exc
        finally:
            client.stop()

    endpoint_api = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    detalhes = [
        f"Endereço da API: {endpoint_api}",
        f"Modelo servido: {cfg.served_model_name} (motor {cfg.runtime})",
        f"Sessão no Colab: {name}",
    ]
    if cfg.runtime == "image":
        detalhes.append(f"API de imagens: {endpoint_api}/images/generations")
    _mostrar_resultado(
        con,
        "up",
        titulo_ok="Sessão iniciada e API pronta",
        linhas_detalhe=detalhes,
        duracao_s=exec_etapas.duracao_total(),
        kwargs_cli={"force": force} if force else {},
        detalhes_tecnicos=buf.getvalue(),
    )
    return _perguntar_proximo_passo(
        "up",
        ler_estado_local(),
        extras=[("acao:harness", M.ACOES["harness"][0])],
    )


def _executar_chat(con: Console, cli: Any, estado: EstadoLocal) -> str:
    import questionary

    renderizar_cabecalho(con, estado, subtela=M.ACOES["chat"][0])
    if not estado.sessao_ativa:
        _painel(
            con,
            "Nenhuma sessão ativa",
            "O chat usa a API do modelo no Colab (127.0.0.1), que só existe com a sessão iniciada.",
            estilo="yellow",
        )
        if _confirmar(
            "up", pergunta="Deseja ir para 'Iniciar sessão no Colab' agora?", default=False
        ):
            return "acao:up"
        return "menu"

    cfg = _carregar_config(con)
    endpoint = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    try:
        resultado = llm_chat.run_setup(url=endpoint, model_name=cfg.served_model_name)
    except (llm_chat.LlmChatError, config.ConfigError) as exc:
        raise _ErroEtapa(
            "Não foi possível preparar o chat.",
            str(exc),
            "Confira a pasta do byom e tente novamente (equivalente: `byom setup`).",
        ) from exc

    _mostrar_pre_acao(con, "chat")
    sim = obter_simbolos(con)
    con.print(f"  {sim.bullet} Modelo: {resultado.model_name}  {sim.barra}  API: {endpoint}")
    con.print(
        f"  {sim.bullet} Histórico (se ativado): {M.encurtar_caminho(resultado.home / 'logs.db')}",
        style="dim",
    )
    con.print()
    escolha = questionary.select(
        "Como deseja conversar?",
        choices=[
            questionary.Choice(title="Conversar e salvar o histórico", value="salvar"),
            questionary.Choice(title="Conversar sem salvar o histórico", value="sem"),
            questionary.Choice(title="Voltar ao menu principal", value="menu"),
        ],
        use_jk_keys=False,
        style=_estilo_select(),
        instruction=M.TECLAS_DETALHE,
    ).ask()
    if escolha is None or escolha == "menu":
        return "menu"

    con.print()
    inicio = time.monotonic()
    try:
        resultado_chat = llm_chat.run_chat(no_history=escolha == "sem")
    except llm_chat.LlmChatError as exc:
        raise _ErroEtapa(
            "Não foi possível abrir o chat.",
            str(exc),
            "Rode 'Conversar' de novo ou execute `byom setup` no terminal.",
        ) from exc

    codigo = resultado_chat.returncode
    renderizar_cabecalho(con, ler_estado_local(), subtela=M.ACOES["chat"][0])
    if codigo == 0 or resultado_chat.aborted:
        _mostrar_resultado(
            con,
            "chat",
            titulo_ok="Conversa encerrada",
            linhas_detalhe=[
                "Histórico não foi salvo (modo sem histórico)."
                if escolha == "sem"
                else f"Histórico salvo em {M.encurtar_caminho(resultado.home / 'logs.db')}."
            ],
            duracao_s=time.monotonic() - inicio,
        )
    else:
        _renderizar_erro(
            con,
            f"O chat terminou com erro (código {codigo}).",
            "A API do modelo pode estar indisponível ou a sessão pode ter sido encerrada.",
            "Use 'Verificar estado da sessão' para conferir o worker e o túnel.",
            detalhes_tecnicos=resultado_chat.stderr,
        )
    return _perguntar_proximo_passo("chat", ler_estado_local())


def _executar_down(con: Console, cli: Any, estado: EstadoLocal) -> str:
    renderizar_cabecalho(con, estado, subtela=M.ACOES["down"][0])
    if not estado.sessao_ativa and not estado.erro_estado:
        _mostrar_resultado(
            con,
            "down",
            titulo_ok="Nenhuma sessão ativa para encerrar",
            linhas_detalhe=["Nenhuma GPU está alocada segundo o registro local."],
        )
        return _perguntar_proximo_passo("down", ler_estado_local())

    _mostrar_pre_acao(con, "down")
    try:
        _opcoes_down()
    except KeyboardInterrupt:
        return "menu"

    etapas = [
        "Lendo registro da sessão ativa",
        "Solicitando encerramento no Colab",
        "Confirmando liberação da GPU",
        "Encerrando processos locais",
    ]
    with _guarda_transacao("down"), ExecutorEtapas(con, etapas) as exec_etapas:
        exec_etapas.iniciar(0)
        try:
            state = config.load_state()
        except config.StateFileError as exc:
            exec_etapas.falhar(0, "estado inválido")
            raise _ErroEtapa(
                "Não foi possível ler o arquivo de estado local.",
                str(exc),
                "Verifique ~/.config/byom/state.json e o painel do Colab manualmente.",
            ) from exc
        if state is None:
            exec_etapas.concluir(0, "sem sessão ativa")
            for idx in (1, 2, 3):
                exec_etapas.concluir(idx, "não necessário")
            _mostrar_resultado(con, "down", titulo_ok="Nenhuma sessão ativa para encerrar")
            return _perguntar_proximo_passo("down", ler_estado_local())
        exec_etapas.concluir(0, state.session)

        exec_etapas.iniciar(1)
        try:
            cli.colab.stop(state.session)
            exec_etapas.concluir(1, "enviado")
        except cli.colab.ColabCliError as exc:
            exec_etapas.registrar_log(str(exc))
            exec_etapas.concluir(1, "falha ao enviar comando", aviso=True)

        exec_etapas.iniciar(2)
        confirmed = False
        try:
            confirmed = not cli.colab.is_allocated(state.session, state.endpoint)
            if confirmed:
                exec_etapas.concluir(2, "GPU liberada")
            else:
                exec_etapas.concluir(2, "ainda aparece alocada", aviso=True)
        except cli.colab.ColabCliError as exc:
            exec_etapas.registrar_log(str(exc))
            exec_etapas.concluir(2, "não confirmado", aviso=True)

        exec_etapas.iniciar(3)
        try:
            cfg = config.load_config()
        except config.ConfigError as exc:
            # `down` precisa funcionar mesmo com config.toml inválido.
            cfg = config.Config()
            exec_etapas.registrar_log(f"config.toml inválido; usando padrões locais: {exc}")
        cli._terminate_local(state)
        cli.remote.Remote(session=state.session, cfg=cfg).stop()
        if confirmed:
            try:
                config.clear_state(expected_session=state.session)
            except config.StateFileError as exc:
                exec_etapas.registrar_log(str(exc))
                confirmed = False
        if confirmed:
            with contextlib.redirect_stdout(io.StringIO()):
                cli._restore_harness(lambda _msg: None)
            exec_etapas.concluir(3, "estado limpo")
        else:
            exec_etapas.concluir(3, "registro local mantido por segurança", aviso=True)

    det_tec = "\n".join(exec_etapas.detalhes_tecnicos)
    if confirmed:
        _mostrar_resultado(
            con,
            "down",
            titulo_ok=f"Sessão '{state.session}' encerrada e confirmada pelo Colab",
            duracao_s=exec_etapas.duracao_total(),
            detalhes_tecnicos=det_tec,
        )
    else:
        _renderizar_erro(
            con,
            f"Não foi possível confirmar a liberação da sessão '{state.session}'.",
            "O Colab ainda indica a sessão ativa ou falhou ao responder.",
            "Confira o painel do Google Colab no navegador para evitar gasto de créditos "
            "e rode `byom status`.",
            detalhes_tecnicos=det_tec,
        )
    return _perguntar_proximo_passo("down", ler_estado_local())


class _ErroEtapa(Exception):
    def __init__(self, falha: str, causa: str, acao: str, *, detalhes_tecnicos: str = "") -> None:
        super().__init__(falha)
        self.falha = falha
        self.causa = causa
        self.acao = acao
        self.detalhes_tecnicos = detalhes_tecnicos


# ---------------------------------------------------------------------------
# Menu principal agrupado e loop de navegação (D1, D2, D3)
# ---------------------------------------------------------------------------


def _rotulo_item_menu(
    acao_id: str,
    estado: EstadoLocal,
    recomendado: str,
    sim: Simbolos,
    *,
    incluir_atalho: bool = True,
) -> str:
    atalho = M.ATALHOS_MENU[acao_id]
    rotulo = M.ACOES[acao_id][0]
    prefixo = f"{atalho}. {rotulo}" if incluir_atalho else rotulo
    if acao_id == recomendado:
        prefixo += f" {sim.estrela} recomendado"

    aviso = ""
    if acao_id == "selecionar_gpu" and estado.sessao_ativa:
        aviso = " [requer encerrar sessão]"
    elif acao_id == "up" and not estado.ambiente_configurado:
        aviso = " [requer preparar ambiente]"
    elif acao_id in ("down", "chat") and not estado.sessao_ativa:
        aviso = " [sem sessão ativa]"

    return f"{prefixo}{aviso}"


def renderizar_menu_estatico(con: Console, estado: EstadoLocal) -> None:
    """Renderiza a tela inicial em texto estruturado (útil para snapshots/testes)."""
    renderizar_cabecalho(con, estado)
    sim = obter_simbolos(con)
    recomendado = proximo_passo_recomendado(estado)

    for acao_id in M.MENU_INICIAL:
        linha = _rotulo_item_menu(acao_id, estado, recomendado, sim)
        estilo = "green" if acao_id == recomendado else ""
        # `markup=False`: os avisos usam colchetes e Rich os trataria como estilo.
        con.print(f"  {linha}", style=estilo, markup=False)

    con.print("  q. Sair")
    con.print()
    con.print(M.TECLAS_MENU, style="dim")


def _construir_escolhas_menu(estado: EstadoLocal, sim: Simbolos) -> tuple[list[Any], Any]:
    import questionary

    recomendado = proximo_passo_recomendado(estado)
    escolhas: list[Any] = []
    escolha_padrao: Any = None
    for acao_id in M.MENU_INICIAL:
        item = questionary.Choice(
            title=_rotulo_item_menu(acao_id, estado, recomendado, sim, incluir_atalho=False),
            value=acao_id,
            shortcut_key=M.ATALHOS_MENU[acao_id],
        )
        if acao_id == recomendado:
            escolha_padrao = item
        escolhas.append(item)
    escolhas.append(questionary.Choice(title="Sair", value="__sair__", shortcut_key="q"))
    return escolhas, escolha_padrao


def _despachar_acao(con: Console, cli: Any, acao_id: str) -> str:
    estado = ler_estado_local()
    if acao_id == "init":
        return _executar_init(con, cli, estado)
    if acao_id == "selecionar_gpu":
        return _executar_selecionar_gpu(con, estado)
    if acao_id == "configure":
        return _executar_configure(con, estado)
    if acao_id == "harness":
        return _executar_harness(con, cli, estado)
    if acao_id == "up":
        return _executar_up(con, cli, estado)
    if acao_id == "status":
        return _executar_status(con, cli, estado)
    if acao_id == "down":
        return _executar_down(con, cli, estado)
    if acao_id == "chat":
        return _executar_chat(con, cli, estado)
    return "menu"


def mostrar_menu(cli: Any) -> None:
    """Abre o menu principal. Chamado apenas pelo callback da CLI."""
    try:
        import questionary
    except ImportError:
        console.print(M.ERRO.format(**_partes_erro_inesperado(Path("questionary"))), style="red")
        raise typer.Exit(2) from None

    sim = obter_simbolos(console)
    erros_consecutivos = 0
    proxima_acao: str | None = None

    while True:
        try:
            if proxima_acao is None:
                estado = ler_estado_local()
                renderizar_cabecalho(console, estado)
                escolhas, _escolha_padrao = _construir_escolhas_menu(estado, sim)
                escolha = questionary.select(
                    M.MENU_PERGUNTA,
                    choices=escolhas,
                    use_shortcuts=True,
                    use_jk_keys=False,
                    style=_estilo_select(),
                    instruction=M.TECLAS_MENU,
                ).ask()
                if escolha is None or escolha == "__sair__":
                    break
                acao_id = str(escolha)
            else:
                acao_id = proxima_acao
                proxima_acao = None

            destino = _despachar_acao(console, cli, acao_id)
            if destino.startswith("acao:"):
                proxima_acao = destino.split(":", 1)[1]
            erros_consecutivos = 0
        except (KeyboardInterrupt, typer.Abort):
            console.print()
            console.print(_erro("cancelado"), style="yellow")
            break
        except _ErroEtapa as exc:
            _renderizar_erro(
                console,
                exc.falha,
                exc.causa,
                exc.acao,
                detalhes_tecnicos=exc.detalhes_tecnicos,
            )
            _aguardar_enter(console)
            proxima_acao = None
        except Exception as exc:  # noqa: BLE001 - nunca falhar em silêncio
            erros_consecutivos += 1
            caminho = _log_erro(exc)
            partes = _partes_erro_inesperado(caminho)
            _renderizar_erro(
                console,
                partes["falha"],
                partes["causa"],
                partes["acao"],
                detalhes_tecnicos=f"{type(exc).__name__}: {exc}",
            )
            proxima_acao = None
            if erros_consecutivos >= 3:
                console.print(
                    "Três falhas consecutivas. Encerrando a interface para evitar repetição.",
                    style="red",
                )
                break
            _aguardar_enter(console)
        finally:
            console.show_cursor()
