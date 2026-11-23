"""Interface interativa (TUI) do byom, construída sobre Typer e Rich.

Este módulo é apenas uma camada de apresentação: ele chama as mesmas funções
dos comandos Typer existentes. A CLI tradicional continua intacta.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from . import catalog, config
from . import mensagens as M

console = Console()
_ACOES: tuple[Acao, ...] = ()  # preenchido em _iniciar()


@dataclass(frozen=True)
class Acao:
    id: str
    rotulo: str
    descricao: str
    atalho: str
    funcao: str
    kwargs: Callable[[], dict[str, Any]]


def is_interactive() -> bool:
    """Verdadeiro apenas quando entrada e saída são um terminal real."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _iniciar(cli: Any) -> tuple[Acao, ...]:
    """Monta o mapa de ações usando as funções públicas dos comandos."""
    return (
        Acao("init", *M.ACOES["init"], "i", "init", lambda: {}),
        Acao("profiles", *M.ACOES["profiles"], "p", "profiles", lambda: {}),
        Acao("gpus", *M.ACOES["gpus"], "g", "gpus", lambda: {}),
        Acao("models", *M.ACOES["models"], "m", "models", lambda: {"gpu": _escolher_gpu()}),
        Acao(
            "configure", *M.ACOES["configure"], "c", "configure",
            lambda: {"profile_id": _escolher_perfil()},
        ),
        Acao("up", *M.ACOES["up"], "u", "up", lambda: _opcoes_up()),
        Acao("status", *M.ACOES["status"], "s", "status", lambda: {}),
        Acao("down", *M.ACOES["down"], "d", "down", lambda: _opcoes_down()),
    )


def _avisar_terminal_pequeno() -> None:
    try:
        size = shutil.get_terminal_size(fallback=(100, 30))
    except OSError:
        return
    if size.columns < 80 or size.lines < 20:
        console.print(
            M.TERMINAL_PEQUENO.format(largura=size.columns, altura=size.lines),
            style="yellow",
        )


def _painel(titulo: str, texto: str, estilo: str = "cyan") -> None:
    console.print(Panel(texto, title=titulo, border_style=estilo, expand=False))


def _erro(chave: str, **kwargs: str) -> str:
    falha, causa, acao = M.ERROS_COMUNS[chave]
    return M.ERRO.format(
        falha=falha.format(**kwargs), causa=causa.format(**kwargs), acao=acao.format(**kwargs)
    )


def _comando_equivalente(acao_id: str, kwargs: dict[str, Any]) -> str:
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


class _CapturaSaida(io.StringIO):
    """Captura a saída do comando e atualiza o spinner com o que está acontecendo."""

    def __init__(self, status: Any) -> None:
        super().__init__()
        self._status = status

    def write(self, text: str) -> int:
        written = super().write(text)
        line = text.strip().splitlines()[-1] if text.strip() else ""
        if line:
            self._status.update(M.DURANTE.format(detalhe=line[:88]))
        return written


def _executar_com_status(func: Callable[..., None], kwargs: dict[str, Any]) -> str:
    with console.status(M.DURANTE.format(detalhe="preparando"), spinner="dots") as status:
        captura = _CapturaSaida(status)
        with contextlib.redirect_stdout(captura):
            func(**kwargs)
    saida = captura.getvalue()
    if saida:
        console.print(saida, end="")
    return saida


def _carregar_escolhas(
    titulo: str, itens: list[Any], descritor: Callable[[Any], tuple[str, str, str]]
) -> list[Any]:
    """Constrói escolhas com progresso determinístico (uma etapa por item)."""
    import questionary

    escolhas: list[Any] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        tarefa = progress.add_task(f"Preparando {titulo}", total=len(itens))
        for item in itens:
            nome, valor, detalhe = descritor(item)
            escolhas.append(questionary.Choice(title=nome, value=valor, description=detalhe))
            progress.advance(tarefa)
    return escolhas


def _escolher_gpu() -> str:
    import questionary

    escolhas = _carregar_escolhas(
        "GPUs",
        list(catalog.GPUS),
        lambda gpu: (f"{gpu.id} — {gpu.label} ({gpu.vram_gib} GiB)", gpu.id, gpu.label),
    )
    if not escolhas:
        raise ValueError(_erro("vazio", origem="catálogo de GPUs"))
    escolha = questionary.select(
        M.SELECAO_GPU,
        choices=escolhas,
        use_search_filter=True,
        use_jk_keys=False,
        show_description=True,
        instruction=M.TECLAS,
    ).ask()
    if escolha is None:
        raise KeyboardInterrupt
    return str(escolha)


def _escolher_perfil() -> str:
    import questionary

    perfis = list(config.available_profiles())
    escolhas = _carregar_escolhas(
        "perfis",
        perfis,
        lambda p: (f"{p.id} — {p.name}", p.id, p.description or "Sem descrição no perfil."),
    )
    if not escolhas:
        raise ValueError(_erro("vazio", origem="diretório de perfis"))
    escolha = questionary.select(
        M.SELECAO_PERFIL + M._busca,
        choices=escolhas,
        use_search_filter=True,
        use_jk_keys=False,
        show_description=True,
        instruction=M.TECLAS,
    ).ask()
    if escolha is None:
        raise KeyboardInterrupt
    return str(escolha)


def _confirmar(chave: str) -> bool:
    import questionary

    return bool(questionary.confirm(M.CONFIRMACOES[chave], default=False).ask())


def _opcoes_down() -> dict[str, Any]:
    if not _confirmar("down"):
        raise KeyboardInterrupt
    return {}


def _opcoes_up() -> dict[str, Any]:
    try:
        state = config.load_state()
    except config.StateFileError as exc:
        raise ValueError(_erro("caminho", caminho=str(exc))) from exc
    chave = "up_force" if state is not None else "up"
    if not _confirmar(chave):
        raise KeyboardInterrupt
    return {"force": chave == "up_force", "opencode_config": None, "dsh_config": None}


def _log_erro(exc: Exception) -> Path:
    diretorio = config.config_dir() / "logs"
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / "tui-erros.log"
    with caminho.open("a", encoding="utf-8") as handle:
        handle.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return caminho


def _resumo(acao: Acao, kwargs: dict[str, Any], inicio: float, ok: bool, detalhe: str = "") -> None:
    simbolo = M.SUCESSO if ok else M.FALHA
    estilo = "green" if ok else "red"
    texto = M.RESUMO.format(
        acao=acao.rotulo,
        resultado=detalhe or ("concluído" if ok else "não concluído"),
        duracao=time.monotonic() - inicio,
        proximo=M.PROXIMO[acao.id],
    )
    conteudo = texto + "\n\n" + M.IGUAL_ARQUIVO.format(
        comando=_comando_equivalente(acao.id, kwargs)
    )
    console.print()
    _painel(f"{simbolo} {M.RESUMO_TITULO}", conteudo, estilo)


def _executar_acao(cli: Any, acao: Acao) -> None:
    _painel(acao.rotulo, M.ANTES[acao.id] + "\n\n" + M.AJUDA_ACOES[acao.id] + "\n\n" + M.TECLAS)
    kwargs: dict[str, Any] = {}
    inicio = time.monotonic()
    try:
        kwargs = acao.kwargs()
        saida = _executar_com_status(getattr(cli, acao.funcao), kwargs)
    except (KeyboardInterrupt, typer.Abort):
        _resumo(acao, kwargs, inicio, False, "cancelado pelo usuário")
        raise
    except typer.Exit as exc:
        if exc.exit_code == 0:
            _resumo(acao, kwargs, inicio, True, "comando concluído sem alterações")
        else:
            _resumo(acao, kwargs, inicio, False, f"código de saída {exc.exit_code}")
        raise
    except Exception as exc:  # noqa: BLE001 - fronteira da TUI
        caminho = _log_erro(exc)
        console.print(M.ERRO.format(**_partes_erro_inesperado(caminho)))
        _resumo(acao, kwargs, inicio, False, "erro inesperado; detalhes no log")
        raise
    else:
        if not saida.strip():
            console.print(_erro("vazio", origem=acao.rotulo), style="yellow")
        _resumo(acao, kwargs, inicio, True)


def _partes_erro_inesperado(caminho: Path) -> dict[str, str]:
    falha, causa, acao = M.ERROS_COMUNS["inesperado"]
    return {
        "falha": falha,
        "causa": causa,
        "acao": acao.format(log=str(caminho)),
    }


def mostrar_menu(cli: Any) -> None:
    """Abre o menu principal. Chamado apenas pelo callback da CLI."""
    try:
        import questionary  # noqa: F401
    except ImportError:
        console.print(M.ERRO.format(**_partes_erro_inesperado(Path("questionary"))), style="red")
        raise typer.Exit(2) from None

    global _ACOES
    _ACOES = _iniciar(cli)
    _avisar_terminal_pequeno()
    console.print(Panel(f"{M.TITULO}\n{M.SUBTITULO}", border_style="cyan", expand=False))

    erros_consecutivos = 0
    while True:
        try:
            escolhas = [
                questionary.Choice(
                    title=f"{acao.rotulo} — {acao.descricao}",
                    value=acao.id,
                    shortcut_key=acao.atalho,
                )
                for acao in _ACOES
            ]
            escolhas += [
                questionary.Choice(
                    title="Ajuda sobre este menu", value="__ajuda__", shortcut_key="h"
                ),
                questionary.Choice(title="Sair", value="__sair__", shortcut_key="q"),
            ]
            escolha = questionary.select(
                M.MENU_PERGUNTA,
                choices=escolhas,
                use_shortcuts=True,
                show_description=True,
                instruction=M.TECLAS,
            ).ask()
            if escolha is None:
                console.print(_erro("cancelado"), style="yellow")
                break
            if escolha == "__sair__":
                break
            if escolha == "__ajuda__":
                _painel("Ajuda", M.MENU_AJUDA + "\n\n" + M.TECLAS)
                continue
            acao = next(item for item in _ACOES if item.id == escolha)
            _executar_acao(cli, acao)
            erros_consecutivos = 0
        except (KeyboardInterrupt, typer.Abort):
            console.print(_erro("cancelado"), style="yellow")
            break
        except typer.Exit as exc:
            if exc.exit_code:
                log = config.config_dir() / "logs" / "tui-erros.log"
                console.print(_erro("inesperado", log=str(log)), style="red")
            continue
        except Exception as exc:  # noqa: BLE001 - nunca falhar em silêncio
            erros_consecutivos += 1
            caminho = _log_erro(exc)
            console.print(M.ERRO.format(**_partes_erro_inesperado(caminho)), style="red")
            if erros_consecutivos >= 3:
                console.print(
                    "Três falhas consecutivas. Encerrando a interface para evitar repetição.",
                    style="red",
                )
                break
            continue
        finally:
            console.show_cursor()
