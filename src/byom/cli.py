"""Commands for the `byom` CLI."""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
import typer

from . import (
    catalog,
    colab,
    config,
    dsh,
    harness_state,
    llm_chat,
    mensagens,
    opencode,
    remote,
    webui,
)

app = typer.Typer(
    add_completion=False,
    help=(
        "Run a coding model on a Colab GPU and expose it at 127.0.0.1:4000/v1. "
        "Workflow: init → configure → up → status → chat → down."
    ),
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)


@app.callback(invoke_without_command=True)
def _entrada_interativa(
    ctx: typer.Context,
    ativar_tui: Annotated[
        bool,
        typer.Option("--tui", help="Abre o menu interativo mesmo sem subcomando."),
    ] = False,
    no_tui: Annotated[
        bool,
        typer.Option(
            "--no-tui", help="Força o modo CLI e mostra a ajuda quando não há subcomando."
        ),
    ] = False,
) -> None:
    """Menu interativo quando não há subcomando e há um terminal real."""
    if ctx.invoked_subcommand is not None:
        return
    from . import mensagens as M
    from . import tui

    if no_tui or not ativar_tui and not tui.is_interactive():
        typer.echo(ctx.get_help(), err=True)
        raise typer.Exit(2)
    if not tui.is_interactive():
        msg, causa, acao = M.ERROS_COMUNS["tty"]
        typer.echo(M.ERRO.format(falha=msg, causa=causa, acao=acao), err=True)
        raise typer.Exit(2)
    tui.mostrar_menu(sys.modules[__name__])
    raise typer.Exit(0)

#: Path to the worker bundle on the VM. It lives in `/content`, which always
#: exists: the CLI content API does not create intermediate directories.
REMOTE_TARBALL = "/content/byom-worker.tar.gz"

WORKER_DIR = Path(__file__).resolve().parent.parent.parent / "worker"
WORKER_FILES = (
    "bootstrap.sh",
    "build_vllm_command.py",
    "prepare_runtime.py",
    "run_vllm.sh",
    "run_runtime.sh",
    "run_gateway.sh",
    "requirements.txt",
    "requirements.lock",
    "gateway.py",
    "watchdog.py",
)


def _worker_dir() -> Path:
    override = os.environ.get("BYOM_WORKER_DIR")
    return Path(override) if override else WORKER_DIR


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_session_name() -> str:
    return f"byom-{secrets.token_hex(4)}"


def _hardware_matches(requested: str, hardware: str) -> bool:
    aliases = {"G4": ("G4", "RTX PRO 6000")}
    normalized = requested.strip().upper()
    if not normalized:
        return False
    expected = aliases.get(normalized, (normalized,))
    actual = hardware.upper()
    return any(name in actual for name in expected)


def _say(message: str) -> None:
    typer.echo(message)


def _die(message: str, code: int = 1) -> NoReturn:
    typer.echo(f"Error: {message}", file=sys.stderr)
    raise typer.Exit(code)


def _load_config_or_die() -> config.Config:
    try:
        return config.load_config()
    except config.ConfigError as exc:
        _die(str(exc))


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------
@app.command()
def init() -> None:
    """Check local dependencies and create the configuration."""
    if shutil.which("ssh") is None:
        _die("`ssh` client not found in PATH")
    try:
        version = colab.version()
    except colab.ColabNotInstalled as exc:
        _die(str(exc))
    except colab.ColabCliError as exc:
        _die(f"Colab CLI returned an error: {exc}")

    _say(f"Colab CLI: {version or 'unknown version'}")

    proc = colab.run(["sessions"], check=False)
    if proc.returncode != 0:
        saida = proc.stderr or proc.stdout or b""
        if isinstance(saida, bytes):
            saida = saida.decode(errors="replace")
        detalhe = colab.tail(str(saida).strip())
        if "auth" in detalhe.lower() or "login" in detalhe.lower():
            acao = "Run `colab sessions` and complete the login flow before using byom."
        else:
            acao = (
                "Run `colab sessions` in a terminal to read the CLI error, fix it, "
                "then run `byom init` again."
            )
        _die(
            "Colab CLI could not list sessions.\n"
            f"  Last CLI output: {detalhe or 'none'}\n"
            f"  What to do: {acao}"
        )

    cfg = _load_config_or_die()
    if not cfg.ssh_key_path.exists():
        cfg.ssh_key_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
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
        )
        # ssh-keygen's default mode is safe, but make it explicit: the private
        # key grants access to the ephemeral tunnel without a passphrase.
        cfg.ssh_key_path.chmod(0o600)
        _say(f"Created SSH key at {cfg.ssh_key_path}")
    else:
        problem = remote.validate_ssh_key(cfg.ssh_key_path)
        if problem:
            _die(
                f"Invalid SSH key: {problem}. Move or delete it and run `byom init` again."
            )
        _say(f"SSH key already exists at {cfg.ssh_key_path}")

    path = config.save_config(cfg)
    _say(f"Configuration saved to {path}")
    _say("Ready")


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
def _package_owners() -> dict[str, str]:
    """profile_id -> publicador dos pesos, derivado do repositório de cada perfil.

    O publicador NÃO é campo do perfil: sai da org do repo de onde o worker baixa
    (`model_repo`, papel `diffusion` ou o `model` do vLLM), então trocar o repo de
    um perfil muda o rótulo sem editar dado novo.
    """
    return {
        profile.id: catalog.package_owner(
            catalog.package_source(profile.worker, str(profile.settings.get("model", "")))
        )
        for profile in config.available_profiles()
    }


@app.command()
def profiles() -> None:
    """List available model profiles with inherited reasoning and both contexts."""
    pacotes = _package_owners()
    for profile in config.available_profiles():
        context = profile.settings["max_model_len"]
        reasoning = profile.settings.get("reasoning_effort") or (
            catalog.model_by_id(profile.catalog_model).reasoning_effort
            if profile.catalog_model
            else None
        )
        shown = reasoning if reasoning is not None else "not documented"
        _say(f"{profile.id} — {profile.name} ({context:,} tokens servidos; reasoning {shown})")
        if profile.catalog_model:
            fornecedor = catalog.model_by_id(profile.catalog_model).provider
            _say(f"  {mensagens.ROTULO_FORNECEDOR}: {fornecedor}")
            pacote = pacotes.get(profile.id, "")
            if mensagens.publicador_diverge(fornecedor, pacote):
                _say(f"  {mensagens.COLUNA_PACOTE}: {pacote}")
            ref = catalog.profile_context(profile.id)
            _say(
                f"  {profile.catalog_model}:{profile.catalog_variant} — "
                f"contexto doc {ref.context_doc:,} / servido {ref.context_served:,}"
            )
            _say(f"  {ref.reason}")
        if profile.description:
            _say(f"  {profile.description}")


@app.command()
def gpus() -> None:
    """List Colab GPUs, VRAM, bf16 and which catalog profiles fit each one."""
    refs = tuple(
        (
            profile.id,
            profile.catalog_model,
            profile.catalog_variant,
            profile.settings["max_model_len"],
        )
        for profile in config.available_profiles()
        if profile.catalog_model
    )
    _say(catalog.format_gpus(refs))


@app.command()
def models(
    gpu: Annotated[
        str | None,
        typer.Option(
            "--gpu",
            help="Catalog GPU ID; filters the listing without changing configuration.",
        ),
    ] = None,
) -> None:
    """List compatible models, optionally filtered by a catalog GPU ID."""
    if gpu is None:
        _say(catalog.format_gpus(numbered=True))
        _say("")
        _say("Enter the number or ID of the GPU to filter models (no config is changed).")
        try:
            gpu = input("GPU: ")
        except (EOFError, KeyboardInterrupt):
            _die("No GPU selected; pass --gpu <ID> to skip the prompt")
    try:
        selected = catalog.parse_gpu_choice(gpu)
    except ValueError as exc:
        _die(str(exc))
    _say(
        catalog.format_compatible_models(
            selected.id,
            _package_owners(),
            config.drafter_gib_by_profile(),
            config.kv_cache_dtype_by_profile(),
        )
    )


@app.command()
def configure(
    profile_id: str = typer.Option(..., "--profile", help="ID of the profile to apply."),
) -> None:
    """Apply a profile without changing local ports or options."""
    try:
        state = config.load_state()
    except config.StateFileError as exc:
        _die(str(exc))
    if state is not None:
        _die(f"Stop session '{state.session}' with `byom down` before switching profiles")
    cfg = _load_config_or_die()
    try:
        profile = config.apply_profile(cfg, profile_id)
    except ValueError as exc:
        _die(str(exc))
    path = config.save_config(cfg)
    _say(f"Applied profile: {profile.id} — {profile.name} (runtime {profile.runtime})")
    memory = "high memory" if cfg.gpu_high_mem else "standard memory"
    if cfg.runtime in catalog.NON_CHAT_RUNTIMES:
        contexto = mensagens.formatar_contexto_detalhe(cfg.max_model_len, cfg.runtime)
        _say(f"Contexto: {contexto}; GPU: {cfg.gpu} ({memory})")
    else:
        _say(f"Contexto servido: {cfg.max_model_len:,} tokens; GPU: {cfg.gpu} ({memory})")
    if profile.catalog_model:
        ref = catalog.profile_context(profile.id)
        if cfg.runtime in catalog.NON_CHAT_RUNTIMES:
            _say(f"Contexto documentado: n/a (runtime {cfg.runtime}); {ref.reason}")
        else:
            _say(
                f"Contexto documentado: {ref.context_doc:,} tokens "
                f"(modelo {profile.catalog_model}:{profile.catalog_variant}); {ref.reason}"
            )
    _say(f"Reasoning padrão do modelo: {cfg.reasoning_effort or 'not documented'}")
    _say(f"Configuration saved to {path}")
    if profile.opencode_model and profile.runtime not in catalog.NON_CHAT_RUNTIMES:
        _say(f"In OpenCode, select `{profile.opencode_model}` to use the matching context limit")


# ---------------------------------------------------------------------------
# up
# ---------------------------------------------------------------------------
def _build_worker_tar(cfg: config.Config, state: config.State, dest: Path) -> None:
    """Build the bundle with worker files and `session.json`."""
    worker = _worker_dir()
    missing = [name for name in WORKER_FILES if not (worker / name).is_file()]
    if missing:
        _die(f"Worker files missing from {worker}: {', '.join(missing)}")

    model_id, variant_id = config.active_catalog_ref(cfg)
    model_entry = catalog.model_by_id(model_id)
    worker_settings = dict((config.load_profile(cfg.profile).worker or {}) if cfg.profile else {})
    worker_settings.setdefault("sampling", dict(model_entry.sampling))
    payload = json.dumps(
        {
            "nonce": state.nonce,
            "model": cfg.model,
            "gpu": cfg.gpu,
            "gpu_high_mem": cfg.gpu_high_mem,
            "runtime": cfg.runtime,
            "catalog_model": model_id,
            "catalog_variant": variant_id,
            "worker": worker_settings,
            "served_model_name": cfg.served_model_name,
            "vllm_port": cfg.vllm_port,
            "gateway_port": cfg.gateway_port,
            "reasoning_effort": cfg.reasoning_effort,
            "thinking_policy": asdict(model_entry.thinking) if model_entry.thinking else None,
            "generation_defaults": dict(model_entry.generation_defaults) or None,
            "image_roles": dict(worker_settings.get("image_roles") or {}) or None,
            "idle_timeout_min": cfg.idle_timeout_min,
            "max_job_min": cfg.max_job_min,
            "watchdog_timeout_s": cfg.watchdog_timeout_s,
            "max_model_len": cfg.max_model_len,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "kv_cache_dtype": cfg.kv_cache_dtype,
            "enable_auto_tool_choice": cfg.enable_auto_tool_choice,
            "tool_call_parser": cfg.tool_call_parser,
            "reasoning_parser": cfg.reasoning_parser,
            "vllm_spec": cfg.vllm_spec,
            "vllm_enforce_eager": cfg.vllm_enforce_eager,
            "vllm_language_model_only": cfg.vllm_language_model_only,
        },
        indent=2,
    ).encode()

    with tarfile.open(dest, "w:gz") as tar:
        for name in WORKER_FILES:
            tar.add(worker / name, arcname=name)
        info = tarfile.TarInfo("session.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


def _declare_daemon_pid() -> None:
    """Run between fork and exec: publish this child's PID to the daemon."""
    # `Popen` passes the parent environment when `env=None`, so changing the
    # child's environment here is visible to `execve`. The daemon then knows
    # its PID before it starts, with no polling window.
    os.environ["BYOM_DAEMON_EXPECTED_PID"] = str(os.getpid())


def _start_daemon() -> int:
    """Start the local daemon (proxy + tunnel) and return its PID."""
    log_path = config.config_dir() / "logs" / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab")  # noqa: SIM115
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "byom", "_daemon"],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
            preexec_fn=_declare_daemon_pid,
        )
    finally:
        handle.close()
    return proc.pid


def _daemon_status(cfg: config.Config, *, expected_session: str | None = None) -> dict | None:
    try:
        response = httpx.get(f"http://127.0.0.1:{cfg.proxy_port}/__byom/status", timeout=3.0)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if expected_session is not None and payload.get("session") != expected_session:
        worker = payload.setdefault("worker", {})
        worker["state"] = "unavailable"
        worker["error"] = "local daemon is serving a different session"
    return payload


def _wait_ready(
    cfg: config.Config,
    timeout_s: float,
    *,
    expected_session: str | None = None,
    state: config.State | None = None,
) -> dict:
    """Wait for the gateway to validate the nonce and load the model.

    Falha rápido (em segundos, não no teto de `bootstrap_timeout_min`) quando não
    há mais o que esperar: o daemon registrado morreu, o status parou de
    responder ou o worker já se declarou indisponível por um motivo terminal
    (túnel com falha permanente, watchdog desistente).
    """
    deadline = time.monotonic() + timeout_s
    last = ""
    silences = 0
    while time.monotonic() < deadline:
        if (
            state is not None
            and state.daemon_pid
            and _pid_is_byom(
                state.daemon_pid, state.daemon_start_ticks, expected_session or ""
            )
            is False
        ):
            raise TimeoutError(
                f"o daemon local (pid {state.daemon_pid}) não existe mais; "
                f"a porta {cfg.proxy_port} pode estar ocupada — veja "
                f"{config.config_dir() / 'logs' / 'daemon.log'}"
            )
        payload = _daemon_status(cfg, expected_session=expected_session)
        if payload is not None:
            silences = 0
            worker = payload.get("worker", {})
            if worker.get("state") == "ready":
                return payload
            tunnel = payload.get("tunnel") or {}
            watchdog = payload.get("watchdog") or {}
            motivos = [
                str(valor)
                for valor in (
                    worker.get("error"),
                    tunnel.get("last_error"),
                    watchdog.get("last_error"),
                )
                if valor
            ]
            motivo = " | ".join(motivos) or "sem motivo informado"
            if worker.get("state") == "unavailable" and (
                tunnel.get("state") == "failed" or watchdog.get("state") == "gave_up"
            ):
                raise TimeoutError(f"worker indisponível: {motivo}")
            current = f"{worker.get('state')}: {worker.get('error') or 'waiting'}"
            if current != last:
                _say(f"  Worker {current}")
                last = current
        else:
            silences += 1
            if silences >= 3:
                raise TimeoutError(
                    f"o daemon local parou de responder em {silences} ciclos "
                    f"(porta {cfg.proxy_port} ocupada ou processo encerrado); "
                    f"veja {config.config_dir() / 'logs' / 'daemon.log'}"
                )
            if last != "daemon":
                _say("  Waiting for the local daemon to respond")
                last = "daemon"
        time.sleep(READY_POLL_S)
    raise TimeoutError(
        f"Worker did not become ready within {int(timeout_s)}s "
        f"(last state: {last or 'no status from the daemon'})"
    )



def _orphan_line(line: colab.SessionLine) -> str:
    """Describe an orphan and how to stop it, with name and endpoint."""
    endpoint = line.endpoint or "endpoint desconhecido"
    if line.name:
        return f"  - {line.name} ({endpoint}) — `colab stop -s {line.name}`"
    return f"  - [?] ({endpoint}) — only the Colab panel can stop this one"


def _check_orphans() -> None:
    """Block allocation while backend sessions have no local record."""
    try:
        state = config.load_state()
    except config.StateFileError as exc:
        _die(f"Cannot check the registered session: {exc}")
    known = {state.session} if state is not None else set()
    try:
        orphans = colab.unregistered_sessions(known)
    except colab.ColabCliError as exc:
        _die(f"Could not check for orphaned Colab sessions: {exc}")
    if not orphans:
        return
    lines = "\n".join(_orphan_line(line) for line in orphans)
    _die(
        f"Sessions still allocated with {mensagens.MARCADOR_SEM_REGISTRO}:\n"
        f"{lines}\n"
        "Stop them as shown above (or with `byom down`) before allocating again."
    )


def _port_free(port: int) -> bool:
    """True quando ninguém está escutando em `127.0.0.1:<port>`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _check_proxy_port(cfg: config.Config) -> None:
    """Recusa alocar GPU quando a porta local do proxy já está ocupada.

    Sem esta checagem o daemon morria no `uvicorn` (porta em uso) e o `up`
    esperava o modelo por até `bootstrap_timeout_min` com o daemon morto.
    """
    if not _port_free(cfg.proxy_port):
        _die(
            f"a porta {cfg.proxy_port} do proxy já está em uso por outro processo. "
            "Encerre o processo que a ocupa (ou rode `byom down`) antes de alocar a GPU."
        )
    if not _port_free(cfg.tunnel_port):
        _die(
            f"a porta {cfg.tunnel_port} do túnel já está em uso por outro processo. "
            "Encerre o processo que a ocupa (ou rode `byom down`) antes de alocar a GPU."
        )


def _prepare_session(
    cfg: config.Config, existing: config.State | None, force: bool
) -> None:
    """Resolve prior state before any external configuration is touched."""
    if existing is not None and not force:
        _die(
            f"Session '{existing.session}' is already registered. Use `byom status` or `byom down`."
        )
    if force and not _down(cfg):
        _die("Could not confirm shutdown; no new session was created")
    _check_proxy_port(cfg)
    # Checked on both paths: a corrupted state file may leave a VM allocated
    # with no local record, and that VM keeps consuming compute units.
    _check_orphans()



def _sync_external_configs(
    cfg: config.Config, opencode_config: Path | None, dsh_config: Path | None
) -> None:
    """Merge agent configs before allocating the GPU.

    Antes de sobrescrever, guarda em `harness_state.json` o modelo padrão que
    cada harness tinha: é o que o `down` devolve depois. Runtimes de mídia
    (imagem/TTS) não têm alias de chat e não devem tocar no harness.
    """
    if cfg.runtime in catalog.NON_CHAT_RUNTIMES:
        _say(
            f"OpenCode/dsh: runtime {cfg.runtime} não é de chat; "
            "configuração de harness ignorada"
        )
        return
    anteriores = harness_state.capture(opencode_config, dsh_config)
    if anteriores:
        harness_state.save(anteriores)
    try:
        opencode_result = opencode.sync_config(cfg, opencode_config)
    except opencode.OpenCodeConfigError as exc:
        _die(f"Could not configure OpenCode before allocating the GPU: {exc}")
    _say(f"OpenCode config: {opencode_result.path}; default model: {opencode_result.model}")
    if opencode_result.changed:
        _say("Restart OpenCode to load the updated configuration")

    if dsh_config is not None or dsh.is_installed():
        try:
            dsh_result = dsh.sync_config(cfg, dsh_config)
        except dsh.DshConfigError as exc:
            _die(f"Could not configure dsh before allocating the GPU: {exc}")
        _say(f"dsh config: {dsh_result.path}; default model: {dsh_result.model}")
        if dsh_result.changed:
            _say("Restart dsh to load the updated configuration")


def _create_colab_session(cfg: config.Config, name: str) -> None:
    """Allocate the backend VM."""
    _say(f"Creating session '{name}' with GPU {cfg.gpu}")
    if cfg.gpu_high_mem:
        colab.new(name, cfg.gpu, high_mem=True)
    else:
        colab.new(name, cfg.gpu)


def _record_allocation(name: str) -> config.State:
    """Persist the session identity as soon as the VM exists.

    `colab status` only enriches the endpoint: it runs *after* the write, so a
    slow or failing CLI can never leave an allocated VM without a local record.
    """
    state = config.State(
        session=name, endpoint="", nonce=secrets.token_hex(16), created_at=_now()
    )
    config.save_state(state)
    _enrich_endpoint(state)
    return state


def _enrich_endpoint(state: config.State) -> None:
    """Fill `state.endpoint` from the CLI; failure keeps the record as it is."""
    try:
        line = colab.status_line(state.session)
    except colab.ColabCliError as exc:
        _say(f"  Endpoint not resolved yet ({exc}); the tunnel retries it later")
        return
    if line is None or not line.endpoint:
        return
    state.endpoint = line.endpoint
    try:
        config.update_state(endpoint=line.endpoint, expected_session=state.session)
    except config.StateFileError:  # pragma: no cover - a corrida exigiria escrita externa
        return


def _stop_allocation(name: str) -> bool:
    """Stop an allocated session by name and try to confirm the deallocation.

    `colab stop -s <name>` treats an unknown session as a no-op, so calling it
    is idempotent and does not depend on `state.json`. When the CLI cannot
    confirm, the local record is kept so `byom down` can try again.
    """
    _say(f"Stopping session '{name}'")
    try:
        colab.stop(name)
    except colab.ColabCliError as exc:
        _say(f"Warning: `colab stop -s {name}` failed ({exc})")
        _say("  The VM may still be allocated; run `byom down` after fixing the CLI")
        return False
    try:
        confirmed = not colab.is_allocated(name)
    except colab.ColabCliError as exc:
        _say(f"Warning: could not confirm shutdown ({exc})")
        confirmed = False
    if not confirmed:
        _say("Unknown: could not confirm deallocation")
        _say("  The VM may still be consuming compute units; run `byom down`")
        return False
    try:
        config.clear_state(expected_session=name)
    except config.StateFileError as exc:
        # Alguém registrou outra sessão no meio do caminho: o state alheio fica.
        _say(f"Warning: {exc}")
        return False
    # O pin da chave de host vale por sessão: a próxima sessão é outra VM.
    remote.forget_session_host_key(name)
    _restore_harness(_say)
    _say("Session stopped (confirmed by Colab)")
    return True


def _bootstrap_worker(cfg: config.Config, state: config.State, client: remote.Remote) -> None:
    """Verify hardware, upload the worker bundle and start it on the VM."""
    hardware = client.gpu_name()
    if not _hardware_matches(cfg.gpu, hardware):
        raise RuntimeError(
            f"Allocated hardware is '{hardware or 'unknown'}'; expected {cfg.gpu}"
        )
    _say(f"Confirmed hardware: {hardware}")

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "worker.tar.gz"
        _build_worker_tar(cfg, state, tarball)
        _say("Uploading worker")
        client.upload(tarball, REMOTE_TARBALL)

    _say(f"Running bootstrap (installing dependencies and starting {cfg.runtime})")
    result = client.run(
        f"mkdir -p /content/byom && tar xzf {REMOTE_TARBALL} -C /content/byom "
        "&& bash /content/byom/bootstrap.sh",
        timeout=3600,
    )
    lines = [line for line in result.output.splitlines() if line.strip()]
    if lines:
        _say(f"  {lines[-1]}")


def _register_daemon(cfg: config.Config, state: config.State) -> None:
    """Start the daemon, confirm it survived, and merge its PID into the state.

    The start time is read right after `Popen`: `execve` does not change it, so
    it still identifies the daemon even before it announces itself. Se o filho
    morrer (porta do proxy ocupada, por exemplo), o `up` aborta aqui em vez de
    esperar o modelo por `bootstrap_timeout_min`.
    """
    pid = _start_daemon()
    ticks = config.process_start_ticks(pid)
    for _ in range(DAEMON_CONFIRM_TICKS):
        if _pid_is_byom(pid, ticks) is True:
            break
        time.sleep(DAEMON_CONFIRM_INTERVAL_S)
    else:
        raise RuntimeError(
            f"o daemon local (pid {pid}) morreu logo após iniciar; a porta "
            f"{cfg.proxy_port} do proxy provavelmente já está em uso — encerre o "
            f"processo que a ocupa (veja {config.config_dir() / 'logs' / 'daemon.log'})"
        )
    state.daemon_pid = pid
    state.daemon_start_ticks = ticks
    config.update_state(
        expected_session=state.session,
        daemon_pid=state.daemon_pid,
        daemon_start_ticks=state.daemon_start_ticks,
    )



#: Expected failures that must tear down an allocated session. Programming
#: errors are intentionally not swallowed.
UP_FAILURE_TYPES = (
    colab.ColabCliError,
    remote.RemoteError,
    OSError,
    RuntimeError,
    TimeoutError,
    ValueError,
    tarfile.TarError,
)


def _run_up(
    cfg: config.Config,
    force: bool,
    opencode_config: Path | None,
    dsh_config: Path | None,
) -> None:
    """Run the `up` transaction under the exclusive allocation lock."""
    try:
        with config.up_lock():
            _run_up_transaction(cfg, force, opencode_config, dsh_config)
    except config.ConfigError as exc:
        # O lock é adquirido antes de qualquer efeito colateral: a recusa não
        # aloca GPU e não toca no state.json da outra execução.
        _die(str(exc))


def _run_up_transaction(
    cfg: config.Config,
    force: bool,
    opencode_config: Path | None,
    dsh_config: Path | None,
) -> None:
    synced = False
    _say("[1/5] Checking local configuration and GPU/model compatibility")
    try:
        existing = config.load_state()
    except config.StateFileError as exc:
        _die(f"Cannot start: {exc}")

    try:
        fit = config.check_compatibility(cfg)
    except ValueError as exc:
        _die(f"Cannot start: {exc}")
    if not fit.fits:
        alternatives = ", ".join(fit.alternatives) or "none in this catalog"
        _die(
            f"GPU/model incompatibility: {fit.reason} "
            f"Alternatives that fit {fit.gpu_id}: {alternatives}. "
            "Change the profile or the GPU in config.toml; byom will not allocate a different GPU."
        )

    key_problem = remote.validate_ssh_key(cfg.ssh_key_path)
    if key_problem:
        _die(
            f"Cannot start: {key_problem}. "
            "Move or delete it and run `byom init` to create a new one."
        )

    _say("[2/5] Preparing the session and harness configuration")
    _prepare_session(cfg, existing, force)
    synced = True
    _sync_external_configs(cfg, opencode_config, dsh_config)

    _say("[3/5] Allocating a Colab runtime")
    name = _new_session_name()
    # The master connection is stopped at the end; the daemon keeps its own
    # connection alive to avoid competing for Colab's single `colab ssh` slot.
    client = remote.Remote(session=name, cfg=cfg)
    allocated = False
    state: config.State | None = None
    try:
        _create_colab_session(cfg, name)
        allocated = True
        state = _record_allocation(name)

        _say("[4/5] Uploading the worker and starting the model")
        client.start()
        _bootstrap_worker(cfg, state, client)

        client.stop()
        _register_daemon(cfg, state)

        _say("[5/5] Waiting for the model to become ready")
        _wait_ready(
            cfg,
            cfg.bootstrap_timeout_min * 60,
            expected_session=state.session,
            state=state,
        )
    except (KeyboardInterrupt, SystemExit):
        # Ctrl+C (or a nested command aborting) after the VM exists still has
        # to release it; the interruption is propagated after the cleanup.
        if allocated:
            stopped = _stop_allocation(name)
            # O daemon local pode ter subido antes da falha: sem isto ele fica
            # segurando a porta do proxy e o próximo `up` espera em vão.
            if state is not None:
                _terminate_local(state)
        else:
            config.clear_state()
            stopped = False
        if synced and not stopped:
            _restore_harness(_say)
        raise
    except UP_FAILURE_TYPES as exc:
        _say(f"Failed: {exc}")
        if allocated:
            # The session name is the only reliable handle here: `state.json`
            # may be unreadable by the time the failure unwinds.
            stopped = _stop_allocation(name)
            if state is not None:
                _terminate_local(state)
        else:
            config.clear_state()
            stopped = False
        if synced and not stopped:
            _restore_harness(_say)
        raise typer.Exit(1) from exc
    finally:
        client.stop()

    _say("")
    _say("Worker ready")
    _say(f"  Endpoint: http://127.0.0.1:{cfg.proxy_port}/v1")
    _say(f"  Model:    {cfg.served_model_name}")
    _say(f"  Runtime:  {cfg.runtime}")
    if cfg.runtime == "image":
        _say(f"  Image API: http://127.0.0.1:{cfg.proxy_port}/v1/images/generations")
    elif cfg.runtime == "tts":
        _say(f"  Speech API: http://127.0.0.1:{cfg.proxy_port}/v1/audio/speech")
    _say(f"  Session:  {name}")


@app.command()
def up(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Stop the current BYOM session and create a new one if shutdown is confirmed.",
        ),
    ] = False,
    opencode_config: Annotated[
        Path | None,
        typer.Option(
            "--opencode-config",
            help="Global OpenCode config file (default: XDG_CONFIG_HOME/opencode).",
        ),
    ] = None,
    dsh_config: Annotated[
        Path | None,
        typer.Option(
            "--dsh-config",
            help="dsh profile patch (default: ~/.dsh/profiles/desktop/cordis.patch.yml, "
            "only touched if it exists).",
        ),
    ] = None,
) -> None:
    """Create the session, start the worker, and publish the local endpoint."""
    cfg = _load_config_or_die()
    _run_up(cfg, force, opencode_config, dsh_config)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def _warn_orphans(orphans: list[colab.SessionLine]) -> None:
    """Warn about allocated sessions that `colab stop` cannot stop."""
    if not orphans:
        return
    _say(f"Warning: {len(orphans)} allocated session(s) have no local record:")
    for line in orphans:
        _say(f"  {line.endpoint} ({line.hardware})")
    _say("  `colab stop` cannot stop these sessions; stop them from Colab")


@app.command()
def status() -> None:
    """Show session, worker, tunnel, and account usage status."""
    cfg = _load_config_or_die()
    state_error: config.StateFileError | None = None
    try:
        state = config.load_state()
    except config.StateFileError as exc:
        state = None
        state_error = exc

    # Sessions allocated without local metadata cannot be stopped with
    # `colab stop` and continue consuming compute units. Show this warning even
    # when no session is registered, since creation may have failed midway.
    try:
        orphans = colab.orphan_assignments()
    except colab.ColabCliError:
        orphans = []
    _warn_orphans(orphans)

    if state is None:
        if state_error is not None:
            _say(f"Warning: {state_error}")
        else:
            _say("No active session.")
        if harness_state.load() is None and cfg.profile:
            _say(
                "Warning: no harness record to restore; OpenCode/dsh may still point to "
                "this byom endpoint (127.0.0.1)."
            )
        raise typer.Exit(0)

    _say(f"Session: {state.session}  (created at {state.created_at})")
    if state.endpoint:
        _say(f"Backend endpoint: {state.endpoint}")

    # Only `colab status`/`sessions` can confirm that a session has stopped;
    # an HTTP failure does not prove that the VM was deallocated.
    try:
        line = colab.status_line(state.session)
        if line is not None:
            _say(f"  Colab session: active (requested hardware: {line.hardware})")
        else:
            allocated = colab.is_allocated(state.session, state.endpoint)
            _say(f"  Colab session: {'active' if allocated else 'stopped'}")
    except colab.ColabCliError as exc:
        _say(f"  Colab session: unknown ({exc})")

    payload = _daemon_status(cfg, expected_session=state.session)
    if payload is None:
        _say("  Worker: unavailable (local daemon is not responding)")
    else:
        worker = payload.get("worker", {})
        tunnel = payload.get("tunnel", {})
        worker_state = worker.get("state")
        _say(f"  Worker: {worker_state}")
        if worker.get("error"):
            _say(f"    Reason: {worker['error']}")
        if worker.get("watchdog_silence_s") is not None:
            _say(f"    Watchdog: last heartbeat {worker['watchdog_silence_s']:.0f}s ago")
        if worker.get("idle_s") is not None and worker.get("idle_timeout_min"):
            remaining = worker["idle_timeout_min"] * 60 - worker["idle_s"]
            _say(f"    Idle time: {worker['idle_s']:.0f}s (shutdown in {max(0, remaining):.0f}s)")
        _say(
            f"  Tunnel: {tunnel.get('state')} "
            f"(reconnects: {tunnel.get('reconnects', 0)}, "
            f"latency: {tunnel.get('latency_ms') or '?'} ms)"
        )
        if tunnel.get("last_error"):
            _say(f"    Reason: {tunnel['last_error']}")
        watchdog_state = payload.get("watchdog") or {}
        if watchdog_state:
            _say(
                f"  Watchdog: {watchdog_state.get('state')} "
                f"(falhas consecutivas: {watchdog_state.get('failures', 0)})"
            )
            if watchdog_state.get("last_error"):
                _say(f"    Reason: {watchdog_state['last_error']}")

    usage = colab.usage()
    if usage:
        _say("Account usage (estimate from `colab usage`):")
        for line in usage.splitlines():
            _say(f"  {line}")


# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------
#: Intervalo entre checagens de `_wait_ready`. Falhas rápidas (daemon morto,
#: status silencioso, worker indisponível) acontecem em segundos, não no teto de
#: `bootstrap_timeout_min`; os testes encurtam este valor.
READY_POLL_S = 3.0

#: Confirmação de que o filho do daemon sobreviveu ao `Popen`: 20 x 0,05 s = 1 s.
DAEMON_CONFIRM_TICKS = 20
DAEMON_CONFIRM_INTERVAL_S = 0.05

#: Espera entre SIGTERM e SIGKILL. O daemon fecha o túnel e o `AsyncClient`
#: antes de sair; 5 s cobrem o caminho normal sem segurar o `down`.
TERMINATE_GRACE_S = 5.0


def _has_proc() -> bool:
    """True quando a plataforma expõe `/proc/self/stat` (Linux)."""
    return Path("/proc/self/stat").exists()


def _process_argv(pid: int) -> list[str]:
    """argv do processo: `/proc` quando existe, `ps -p <pid> -o command=` fora dele.

    O macOS não publica `start_ticks`, então a identidade lá tem de vir do argv.
    """
    if _has_proc():
        return config.process_cmdline(pid).split()
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return proc.stdout.decode(errors="replace").split()


def _has_sequence(argv: list[str], sequence: tuple[str, ...]) -> bool:
    want = list(sequence)
    size = len(want)
    return any(argv[i : i + size] == want for i in range(len(argv) - size + 1))


def _structural_kind(argv: list[str], session: str = "") -> str | None:
    """Classifica o argv como processo do byom: "daemon", "watchdog" ou None.

    Identidade ESTRUTURAL, não a palavra "byom" em qualquer posição: um processo
    do usuário com "byom" no argv não pode ser confundido com o daemon.
    """
    if not argv:
        return None
    if _has_sequence(argv, ("-m", "byom", "_daemon")) or (
        len(argv) >= 2 and Path(argv[0]).name == "byom" and argv[1] == "_daemon"
    ):
        return "daemon"
    # `colab exec -s <sessão> -f .../watchdog.py`: a sessão está no argv.
    if any(token.endswith("watchdog.py") for token in argv) and (
        not session or session in argv
    ):
        return "watchdog"
    return None


def _pid_is_byom(pid: int, start_ticks: int | None, session: str = "") -> bool | None:
    """True/False/None: é o processo registrado, não é, ou não dá para saber.

    - Com `/proc`, o instante de início (`campo 22`) é a prova: depois de um
      reboot o mesmo número pode pertencer a outro processo do usuário.
    - State antigo sem start time: só aceita com identidade estrutural no argv.
    - Sem `/proc` (macOS): a decisão vem do argv estrutural via `ps`. Quando
      nem isso confirma, devolve None — o chamador NÃO sinaliza, limpa o campo
      e pede que o usuário confira com `ps -p <pid> -o pid,command`.
    """
    if pid <= 0:
        return False
    current = config.process_start_ticks(pid)
    if current is not None:
        if start_ticks is not None:
            return current == start_ticks
        return _structural_kind(_process_argv(pid), session) is not None
    if _has_proc():
        return False  # /proc existe e não leu o stat: o processo morreu
    kind = _structural_kind(_process_argv(pid), session)
    return None if kind is None else True


def _wait_for_exit(
    pid: int, start_ticks: int | None, timeout_s: float, session: str = ""
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _pid_is_byom(pid, start_ticks, session) is not True:
            return True
        time.sleep(0.1)
    return _pid_is_byom(pid, start_ticks, session) is not True



def _clear_pid_fields(values: dict[str, int | None], expected_session: str) -> None:
    """Limpa campos de PID só no state da própria sessão.

    `StateFileError` aqui significa que o `state.json` atual é de outra sessão
    (ou desapareceu): não é meu registro para editar.
    """
    try:
        config.update_state(expected_session=expected_session, **values)
    except (config.StateFileError, OSError):
        return


def _terminate_local(state: config.State) -> None:
    """Stop the local daemon and watchdog, verifying identity first."""
    targets = (
        ("daemon", state.daemon_pid, state.daemon_start_ticks),
        ("watchdog", state.watchdog_pid, state.watchdog_start_ticks),
    )
    for label, pid, start_ticks in targets:
        if not pid:
            continue
        veredito = _pid_is_byom(pid, start_ticks, state.session)
        if veredito is None:
            # Sem start_ticks e sem estrutura reconhecível: não dá para provar
            # que é o processo do byom, então nada é sinalizado.
            _say(
                f"  {label} pid {pid}: identidade não confirmada; nenhum sinal enviado. "
                f"Confira com `ps -p {pid} -o pid,command`."
            )
            _clear_pid_fields(
                {f"{label}_pid": None, f"{label}_start_ticks": None}, state.session
            )
            continue
        if veredito is False:
            _say(
                f"  {label} pid {pid} is not the byom process registered in state.json "
                f"(stale state after a reboot?); no signal sent. "
                f"Check with `ps -p {pid} -o pid,command`."
            )
            _clear_pid_fields(
                {f"{label}_pid": None, f"{label}_start_ticks": None}, state.session
            )
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError as exc:  # pragma: no cover - system-dependent
            _say(f"Warning: could not stop process {pid}: {exc}")
            continue
        if _wait_for_exit(pid, start_ticks, TERMINATE_GRACE_S, state.session):
            continue
        _say(f"  {label} pid {pid} ignored SIGTERM for {int(TERMINATE_GRACE_S)}s; sending SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as exc:  # pragma: no cover - system-dependent
            _say(f"Warning: could not kill process {pid}: {exc}")


def _restore_harness(say: Callable[[str], None]) -> None:
    """Devolve ao OpenCode/dsh o modelo padrão anterior ou avisa que não há registro."""
    restaurados, falha = harness_state.restore()
    if restaurados:
        say(f"Harness config restored: {', '.join(restaurados)}")
    if falha:
        say(f"Warning: could not restore harness config ({falha})")
    if not restaurados and not falha:
        say(
            "Warning: nothing recorded to restore; the harness default may still point "
            "to this byom endpoint (run `byom up` then `byom down`, or fix it by hand)"
        )


def _down(cfg: config.Config, *, quiet: bool = False) -> bool:
    """Stop the session and confirm it was deallocated. Return True if confirmed."""

    def say(message: str) -> None:
        if not quiet:
            _say(message)

    try:
        state = config.load_state()
    except config.StateFileError as exc:
        say(f"Cannot stop: {exc}")
        return False
    if state is None:
        if not quiet:
            _say("No active session.")
        return True

    try:
        say(f"Stopping session '{state.session}'")
        colab.stop(state.session)
    except colab.ColabCliError as exc:
        say(f"Warning: `colab stop` failed ({exc})")

    confirmed = False
    try:
        confirmed = not colab.is_allocated(state.session, state.endpoint)
    except colab.ColabCliError as exc:
        say(f"Warning: could not confirm shutdown ({exc})")

    _terminate_local(state)
    # The daemon closes the master connection when it exits; this also handles
    # the case where it has already exited and left the socket behind.
    remote.Remote(session=state.session, cfg=cfg).stop()

    if confirmed:
        try:
            config.clear_state(expected_session=state.session)
        except config.StateFileError as exc:
            say(f"Warning: {exc}")
            return False
        remote.forget_session_host_key(state.session)
        _restore_harness(say)
        say("Stopped (confirmed by Colab)")
    else:
        # A VM pode ter sumido sem confirmação: devolver o harness evita deixar
        # OpenCode/dsh apontando para um endpoint que não responde mais.
        _restore_harness(say)
        say("Unknown: could not confirm deallocation")
        say("  The VM may still be consuming compute units; check Colab and stop it manually")
        say("  Local state was retained so you can check again")
    return confirmed


@app.command()
def down() -> None:
    """Stop the session and verify the result with Colab."""
    cfg = _load_config_or_die()
    # Mesmo lock do `up`: um `down` lento não pode apagar o state de uma sessão
    # que outro processo acabou de criar.
    try:
        with config.up_lock(holder="down"):
            confirmed = _down(cfg)
    except config.ConfigError as exc:
        _die(str(exc))
    if not confirmed:
        raise typer.Exit(1)


@app.command()
def setup(
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help="URL base da API compatível com OpenAI no Colab/túnel (ex.: http://127.0.0.1:4000/v1).",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=(
                "Nome do modelo servido no endpoint remoto "
                "(padrão: served_model_name do config.toml)."
            ),
        ),
    ] = None,
    model_id: Annotated[
        str | None,
        typer.Option(
            "--model-id",
            help="Identificador local do modelo no llm (padrão: byom).",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="Chave de API opcional para o endpoint remoto.",
        ),
    ] = None,
) -> None:
    """Create or update the isolated byom directory and llm chat preferences."""
    try:
        result = llm_chat.run_setup(
            url=url,
            model_name=model,
            model_id=model_id,
            api_key=api_key,
        )
    except (llm_chat.LlmChatError, config.ConfigError) as exc:
        _die(str(exc))

    _say(f"Diretório byom: {result.home}")
    _say(f"URL da API:     {result.api_base}")
    _say(f"Modelo remoto:  {result.model_name} (ID local no llm: {result.model_id})")
    _say(f"Configuração:   {result.extra_models_path}")
    _say("Setup concluído. Execute `byom chat` para conversar com o modelo.")


@app.command()
def chat(
    system: Annotated[
        str | None,
        typer.Option(
            "--system",
            "-s",
            help="System prompt enviado ao modelo no início da conversa.",
        ),
    ] = None,
    no_history: Annotated[
        bool,
        typer.Option(
            "--no-history",
            "--no-log",
            "-n",
            help="Desliga a gravação do histórico em logs.db nesta sessão.",
        ),
    ] = False,
    continue_last: Annotated[
        bool,
        typer.Option(
            "--continue",
            "-c",
            help="Continua a conversa anterior registrada em logs.db.",
        ),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="ID do modelo configurado no llm (padrão: modelo salvo pelo setup).",
        ),
    ] = None,
) -> None:
    """Start an isolated `llm chat` session. Ctrl+D ends the conversation."""
    try:
        result = llm_chat.run_chat(
            system=system,
            no_history=no_history,
            continue_last=continue_last,
            model_id=model,
        )
    except llm_chat.LlmChatError as exc:
        _die(str(exc))
    if result.returncode != 0 and not result.aborted:
        if result.stderr:
            typer.echo(result.stderr, err=True)
        raise typer.Exit(result.returncode)


@app.command()
def web(
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Porta local do frontend Open WebUI."),
    ] = webui.DEFAULT_PORT,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Não abre o navegador automaticamente."),
    ] = False,
) -> None:
    """Start the local Open WebUI frontend connected to the byom proxy."""
    cfg = _load_config_or_die()
    try:
        plan = webui.plan(port=port, proxy_port=cfg.proxy_port)
    except (webui.WebUIError, config.ConfigError) as exc:
        _die(str(exc))

    _say(f"Frontend: {plan.url}")
    _say(f"Modelo:   {plan.proxy_url}")
    modelo_padrao = plan.env.get("DEFAULT_MODELS")
    if modelo_padrao:
        _say(f"Padrão:   {modelo_padrao}")
    if plan.overrides:
        _say("Ambiente: " + ", ".join(plan.overrides) + " (valores preservados)")
    _say(f"Dados:    {plan.data_dir}")
    _say(
        f"Open WebUI {webui.OPEN_WEBUI_VERSION} via uv; a primeira execução baixa o pacote "
        "e o Python 3.12."
    )
    _say("Suba a sessão com `byom up` e mantenha este processo aberto. Ctrl+C encerra.")
    _say(
        "Para usar outro provedor, defina OPENAI_API_BASE_URLS e OPENAI_API_KEYS "
        "antes de `byom web`."
    )
    try:
        code = webui.run(plan, open_browser=not no_browser)
    except webui.WebUIError as exc:
        _die(str(exc))
    if code != 0:
        raise typer.Exit(code)


@app.command()
def uninstall(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Pula a confirmação interativa antes de remover o diretório do byom.",
        ),
    ] = False,
) -> None:
    """Remove only the isolated byom directory after safety checks."""
    try:
        target = llm_chat.inspect_uninstall_target()
    except llm_chat.LlmChatError as exc:
        _die(str(exc))

    _say(f"Diretório a remover: {target.home}")
    if target.entries:
        _say("Itens encontrados:")
        for entry in target.entries:
            suffix = "/" if entry.is_dir() else ""
            _say(f"  - {entry.name}{suffix}")
    else:
        _say("  (diretório vazio)")

    if not yes:
        confirmed = typer.confirm("Confirmar remoção do diretório do byom?", default=False)
        if not confirmed:
            _say("Operação cancelada.")
            raise typer.Exit(0)

    try:
        llm_chat.execute_uninstall(target)
    except llm_chat.LlmChatError as exc:
        _die(str(exc))

    _say(f"Diretório removido: {target.home}")
    _say("Para desinstalar o pacote Python, execute: pip uninstall byom")


# ---------------------------------------------------------------------------
# daemon (hidden)
# ---------------------------------------------------------------------------
@app.command("_daemon", hidden=True)
def daemon_command() -> None:
    """Run the proxy and tunnel supervisor (internal use by `byom up`)."""
    from . import daemon as daemon_module

    raise typer.Exit(daemon_module.main())


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    app()


if __name__ == "__main__":
    main()
