"""Commands for the `byom` CLI."""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
import typer

from . import catalog, colab, config, dsh, llm_chat, opencode, remote

app = typer.Typer(
    add_completion=False,
    help="Run a coding model on a Colab GPU and expose it at 127.0.0.1:4000/v1.",
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
    expected = aliases.get(requested.strip().upper(), (requested.strip(),))
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
        _die(
            "Colab CLI is not authenticated. Run `colab sessions` and complete "
            "the login flow before using byom."
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
@app.command()
def profiles() -> None:
    """List available model profiles with inherited reasoning and both contexts."""
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
    _say(catalog.format_compatible_models(selected.id))


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
    _say(f"Contexto servido: {cfg.max_model_len:,} tokens; GPU: {cfg.gpu} ({memory})")
    if profile.catalog_model:
        ref = catalog.profile_context(profile.id)
        _say(
            f"Contexto documentado: {ref.context_doc:,} tokens "
            f"(modelo {profile.catalog_model}:{profile.catalog_variant}); {ref.reason}"
        )
    _say(f"Reasoning padrão do modelo: {cfg.reasoning_effort or 'not documented'}")
    _say(f"Configuration saved to {path}")
    if profile.opencode_model:
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
) -> dict:
    """Wait for the gateway to validate the nonce and load the model."""
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        payload = _daemon_status(cfg, expected_session=expected_session)
        if payload is not None:
            worker = payload.get("worker", {})
            if worker.get("state") == "ready":
                return payload
            current = f"{worker.get('state')}: {worker.get('error') or 'waiting'}"
            if current != last:
                _say(f"  Worker {current}")
                last = current
        else:
            if last != "daemon":
                _say("  Waiting for the local daemon to respond")
                last = "daemon"
        time.sleep(3)
    raise TimeoutError(f"Worker did not become ready within {int(timeout_s)}s")


def _check_orphans() -> None:
    """Block allocation while backend sessions have no local record."""
    try:
        orphans = colab.orphan_assignments()
    except colab.ColabCliError as exc:
        _die(f"Could not check for orphaned Colab sessions: {exc}")
    if not orphans:
        return
    endpoints = ", ".join(line.endpoint for line in orphans)
    _die(
        "Orphaned Colab sessions found "
        f"({endpoints}); stop them in Colab before creating another session"
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
    # Checked on both paths: a corrupted state file may leave a VM allocated
    # with no local record, and that VM keeps consuming compute units.
    _check_orphans()


def _sync_external_configs(
    cfg: config.Config, opencode_config: Path | None, dsh_config: Path | None
) -> None:
    """Merge agent configs before allocating the GPU."""
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
    """Persist the session identity as soon as the VM exists."""
    state = config.State(
        session=name, endpoint="", nonce=secrets.token_hex(16), created_at=_now()
    )
    line = colab.status_line(name)
    if line is not None:
        state.endpoint = line.endpoint
    config.save_state(state)
    return state


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


def _register_daemon(state: config.State) -> None:
    """Start the daemon and merge its PID into the existing state."""
    state.daemon_pid = _start_daemon()
    config.update_state(daemon_pid=state.daemon_pid)


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

    _prepare_session(cfg, existing, force)
    _sync_external_configs(cfg, opencode_config, dsh_config)

    name = _new_session_name()
    # The master connection is stopped at the end; the daemon keeps its own
    # connection alive to avoid competing for Colab's single `colab ssh` slot.
    client = remote.Remote(session=name, cfg=cfg)
    allocated = False
    try:
        _create_colab_session(cfg, name)
        allocated = True
        state = _record_allocation(name)

        client.start()
        _bootstrap_worker(cfg, state, client)

        client.stop()
        _register_daemon(state)

        _say("Waiting for the model to become ready")
        _wait_ready(cfg, cfg.bootstrap_timeout_min * 60, expected_session=state.session)
    except UP_FAILURE_TYPES as exc:
        _say(f"Failed: {exc}")
        if allocated:
            _say("Stopping session")
            _down(cfg, quiet=True)
        else:
            config.clear_state()
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

    usage = colab.usage()
    if usage:
        _say("Account usage (estimate from `colab usage`):")
        for line in usage.splitlines():
            _say(f"  {line}")


# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------
def _terminate_local(state: config.State) -> None:
    for pid in (state.daemon_pid, state.watchdog_pid):
        if not pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError as exc:  # pragma: no cover - system-dependent
            _say(f"Warning: could not stop process {pid}: {exc}")


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
        config.clear_state()
        say("Stopped (confirmed by Colab)")
    else:
        say("Unknown: could not confirm deallocation")
        say("  The VM may still be consuming compute units; check Colab and stop it manually")
        say("  Local state was retained so you can check again")
    return confirmed


@app.command()
def down() -> None:
    """Stop the session and verify the result with Colab."""
    cfg = _load_config_or_die()
    if not _down(cfg):
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
    """Start an interactive `llm chat` session isolated inside BYOM_HOME."""
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
