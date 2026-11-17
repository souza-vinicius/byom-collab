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
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer

from . import colab, config, dsh, opencode, remote

app = typer.Typer(
    add_completion=False,
    help="Run a coding model on a Colab GPU and expose it at 127.0.0.1:4000/v1.",
)

#: Path to the worker bundle on the VM. It lives in `/content`, which always
#: exists: the CLI content API does not create intermediate directories.
REMOTE_TARBALL = "/content/byom-worker.tar.gz"

WORKER_DIR = Path(__file__).resolve().parent.parent.parent / "worker"
WORKER_FILES = (
    "bootstrap.sh",
    "build_vllm_command.py",
    "run_vllm.sh",
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


def _die(message: str, code: int = 1) -> None:
    typer.echo(f"Error: {message}", file=sys.stderr)
    raise typer.Exit(code)


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

    cfg = config.load_config()
    if not cfg.ssh_key_path.exists():
        cfg.ssh_key_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603
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
        _say(f"Created SSH key at {cfg.ssh_key_path}")
    else:
        _say(f"SSH key already exists at {cfg.ssh_key_path}")

    path = config.save_config(cfg)
    _say(f"Configuration saved to {path}")
    _say("Ready")


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
@app.command()
def profiles() -> None:
    """List available model profiles."""
    for profile in config.available_profiles():
        context = profile.settings["max_model_len"]
        _say(f"{profile.id} — {profile.name} ({context:,} tokens)")
        if profile.description:
            _say(f"  {profile.description}")


@app.command()
def configure(
    profile_id: str = typer.Option(..., "--profile", help="ID of the profile to apply."),
) -> None:
    """Apply a profile without changing local ports or options."""
    state = config.load_state()
    if state is not None:
        _die(f"Stop session '{state.session}' with `byom down` before switching profiles")
    cfg = config.load_config()
    try:
        profile = config.apply_profile(cfg, profile_id)
    except ValueError as exc:
        _die(str(exc))
    path = config.save_config(cfg)
    _say(f"Applied profile: {profile.id} — {profile.name}")
    memory = "high memory" if cfg.gpu_high_mem else "standard memory"
    _say(f"Maximum context: {cfg.max_model_len:,} tokens; GPU: {cfg.gpu} ({memory})")
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

    payload = json.dumps(
        {
            "nonce": state.nonce,
            "model": cfg.model,
            "served_model_name": cfg.served_model_name,
            "vllm_port": cfg.vllm_port,
            "gateway_port": cfg.gateway_port,
            "reasoning_effort": cfg.reasoning_effort,
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


def _start_daemon() -> int:
    """Start the local daemon (proxy + tunnel) and return its PID."""
    log_path = config.config_dir() / "logs" / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab")  # noqa: SIM115
    try:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "byom", "_daemon"],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
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
    cfg = config.load_config()
    existing = config.load_state()
    if existing is not None and not force:
        _die(
            f"Session '{existing.session}' is already registered. Use `byom status` or `byom down`."
        )

    if force:
        if not _down(cfg):
            _die("Could not confirm shutdown; no new session was created")
        try:
            orphans = colab.orphan_assignments()
        except colab.ColabCliError as exc:
            _die(f"Could not check for orphaned Colab sessions: {exc}")
        if orphans:
            endpoints = ", ".join(line.endpoint for line in orphans)
            _die(
                "Orphaned Colab sessions found "
                f"({endpoints}); stop them in Colab before creating another session"
            )

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

    name = _new_session_name()
    state = config.State(session=name, endpoint="", nonce=secrets.token_hex(16), created_at=_now())
    # The master connection is stopped at the end; the daemon keeps its own
    # connection alive to avoid competing for Colab's single `colab ssh` slot.
    client = remote.Remote(session=name, cfg=cfg)
    allocated = False
    try:
        _say(f"Creating session '{name}' with GPU {cfg.gpu}")
        if cfg.gpu_high_mem:
            colab.new(name, cfg.gpu, high_mem=True)
        else:
            colab.new(name, cfg.gpu)
        allocated = True

        line = colab.status_line(name)
        if line is not None:
            state.endpoint = line.endpoint
        config.save_state(state)

        client.start()

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

        _say("Running bootstrap (installing dependencies and starting vLLM)")
        result = client.run(
            f"mkdir -p /content/byom && tar xzf {REMOTE_TARBALL} -C /content/byom "
            "&& bash /content/byom/bootstrap.sh",
            timeout=3600,
        )
        lines = [line for line in result.output.splitlines() if line.strip()]
        if lines:
            _say(f"  {lines[-1]}")

        client.stop()

        state.daemon_pid = _start_daemon()
        config.save_state(state)

        _say("Waiting for the model to become ready")
        _wait_ready(cfg, cfg.bootstrap_timeout_min * 60, expected_session=state.session)
    except Exception as exc:  # noqa: BLE001 - any failure triggers session shutdown
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
    _say(f"  Session:  {name}")


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
    cfg = config.load_config()
    state = config.load_state()

    # Sessions allocated without local metadata cannot be stopped with
    # `colab stop` and continue consuming compute units. Show this warning even
    # when no session is registered, since creation may have failed midway.
    try:
        orphans = colab.orphan_assignments()
    except colab.ColabCliError:
        orphans = []
    _warn_orphans(orphans)

    if state is None:
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
    state = config.load_state()
    if state is None:
        if not quiet:
            _say("No active session.")
        return True

    def say(message: str) -> None:
        if not quiet:
            _say(message)

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
    cfg = config.load_config()
    if not _down(cfg):
        raise typer.Exit(1)


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
