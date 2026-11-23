"""Run remote commands on the VM over a multiplexed SSH connection.

The published CLI (0.7.5) does not accept `colab ssh -- CMD`, and `colab
exec` is unfit for commands: `--timeout` is a whole-call deadline, and once it
expires the CLI keeps waiting for the kernel's `idle` instead of exiting. A
command that misses the `idle` message leaves the local process stuck
forever.

That is why byom commands do not go through the kernel. An OpenSSH master
connection (`ControlMaster`) holds **one** `colab ssh --proxy-mode` connection
and multiplexes commands, port forwarding, and file uploads. The kernel is
reserved for what truly needs it: the watchdog and `runtime.unassign()`.

The runtime accepts a single `colab ssh` connection at a time (HTTP 429); the
master owns that slot. A stray `colab ssh` kills the master.
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import colab, config

log = logging.getLogger("byom.remote")

#: The VM's sshd refuses the handshake right after another connection ends
#: (`Connection timed out during banner exchange`), and the runtime answers
#: 429 while a connection is active. Both cases pass on a retry.
START_ATTEMPTS = 6
START_BACKOFF_MIN_S = 5.0
START_BACKOFF_MAX_S = 30.0
MASTER_TIMEOUT_S = 90.0

#: Default timeout for a remote command. It is not a connection deadline but
#: a local limit: a command that leaves a background process holding the
#: output never returns EOF, and without this limit the wait would be
#: unbounded.
DEFAULT_TIMEOUT_S = 120.0

CHECK_TIMEOUT_S = 20.0
STOP_TIMEOUT_S = 30.0

#: Ports forwarded by the master: (local port, VM port).
Forward = tuple[int, int]


class RemoteError(RuntimeError):
    """Error raised when a VM operation fails."""


@dataclass
class RemoteResult:
    """Result of a remote command."""

    returncode: int
    output: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def socket_path(session: str) -> Path:
    """Path of the master connection socket.

    It must be short: Unix sockets have a ~104-byte path limit on macOS, and
    `~/.config/byom` already consumes much of that budget.
    """
    return config.config_dir() / f"cm-{session}.sock"


class Remote:
    """Commands and forwards on the VM over an OpenSSH master connection."""

    def __init__(
        self,
        *,
        session: str,
        cfg: config.Config,
        ssh_bin: str = "ssh",
        log_path: Path | None = None,
    ) -> None:
        self.session = session
        self.cfg = cfg
        self.ssh_bin = ssh_bin
        self.socket = socket_path(session)
        if log_path is None:
            log_path = config.config_dir() / "logs" / f"ssh-{session}.log"
        self.log_path = log_path
        self._forwards: tuple[Forward, ...] = ()

    # -- commands ---------------------------------------------------------
    def proxy_command(self) -> str:
        """`ProxyCommand` that links OpenSSH to the session's WebSocket.

        The identity must appear on both ends: the CLI sends the matching
        public key for the server to authorize, and the outer `ssh` offers
        the key passed with `-i`. Without `-i` here, the CLI uses the default
        key (`~/.ssh/id_ed25519`) and authentication fails with
        `Permission denied (publickey)`.
        """
        return shlex.join(
            [
                colab.colab_bin(),
                "--config",
                str(config.colab_sessions_path()),
                "ssh",
                "-s",
                self.session,
                "-i",
                str(self.cfg.ssh_key_path),
                "--proxy-mode",
            ]
        )

    def common_options(self) -> list[str]:
        """Options shared by the master and the multiplexed connections."""
        return [
            "-i",
            str(self.cfg.ssh_key_path),
            "-o",
            f"ProxyCommand={self.proxy_command()}",
            "-o",
            "ExitOnForwardFailure=yes",
            # The SSH channel is an ephemeral `colab ssh --proxy-mode` tunnel:
            # the host key changes with every VM, and the Colab CLI offers no
            # way to pass a known_hosts entry through. Documented in README.
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-S",
            str(self.socket),
        ]

    def master_command(self, forwards: Sequence[Forward] = ()) -> list[str]:
        """Command that starts the master and sets up the forwards."""
        command = [self.ssh_bin, "-M", "-fN", "-E", str(self.log_path)]
        command += self.common_options()
        for local, remote in forwards:
            command += ["-L", f"127.0.0.1:{local}:127.0.0.1:{remote}"]
        return [*command, "-l", "root", self.session]

    def client_command(self, command: str | None = None) -> list[str]:
        """Command that reuses the existing master."""
        base = [self.ssh_bin, *self.common_options(), "-l", "root", self.session]
        return base if command is None else [*base, command]

    def control_command(self, operation: str) -> list[str]:
        """`ssh -O <operation>` over the socket, without opening a connection."""
        return [
            self.ssh_bin,
            "-S",
            str(self.socket),
            "-O",
            operation,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-l",
            "root",
            self.session,
        ]

    # -- lifecycle --------------------------------------------------------
    def _clear_socket(self) -> None:
        self.socket.unlink(missing_ok=True)

    def log_tail(self, lines: int = 4) -> str:
        """Tail of the `ssh` log, where the real failure cause appears.

        Without this, a refused handshake or a runtime 429 becomes just
        "the connection did not come up".
        """
        if not self.log_path.exists():
            return ""
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        tail = [line for line in content.strip().splitlines() if line.strip()][-lines:]
        return " | ".join(tail)

    def start(self, forwards: Sequence[Forward] = ()) -> None:
        """Start the master, backing off for a refused banner and for 429s."""
        self.stop()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.unlink(missing_ok=True)

        reason = "no details"
        backoff = START_BACKOFF_MIN_S
        for attempt in range(1, START_ATTEMPTS + 1):
            self._clear_socket()
            try:
                # No pipes: the master detached by `-f` keeps the inherited
                # descriptors open, and `capture_output` would wait for an EOF
                # that never arrives. The failure reason comes from the `-E` log.
                proc = subprocess.run(
                    self.master_command(forwards),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=MASTER_TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                reason = f"connection did not return within {int(MASTER_TIMEOUT_S)}s"
            else:
                if proc.returncode == 0 and self.socket.exists():
                    self._forwards = tuple(forwards)
                    return
                reason = self.log_tail() or f"ssh exited with code {proc.returncode}"

            if attempt < START_ATTEMPTS:
                time.sleep(backoff)
                backoff = min(START_BACKOFF_MAX_S, backoff * 2)

        raise RemoteError(f"Could not connect to the VM over SSH: {reason}")

    def alive(self) -> bool:
        """True if the master responds over the socket."""
        if not self.socket.exists():
            return False
        try:
            proc = subprocess.run(
                self.control_command("check"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=CHECK_TIMEOUT_S,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.debug("SSH master check failed for %s: %s", self.session, exc)
            return False
        return proc.returncode == 0

    def stop(self) -> None:
        """Stop the master and remove the socket. Idempotent."""
        if self.socket.exists():
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                subprocess.run(
                    self.control_command("exit"),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=STOP_TIMEOUT_S,
                    check=False,
                )
        self._clear_socket()
        self._forwards = ()

    # -- usage ------------------------------------------------------------
    def run(
        self,
        command: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        check: bool = True,
        input_bytes: bytes | None = None,
    ) -> RemoteResult:
        """Run a shell command on the VM.

        Commands that leave background processes must redirect their output
        (`>/path/log 2>&1 </dev/null &`): a process inheriting the channel
        output keeps the connection open, and only the timeout ends the wait.
        """
        if not self.alive():
            raise RemoteError("SSH connection to the VM is not active")

        kwargs: dict[str, object] = {}
        if input_bytes is None:
            kwargs["stdin"] = subprocess.DEVNULL
        else:
            kwargs["input"] = input_bytes

        try:
            proc = subprocess.run(
                self.client_command(command),
                capture_output=True,
                timeout=timeout,
                check=False,
                **kwargs,  # type: ignore[arg-type]
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(f"`{command}` timed out on the VM after {int(timeout)}s") from exc

        output = (proc.stdout + proc.stderr).decode(errors="replace")
        result = RemoteResult(returncode=proc.returncode, output=output)
        if check and not result.ok:
            raise RemoteError(
                f"Remote command failed (exit code {result.returncode}): {colab.tail(output)}"
            )
        return result

    def upload(self, local_path: Path, remote_path: str, *, timeout: float = 300.0) -> None:
        """Send a file over the SSH channel.

        The remote directory must exist: the writer is `cat`, not an API that
        creates directories.
        """
        data = local_path.read_bytes()
        self.run(
            f"cat > {shlex.quote(remote_path)}",
            timeout=timeout,
            input_bytes=data,
        )

    # -- queries ----------------------------------------------------------
    def python_version(self) -> str:
        result = self.run("python3 --version", check=False)
        return result.output.strip()

    def gpu_name(self) -> str:
        """Name of the GPU actually allocated.

        `colab status` shows only the requested accelerator, so hardware
        confirmation has to come from inside the VM.
        """
        result = self.run(
            "nvidia-smi --query-gpu=name --format=csv,noheader",
            check=False,
        )
        return result.output.strip()
