"""Integration with the official Colab CLI.

Every call passes `--config <byom dir>/colab-sessions.json`, which isolates
sessions created by byom from the user's sessions. This matters:
`colab ssh --proxy-mode -s <name>` creates a new session when the name is not
in the local store, so the file must contain only byom sessions.

The CLI has no structured output: `sessions` and `status` print text lines in
the format

    [name] endpoint | Hardware: X | Shape: Y | Variant: Z[ | Status: W]

with `[?]` in place of the name when no matching local metadata exists.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config

#: Prefix of the CLI's informational messages, which are not session lines.
_INFO_PREFIX = "[colab]"

SESSION_LINE_RE = re.compile(
    r"""
    ^\[(?P<name>[^\]]*)\]\s+
    (?P<endpoint>\S+)\s+\|\s+
    Hardware:\s*(?P<hardware>[^|]*?)\s+\|\s+
    Shape:\s*(?P<shape>[^|]*?)\s+\|\s+
    Variant:\s*(?P<variant>[^|]*?)
    (?:\s+\|\s+Status:\s*(?P<status>.*?))?
    \s*$
    """,
    re.VERBOSE,
)


class ColabCliError(RuntimeError):
    """Error raised when the Colab CLI fails."""


class ColabNotInstalled(ColabCliError):
    """Error raised when the `colab` CLI is not available on PATH."""


@dataclass
class SessionLine:
    name: str | None
    endpoint: str
    hardware: str
    shape: str
    variant: str
    status: str | None = None

    @property
    def is_l4(self) -> bool:
        return self.hardware.strip().upper() == "L4"


def colab_bin() -> str:
    """Path of the `colab` CLI; tests can override it with `BYOM_COLAB_BIN`."""
    override = os.environ.get("BYOM_COLAB_BIN")
    if override:
        return override
    found = shutil.which("colab")
    if not found:
        raise ColabNotInstalled(
            "`colab` CLI not found on PATH. Install it with "
            "`uv tool install google-colab-cli` and authenticate with `colab sessions`."
        )
    return found


def base_command() -> list[str]:
    """Base command with byom's isolated `--config`.

    The directory is created here because the CLI builds the session file lock
    before creating the directory of the path given to `--config`: without
    this, the first call fails with `FileNotFoundError` on the `.lock`.
    """
    sessions = config.colab_sessions_path()
    sessions.parent.mkdir(parents=True, exist_ok=True)
    return [colab_bin(), "--config", str(sessions)]


def run(
    args: list[str],
    *,
    timeout: float | None = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run the Colab CLI and return the raw result."""
    cmd = base_command() + args
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise ColabNotInstalled(str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise ColabCliError(f"`colab {' '.join(args)}` timed out after {timeout}s") from exc
    if check and proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise ColabCliError(
            f"`colab {' '.join(args)}` failed (exit code {proc.returncode}): {tail(stderr)}"
        )
    return proc


def tail(text: str, limit: int = 2000) -> str:
    """Keep the end of the output, where the CLI prints the real error cause."""
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


def _text(proc: subprocess.CompletedProcess[bytes]) -> str:
    return proc.stdout.decode(errors="replace")


def version() -> str:
    return _text(run(["version"])).strip()


def parse_session_lines(text: str) -> list[SessionLine]:
    """Extract session lines, ignoring informational messages."""
    lines: list[SessionLine] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(_INFO_PREFIX):
            continue
        match = SESSION_LINE_RE.match(line)
        if not match:
            continue
        groups = match.groupdict()
        name = groups["name"].strip()
        lines.append(
            SessionLine(
                name=None if name in {"", "?"} else name,
                endpoint=groups["endpoint"],
                hardware=groups["hardware"],
                shape=groups["shape"],
                variant=groups["variant"],
                status=(groups["status"] or "").strip() or None,
            )
        )
    return lines


def sessions() -> list[SessionLine]:
    """Return active backend sessions from `colab sessions`."""
    return parse_session_lines(_text(run(["sessions"])))


def status_line(name: str) -> SessionLine | None:
    """Return the `colab status -s <name>` line, or None if metadata is missing.

    The hardware shown here is the **requested** accelerator (local
    metadata), not the one actually allocated.
    """
    found = parse_session_lines(_text(run(["status", "-s", name])))
    for line in found:
        # An endpoint from a differently named session is not this session's
        # endpoint; accepting it would point the tunnel at another worker.
        if line.name == name:
            return line
    return None


def find_by_name(name: str) -> SessionLine | None:
    for line in sessions():
        if line.name == name:
            return line
    return None


def find_by_endpoint(endpoint: str) -> SessionLine | None:
    if not endpoint:
        return None
    for line in sessions():
        if line.endpoint == endpoint:
            return line
    return None


def is_allocated(name: str, endpoint: str = "") -> bool:
    """Return True if the session (or its endpoint) still appears in `colab sessions`."""
    if endpoint and find_by_endpoint(endpoint) is not None:
        return True
    return find_by_name(name) is not None


def orphan_assignments() -> list[SessionLine]:
    """Return backend assignments without local metadata (shown as `[?]`).

    A session in this state consumes compute units, and `colab stop -s <name>`
    cannot stop it because the CLI cannot find it in the local store.
    """
    return [line for line in sessions() if line.name is None]


def new(name: str, gpu: str = "", *, high_mem: bool = False) -> None:
    """Create a session. Without `gpu`, the backend allocates a CPU runtime.

    The CLI saves local state **after** allocating the VM. If that write
    fails, the session stays allocated without a record, so the failure is
    reported with the list of orphaned sessions.
    """
    args = ["new", "-s", name]
    if gpu:
        args += ["--gpu", gpu]
    if high_mem:
        args.append("--high-mem")
    try:
        run(args, timeout=900)
    except ColabCliError as exc:
        raise ColabCliError(f"{exc}{_orphan_hint()}") from exc


def _orphan_hint() -> str:
    try:
        orphans = orphan_assignments()
    except ColabCliError:
        return "\nCheck Colab to see whether any sessions were left running."
    if not orphans:
        return ""
    endpoints = ", ".join(line.endpoint for line in orphans)
    return (
        f"\nWarning: {len(orphans)} allocated session(s) have no local metadata "
        f"({endpoints}). `colab stop` cannot stop these sessions; stop them in "
        "Colab to avoid continued compute unit usage."
    )


def stop(name: str) -> None:
    """Stop the session. The CLI treats a missing session as a no-op."""
    run(["stop", "-s", name], timeout=300)


def usage() -> str:
    return _text(run(["usage"], check=False)).strip()


def exec_popen(
    name: str,
    file: Path,
    *,
    timeout: float = 86400,
    log_path: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Run a local file as a long-running kernel cell.

    The `colab exec` `--timeout` is a whole-call deadline, and its 30s default
    would kill a long-running cell. The default here is one day.
    """
    cmd = base_command() + [
        "exec",
        "-s",
        name,
        "-f",
        str(file),
        "--timeout",
        str(timeout),
    ]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_path, "ab")  # noqa: SIM115 - the handle lives with the process
        try:
            return subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=handle,
                start_new_session=True,
            )
        finally:
            handle.close()
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
