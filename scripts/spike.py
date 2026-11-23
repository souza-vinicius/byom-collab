#!/usr/bin/env python3
"""Validate the PRD hypotheses and the model viability criterion on real Colab.

Runs in two phases:

  cheap  CPU session: H1 (tunnel), H2 (tmux), H3 (cell survives),
         H4 (unassign), H5 (busy kernel), H7 (session/endpoint correlation),
         H8 (silent cell outlasting the default timeout)
  model  GPU session: H6 (driver/CUDA/vLLM) and RF-08 (decode, context,
         tool calling). Requests a T4 by default; use `--gpu` for another
         accelerator.

Commands on the VM go through the SSH master connection (`byom.remote`), not
the kernel: `colab exec` may never return because CLI 0.7.5 keeps the call
alive past `--timeout`. The kernel is reserved for the cells of tests H3, H4,
H5, and H8, which are precisely about its behavior.

Usage:
    uv run python scripts/spike.py --phase cheap
    uv run python scripts/spike.py --phase model --keep
    uv run python scripts/spike.py --phase model --gpu L4    # if quota exists

The `model` phase does not use `gpu` from `config.toml`: the account used by
the spike has no L4 quota (the backend rejects with "no quota or
entitlement"), so the default is the T4 that the Colab CLI itself suggests as
an alternative.

Requires an installed and authenticated Colab CLI. The session is stopped at
the end unless `--keep` is passed. Nothing here uses the byom settings: the
spike creates its own sessions with the byom `--config`, so it cannot collide
with a session created by `byom up`.
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from byom import colab, config, remote  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []
REMOTE_HTTP_PORT = 8765
WATCHDOG_CELL_TIMEOUT_S = 86400
#: Accelerator requested by the `model` phase. The account used by the spike
#: has no L4 quota; the backend rejects the allocation with
#: "no quota or entitlement".
DEFAULT_GPU = "T4"


def record(ident: str, ok: bool | None, detail: str) -> None:
    mark = {True: "OK", False: "FAILED", None: "BLOCKED"}[ok]
    RESULTS.append((ident, mark, detail))
    print(f"  -> {ident}: {mark} - {detail}", flush=True)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(port: int, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback
            return response.status, response.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


# ---------------------------------------------------------------------------
# remote file helpers (over the master connection, not the kernel)
# ---------------------------------------------------------------------------
def _file_exists(client: remote.Remote, path: str) -> bool:
    result = client.run(f"test -f {path} && echo yes || echo no", check=False)
    return "yes" in result.output


def _count_lines(client: remote.Remote, path: str) -> int:
    result = client.run(f"wc -l < {path} 2>/dev/null || echo 0", check=False)
    lines = result.output.splitlines()
    try:
        return int(lines[-1])
    except (ValueError, IndexError):
        return 0


def _await_file(client: remote.Remote, path: str, timeout_s: float) -> bool:
    """Wait for the kernel cell to finish without occupying the kernel."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _file_exists(client, path):
            return True
        time.sleep(3)
    return False


def _cell(name: str, source: str) -> Path:
    """Write a local file with the code the kernel will execute."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(source)
        return Path(handle.name)


def _start_cell(session: str, source: str) -> subprocess.Popen[bytes]:
    """Start a long-running kernel cell in the background."""
    return colab.exec_popen(session, _cell(session, source), timeout=WATCHDOG_CELL_TIMEOUT_S)


# ---------------------------------------------------------------------------
# H1: SSH tunnel with ProxyCommand
# ---------------------------------------------------------------------------
def check_tunnel(client: remote.Remote, port: int) -> bool:
    print("H1: port forwarding over the master connection")
    try:
        client.start(forwards=[(port, REMOTE_HTTP_PORT)])
    except remote.RemoteError as exc:
        record("H1", False, str(exc))
        return False

    # The test server goes to the background with redirected output: a
    # process inheriting the channel output keeps the connection open.
    client.run(
        f"nohup python3 -m http.server {REMOTE_HTTP_PORT} --bind 127.0.0.1 "
        ">/content/spike-http.log 2>&1 </dev/null & sleep 1; echo started",
        timeout=60,
    )

    if not wait_for_port(port):
        record("H1", False, "the local tunnel port did not open")
        return False

    status, _ = http_get(f"http://127.0.0.1:{port}/")
    if status == 200:
        record("H1", True, f"HTTP 200 through the tunnel on port {port}")
        return True

    record("H1", False, f"tunnel opened, but the remote HTTP returned status {status}")
    return False


# ---------------------------------------------------------------------------
# H2: tmux processes survive the end of the command
# ---------------------------------------------------------------------------
def check_tmux(client: remote.Remote) -> None:
    print("H2: tmux processes survive the end of the SSH command")
    marker = f"/content/spike-tmux-{secrets.token_hex(4)}.txt"
    client.run(
        f"tmux kill-session -t spike 2>/dev/null; "
        f"tmux new-session -d -s spike 'for i in 1 2 3 4 5 6; do echo tick >> {marker}; "
        "sleep 5; done'",
        timeout=120,
    )
    time.sleep(20)
    count = _count_lines(client, marker)
    record(
        "H2",
        count >= 2,
        f"{count} marker lines after 20s (expected >= 2)",
    )
    client.run("tmux kill-session -t spike 2>/dev/null || true", check=False)


# ---------------------------------------------------------------------------
# H8: a silent cell is not killed by the default timeout
# ---------------------------------------------------------------------------
def check_long_silent_cell(client: remote.Remote, session: str) -> None:
    """H8: a silent cell outlasts the default timeout (30s) and finishes alone."""
    print("H8: a silent cell survives the default `colab exec` timeout")
    marker = f"/content/spike-silent-{secrets.token_hex(4)}.txt"
    source = f"""
import time
from pathlib import Path

time.sleep(45)
Path("{marker}").write_text("survived")
print("done")
"""
    proc = _start_cell(session, source)
    time.sleep(20)
    alive_after_20s = proc.poll() is None
    time.sleep(15)  # 35s: past the default 30s timeout
    alive_after_35s = proc.poll() is None
    finished = _await_file(client, marker, 45)
    survived = _file_exists(client, marker)
    proc.terminate()

    record(
        "H8",
        alive_after_20s and alive_after_35s and finished and survived,
        f"alive at 20s={alive_after_20s}; alive at 35s={alive_after_35s}; "
        f"finished={finished}; marker={survived}",
    )


# ---------------------------------------------------------------------------
# H3: the watchdog cell survives the death of the local process
# ---------------------------------------------------------------------------
def check_watchdog_cell(client: remote.Remote, session: str) -> None:
    print("H3: the long-running cell survives the death of the local `colab exec`")
    marker = f"/content/spike-cell-{secrets.token_hex(4)}.txt"
    done = f"/content/spike-cell-done-{secrets.token_hex(4)}.txt"
    source = f"""
import time
from pathlib import Path

path = Path("{marker}")
for _ in range(12):
    with path.open("a") as handle:
        handle.write("tick\\n")
    time.sleep(2)
Path("{done}").write_text("done")
print("done")
"""
    proc = _start_cell(session, source)
    time.sleep(8)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    local_gone = proc.poll() is not None
    before = _count_lines(client, marker)
    time.sleep(14)
    after = _count_lines(client, marker)
    # Wait for the cell to finish so nothing stays queued in the kernel.
    finished = _await_file(client, done, 40)

    record(
        "H3",
        local_gone and after > before,
        f"local process terminated={local_gone}; lines {before} -> {after} "
        f"after killing the client; cell finished={finished}",
    )


# ---------------------------------------------------------------------------
# H5: session commands with a busy kernel
# ---------------------------------------------------------------------------
def check_busy_kernel(client: remote.Remote, session: str) -> None:
    print("H5: `status`, `sessions`, and remote command with a busy kernel")
    source = "import time\nfor _ in range(6):\n    print('busy', flush=True)\n    time.sleep(5)\n"
    busy = _start_cell(session, source)
    time.sleep(8)

    status_ok = False
    try:
        status_ok = colab.status_line(session) is not None
    except colab.ColabCliError:
        status_ok = False

    sessions_ok = False
    try:
        sessions_ok = any(line.name == session or line.endpoint for line in colab.sessions())
    except colab.ColabCliError:
        sessions_ok = False

    shell_ok = False
    try:
        shell = client.run("echo ok", check=False, timeout=60)
        shell_ok = shell.ok and "ok" in shell.output
    except remote.RemoteError:
        shell_ok = False

    busy.terminate()
    record(
        "H5",
        status_ok and sessions_ok and shell_ok,
        f"status={status_ok} sessions={sessions_ok} remote_command={shell_ok}",
    )


# ---------------------------------------------------------------------------
# H7: correlation between name and endpoint
# ---------------------------------------------------------------------------
def check_endpoint_correlation(name: str) -> str:
    print("H7: `colab sessions` correlates name and endpoint")
    line = colab.find_by_name(name)
    if line is None:
        record("H7", False, "session not found in `colab sessions`")
        return ""
    status = colab.status_line(name)
    consistent = status is not None and status.endpoint == line.endpoint
    record(
        "H7",
        consistent,
        f"endpoint={line.endpoint} requested_hardware={line.hardware} "
        f"consistent_with_status={consistent}",
    )
    return line.endpoint


# ---------------------------------------------------------------------------
# H4: runtime.unassign() from a cell
# ---------------------------------------------------------------------------
def check_unassign(session_name: str) -> None:
    print("H4: `runtime.unassign()` deallocates the VM from a cell")
    source = """
import time
from google.colab import runtime

time.sleep(15)
print("calling unassign", flush=True)
runtime.unassign()
print("returned", flush=True)
"""
    proc = _start_cell(session_name, source)
    deadline = time.monotonic() + 240
    gone = False
    while time.monotonic() < deadline:
        time.sleep(10)
        if not colab.is_allocated(session_name):
            gone = True
            break
    proc.terminate()
    record(
        "H4",
        gone,
        "session deallocated after unassign from the cell"
        if gone
        else "session stayed allocated 240s after unassign",
    )


# ---------------------------------------------------------------------------
# H6 / RF-08: runtime and vLLM on GPU
# ---------------------------------------------------------------------------
def check_runtime(client: remote.Remote) -> None:
    print("H6: runtime (driver, CUDA, Python)")
    queries = (
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; "
        "python3 --version; "
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true"
    )
    result = client.run(queries, check=False, timeout=120)
    detail = " | ".join(result.output.strip().splitlines())
    record("H6", bool(detail), detail or "no runtime response")


def check_vllm(client: remote.Remote, cfg: config.Config) -> None:
    print(f"H6/RF-08: vLLM with Qwen3.8-27B on {cfg.gpu}")
    session_file = json.dumps(
        {
            "model": cfg.model,
            "vllm_spec": cfg.vllm_spec,
            "max_model_len": cfg.max_model_len,
            "vllm_port": cfg.vllm_port,
            "served_model_name": "qwen3.8-27b",
            "gpu_memory_utilization": 0.92,
            "vllm_language_model_only": True,
            "vllm_enforce_eager": cfg.vllm_enforce_eager,
        }
    ).encode()

    worker = Path(__file__).resolve().parent.parent / "worker"
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "worker.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            for path in worker.iterdir():
                if path.is_file():
                    tar.add(path, arcname=path.name)
            import io

            info = tarfile.TarInfo("session.json")
            info.size = len(session_file)
            tar.addfile(info, io.BytesIO(session_file))
        client.upload(tarball, "/content/byom-worker.tar.gz")

    started = time.monotonic()
    result = client.run("bash /content/byom/bootstrap.sh", timeout=3600, check=False)
    if not result.ok:
        record("H6", False, f"bootstrap failed: {colab.tail(result.output)}")
        return

    # The model still needs to load; probe from inside the VM.
    ready = False
    for _ in range(60):
        probe = client.run(
            f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{cfg.vllm_port}/v1/models",
            check=False,
            timeout=120,
        )
        if probe.output.strip() == "200":
            ready = True
            break
        time.sleep(15)

    if not ready:
        log = client.run("tail -40 /content/byom/logs/vllm.log", check=False)
        record("H6", False, f"vLLM did not respond; log: {colab.tail(log.output, 1200)}")
        return

    load_s = time.monotonic() - started
    record("H6", True, f"vLLM ready in {load_s:.0f}s (includes install and download)")

    check_decode_and_tools(client, cfg)


def check_decode_and_tools(client: remote.Remote, cfg: config.Config) -> None:
    script = f"""
import json, time, urllib.request

base = "http://127.0.0.1:{cfg.vllm_port}"
tools = [{{"type": "function", "function": {{
    "name": "get_status", "description": "Returns the status",
    "parameters": {{"type": "object", "properties": {{}}, "additionalProperties": False}},
}}}}]

def call(payload):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={{"content-type": "application/json"}},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode())

start = time.monotonic()
answer = call({{
    "model": "qwen3.8-27b",
    "messages": [{{"role": "user", "content": "Write a Python function that adds two numbers."}}],
    "max_tokens": 256,
    "chat_template_kwargs": {{"reasoning_effort": "low"}},
}})
elapsed = time.monotonic() - start
usage = answer.get("usage", {{}})
print(json.dumps({{
    "kind": "decode",
    "completion_tokens": usage.get("completion_tokens"),
    "seconds": round(elapsed, 2),
    "tok_s": round((usage.get("completion_tokens") or 0) / elapsed, 2) if elapsed else 0,
}}))

tool_answer = call({{
    "model": "qwen3.8-27b",
    "messages": [{{"role": "user", "content": "Use get_status to check the state."}}],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 256,
}})
message = tool_answer["choices"][0]["message"]
print(json.dumps({{
    "kind": "tools",
    "tool_calls": len(message.get("tool_calls") or []),
    "has_reasoning_field": "reasoning" in message,
}}))
"""
    remote_script = "/content/spike-decode.py"
    client.upload(_cell("decode", script), remote_script)
    result = client.run(f"python3 {remote_script}", check=False, timeout=1800)
    output = result.output
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            metrics[payload.get("kind", "?")] = payload

    decode = metrics.get("decode", {})
    tok_s = decode.get("tok_s")
    record(
        "RF-08a",
        bool(tok_s) and tok_s >= 10,
        f"decode {tok_s} tokens/s (target >= 10) in {decode.get('seconds')}s",
    )
    tools = metrics.get("tools", {})
    record(
        "RF-08b",
        bool(tools.get("tool_calls")),
        f"tool_calls={tools.get('tool_calls')} reasoning field={tools.get('has_reasoning_field')}",
    )
    if not metrics:
        record("RF-08", False, f"no metrics; output: {output[-800:]}")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def phase_cheap(cfg: config.Config, keep: bool) -> None:
    name = f"byom-spike-{secrets.token_hex(3)}"
    print(f"creating CPU session '{name}'")
    colab.new(name)
    client = remote.Remote(session=name, cfg=cfg)
    port = free_port()
    try:
        check_endpoint_correlation(name)
        if check_tunnel(client, port):
            check_tmux(client)
            check_long_silent_cell(client, name)
            check_watchdog_cell(client, name)
            check_busy_kernel(client, name)
        # unassign tears down the VM, so it runs last and the master exits first.
        client.stop()
        check_unassign(name)
    finally:
        client.stop()
        if keep:
            print(f"keeping session {name} (--keep); stop it with `colab stop -s {name}`")
        else:
            print(f"stopping session {name}")
            colab.stop(name)


def phase_model(cfg: config.Config, keep: bool) -> None:
    name = f"byom-spike-{secrets.token_hex(3)}"
    print(f"creating {cfg.gpu} session '{name}'")
    try:
        colab.new(name, cfg.gpu)
    except colab.ColabCliError as exc:
        record("H6", None, f"{cfg.gpu} allocation blocked: {exc}")
        return

    client = remote.Remote(session=name, cfg=cfg)
    try:
        client.start()
        hardware = client.gpu_name()
        record("H6-hw", cfg.gpu.upper() in hardware.upper(), f"allocated hardware: {hardware}")
        check_runtime(client)
        check_vllm(client, cfg)
    finally:
        client.stop()
        if keep:
            print(f"keeping session {name} (--keep); stop it with `colab stop -s {name}`")
        else:
            print(f"stopping session {name}")
            colab.stop(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["cheap", "model"], default="cheap")
    parser.add_argument("--keep", action="store_true", help="keep the session at the end")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help=f"accelerator requested by the model phase (default: {DEFAULT_GPU})",
    )
    args = parser.parse_args()

    cfg = config.load_config()
    cfg.gpu = args.gpu
    cfg.ssh_key_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.ssh_key_path.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "byom", "-f", str(cfg.ssh_key_path)],
            check=True,
        )
    cfg.vllm_enforce_eager = args.enforce_eager

    print(f"using the byom session config: {config.colab_sessions_path()}")
    if args.phase == "cheap":
        phase_cheap(cfg, args.keep)
    else:
        phase_model(cfg, args.keep)

    print("\nsummary")
    for ident, mark, detail in RESULTS:
        print(f"  {ident:<8} {mark:<7} {detail}")
    if any(mark == "FAILED" for _, mark, _ in RESULTS):
        return 1
    if any(mark == "BLOCKED" for _, mark, _ in RESULTS):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
