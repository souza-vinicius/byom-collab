from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

from typer.testing import CliRunner

from byom import cli, config, opencode

PROFILE_IDS = {"qwen3.8-27b-32k", "qwen3.8-27b-256k"}


def test_templates_definem_os_dois_contextos_e_aliases_opencode():
    profiles = {profile.id: profile for profile in config.available_profiles()}
    assert set(profiles) == PROFILE_IDS
    assert profiles["qwen3.8-27b-32k"].settings["max_model_len"] == 32768
    assert profiles["qwen3.8-27b-256k"].settings["max_model_len"] == 262144
    assert all(profile.settings["gpu"] == "A100" for profile in profiles.values())
    assert profiles["qwen3.8-27b-32k"].settings["gpu_high_mem"] is False
    assert profiles["qwen3.8-27b-256k"].settings["gpu_high_mem"] is True

    aliases = opencode.profile_aliases_from_config()
    assert set(aliases) == PROFILE_IDS
    assert aliases["qwen3.8-27b-32k"]["id"] == "qwen3.8-27b"
    assert aliases["qwen3.8-27b-32k"]["limit"]["context"] == 32768
    assert aliases["qwen3.8-27b-256k"]["limit"]["context"] == 262144
    assert all(alias["limit"]["output"] == 8192 for alias in aliases.values())


def test_apply_profile_preserves_local_connection_settings():
    cfg = config.Config(proxy_port=4550, tunnel_port=4551, ssh_key="~/custom_key")

    profile = config.apply_profile(cfg, "qwen3.8-27b-256k")

    assert profile.opencode_model == "byom/qwen3.8-27b-256k"
    assert cfg.profile == profile.id
    assert cfg.max_model_len == 262144
    assert cfg.proxy_port == 4550
    assert cfg.tunnel_port == 4551
    assert cfg.ssh_key == "~/custom_key"


def test_load_profile_rejects_path_input():
    try:
        config.load_profile("../qwen3.8-27b-32k")
    except ValueError as exc:
        assert "Invalid profile ID" in str(exc)
    else:
        raise AssertionError("path-like profile ID should be rejected")


def test_configure_cli_saves_selected_profile():
    result = CliRunner().invoke(cli.app, ["configure", "--profile", "qwen3.8-27b-32k"])

    assert result.exit_code == 0, result.output
    assert "byom/qwen3.8-27b-32k" in result.output
    assert "standard memory" in result.output
    saved = config.load_config()
    assert saved.profile == "qwen3.8-27b-32k"
    assert saved.max_model_len == 32768


def test_worker_payload_and_command_use_profile_values(tmp_path: Path):
    cfg = config.Config()
    config.apply_profile(cfg, "qwen3.8-27b-32k")
    tar_path = tmp_path / "worker.tar.gz"
    state = config.State(session="s", endpoint="", nonce="test", created_at="now")

    cli._build_worker_tar(cfg, state, tar_path)

    with tarfile.open(tar_path, "r:gz") as archive:
        payload = json.load(archive.extractfile("session.json"))
    assert payload["served_model_name"] == "qwen3.8-27b"
    assert payload["max_model_len"] == 32768
    assert payload["gpu_memory_utilization"] == 0.92
    assert payload["kv_cache_dtype"] == "fp8"

    session_file = tmp_path / "session.json"
    session_file.write_text(json.dumps(payload), encoding="utf-8")
    command_script = Path(cli._worker_dir()) / "build_vllm_command.py"
    result = subprocess.run(
        [sys.executable, str(command_script), str(session_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    args = shlex.split(result.stdout.removeprefix("exec ").strip())
    assert "--max-model-len" in args
    assert args[args.index("--max-model-len") + 1] == "32768"
    assert args[args.index("--served-model-name") + 1] == "qwen3.8-27b"
    assert "--tool-call-parser" in args
