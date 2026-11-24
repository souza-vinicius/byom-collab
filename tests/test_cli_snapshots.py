"""Snapshots de regressão da CLI tradicional (`cli.py`).

Garante que o redesenho da TUI não altere comandos, flags, texto de saída ou
códigos de saída da CLI.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from byom import cli, colab, config

runner = CliRunner()


def _saida(resultado) -> str:
    return (getattr(resultado, "stdout", "") or "") + (getattr(resultado, "stderr", "") or "")


def test_snapshot_cli_sem_argumentos_e_flags_tui():
    r_default = runner.invoke(cli.app, [])
    r_no_tui = runner.invoke(cli.app, ["--no-tui"])
    r_tui = runner.invoke(cli.app, ["--tui"])

    assert r_default.exit_code == 2
    assert r_no_tui.exit_code == 2
    assert r_tui.exit_code == 2
    assert _saida(r_default) == _saida(r_no_tui)
    assert "Usage:" in _saida(r_default)
    assert "O que aconteceu:" in _saida(r_tui)


def test_snapshot_cli_profiles():
    res = runner.invoke(cli.app, ["profiles"])
    assert res.exit_code == 0
    esperado = (
        'gemma-4-12b-q4-gguf — Gemma 4 12b-q4 (32,768 tokens servidos; reasoning not documented)\n'
        '  gemma-4:12b-q4 — contexto doc 262,144 / servido 32,768\n'
        '  8 GiB de pesos em T4: 32K mantêm 2 GiB de KV.\n'
        '  Gemma 4 12B Unified 4 bit; thinking ligado por <|think|>.\n'
        'gemma-4-26b-a4b-q4-gguf — Gemma 4 26b-a4b-q4 (32,768 tokens servidos; reasoning not '
        'documented)\n'
        '  gemma-4:26b-a4b-q4 — contexto doc 262,144 / servido 32,768\n'
        '  18 GiB pesos + 0,65 visão + 2 runtime + 2 KV = 22,65: L4 fica com 1,2 GiB de margem '
        'em 32K.\n'
        '  Gemma 4 26B-A4B 4 bit com mmproj de visão.\n'
        'gemma-4-31b-q4-gguf — Gemma 4 31b-q4 (131,072 tokens servidos; reasoning not '
        'documented)\n'
        '  gemma-4:31b-q4 — contexto doc 262,144 / servido 131,072\n'
        '  20 GiB pesos + 0,65 visão + 2 runtime + 8 KV = 30,65 na A100-40, com 2 GiB de margem.\n'
        '  Gemma 4 31B 4 bit; requer A100.\n'
        'gemma-4-e4b-q4-gguf — Gemma 4 e4b-q4 (65,536 tokens servidos; reasoning not documented)\n'
        '  gemma-4:e4b-q4 — contexto doc 131,072 / servido 65,536\n'
        '  6 GiB de pesos em T4: 65.536 tokens mantêm 4 GiB de KV.\n'
        '  Gemma 4 E4B 4 bit; thinking ligado por <|think|>.\n'
        'inkling-2-mini-iq2-gguf — Inkling 2 Mini (2 bit, GGUF) (32,768 tokens servidos; '
        'reasoning high)\n'
        '  inkling-2-mini:q2 — contexto doc 1,000,000 / servido 32,768\n'
        '  11 GiB pesos + 0,65 visão + 2 runtime estouram os 15 GiB utilizáveis da T4; em L4, '
        '32K fica com ~7 GiB de margem.\n'
        '  2 bit para L4 (a T4 não comporta pesos + projetor de visão); build mainline b12611 '
        'com arquivos Mainline_Support.\n'
        'inkling-2-mini-q4-gguf — Inkling 2 Mini (4 bit, GGUF) (65,536 tokens servidos; '
        'reasoning high)\n'
        '  inkling-2-mini:q4 — contexto doc 1,000,000 / servido 65,536\n'
        '  18 GiB pesos + 0,65 visão + 2 runtime + 4 KV = 24,65: não cabe em L4 (24 GiB) com a '
        'folga de 1,2 GiB.\n'
        '  Quant recomendado UD-Q4_K_XL com visão; 64K exige A100 de 40 GB (L4 fica 1,2 GiB '
        'abaixo da folga).\n'
        'inkling-2-mini-q8-gguf — Inkling 2 Mini (8 bit, GGUF) (131,072 tokens servidos; '
        'reasoning high)\n'
        '  inkling-2-mini:q8 — contexto doc 1,000,000 / servido 131,072\n'
        '  24 GiB pesos + 0,65 visão + 2 runtime + 8 KV = 34,65 na A100-40, com 2 GiB de margem.\n'
        '  8 bit sem perdas; build mainline b12611 com arquivos Mainline_Support; requer A100 de '
        '40 GB ou mais.\n'
        'muse-glimmer-30b-q2-gguf — Muse Glimmer 30B (q2) (32,768 tokens servidos; reasoning '
        'high)\n'
        '  muse-glimmer-30b:q2 — contexto doc 131,072 / servido 32,768\n'
        '  12 GiB pesos + 0,65 visão + 2 runtime + 2 KV = 16,65: a T4 de 15 GiB utilizáveis não '
        'comporta.\n'
        '  Muse Glimmer 2 bit para L4 (a T4 não comporta o projetor de visão); build llama.cpp '
        'fixado.\n'
        'muse-glimmer-30b-q4-gguf — Muse Glimmer 30B (q4) (65,536 tokens servidos; reasoning '
        'high)\n'
        '  muse-glimmer-30b:q4 — contexto doc 131,072 / servido 65,536\n'
        '  17 GiB pesos + 0,65 visão + 2 runtime + 4 KV = 23,65: não cabe em L4 com a folga de '
        '1,2 GiB.\n'
        '  Muse Glimmer 4 bit com visão; 64K exige A100 de 40 GB (L4 fica 1,2 GiB abaixo da '
        'folga).\n'
        'qwen-image-2.1-fp8 — Qwen-Image-2.1 (FP8/INT8-ConvRot) (1 tokens servidos; reasoning '
        'not documented)\n'
        '  qwen-image-2.1:fp8-int8 — contexto doc 0 / servido 1\n'
        '  Modelo de difusão não tem janela autoregressiva; 1 é sentinela.\n'
        '  Pesos INT8-ConvRot para GPU de 24 GB+; endpoint de imagem, não chat.\n'
        'qwen-image-2.1-q4km — Qwen-Image-2.1 (GGUF Q4_K_M) (1 tokens servidos; reasoning not '
        'documented)\n'
        '  qwen-image-2.1:q4km — contexto doc 0 / servido 1\n'
        '  Modelo de difusão não tem janela autoregressiva; 1 é sentinela.\n'
        '  Geração de imagens em T4/L4; POST /v1/images/generations, não chat.\n'
        'qwen3.8-27b-256k — Qwen3.8 27B (256k) (262,144 tokens servidos; reasoning xhigh)\n'
        '  qwen3.8-27b-awq:q4-awq — contexto doc 262,144 / servido 262,144\n'
        '  Contexto máximo documentado mantido em A100 80 GB (high_mem).\n'
        '  Full native context for a high-memory A100.\n'
        'qwen3.8-27b-32k — Qwen3.8 27B (32k) (32,768 tokens servidos; reasoning xhigh)\n'
        '  qwen3.8-27b-awq:q4-awq — contexto doc 262,144 / servido 32,768\n'
        '  A100 40 GB sem high_mem: 32K limita o KV a 2 GiB.\n'
        '  Lower-context profile for an A100 without a high-memory requirement.\n'
        'qwen3.8-27b-q3-gguf — Qwen3.8 27B (3 bit, GGUF) (32,768 tokens servidos; reasoning '
        'xhigh)\n'
        '  qwen3.8-27b-gguf:q3 — contexto doc 262,144 / servido 32,768\n'
        '  12 GiB pesos + 2 runtime + 2 KV = 16 GiB: não cabe nos 15 GiB utilizáveis da T4.\n'
        '  GGUF 3 bit para L4 (pesos + runtime + KV passam dos 15 GiB da T4); alternativa ao AWQ '
        'em vLLM.\n'
        'qwen3.8-27b-q4-gguf — Qwen3.8 27B (4 bit, GGUF) (32,768 tokens servidos; reasoning '
        'xhigh)\n'
        '  qwen3.8-27b-gguf:q4 — contexto doc 262,144 / servido 32,768\n'
        '  17 GiB de pesos em L4: 32K mantém 2 GiB de KV.\n'
        '  GGUF 4 bit recomendado; requer L4 ou melhor.\n'
    )
    assert res.output == esperado


def test_snapshot_cli_gpus():
    res = runner.invoke(cli.app, ["gpus"])
    assert res.exit_code == 0
    esperado = (
        'GPU | VRAM (GiB) | bf16 | VRAM aproximada | Perfis que cabem\n'
        '\n'
        'T4 (Tesla T4) | 15 | nao | sim | gemma-4-12b-q4-gguf, gemma-4-e4b-q4-gguf, '
        'qwen-image-2.1-q4km\n'
        'L4 (L4) | 24 | sim | sim | gemma-4-12b-q4-gguf, gemma-4-26b-a4b-q4-gguf, '
        'gemma-4-e4b-q4-gguf, inkling-2-mini-iq2-gguf, muse-glimmer-30b-q2-gguf, '
        'qwen-image-2.1-fp8, qwen-image-2.1-q4km, qwen3.8-27b-32k, qwen3.8-27b-q3-gguf, '
        'qwen3.8-27b-q4-gguf\n'
        'A100-40 (A100 40 GB) | 40 | sim | sim | gemma-4-12b-q4-gguf, gemma-4-26b-a4b-q4-gguf, '
        'gemma-4-31b-q4-gguf, gemma-4-e4b-q4-gguf, inkling-2-mini-iq2-gguf, '
        'inkling-2-mini-q4-gguf, inkling-2-mini-q8-gguf, muse-glimmer-30b-q2-gguf, '
        'muse-glimmer-30b-q4-gguf, qwen-image-2.1-fp8, qwen-image-2.1-q4km, qwen3.8-27b-256k, '
        'qwen3.8-27b-32k, qwen3.8-27b-q3-gguf, qwen3.8-27b-q4-gguf\n'
        'A100-80 (A100 80 GB) | 80 | sim | sim | gemma-4-12b-q4-gguf, gemma-4-26b-a4b-q4-gguf, '
        'gemma-4-31b-q4-gguf, gemma-4-e4b-q4-gguf, inkling-2-mini-iq2-gguf, '
        'inkling-2-mini-q4-gguf, inkling-2-mini-q8-gguf, muse-glimmer-30b-q2-gguf, '
        'muse-glimmer-30b-q4-gguf, qwen-image-2.1-fp8, qwen-image-2.1-q4km, qwen3.8-27b-256k, '
        'qwen3.8-27b-32k, qwen3.8-27b-q3-gguf, qwen3.8-27b-q4-gguf\n'
        'H100-80 (H100 80 GB) | 80 | sim | sim | gemma-4-12b-q4-gguf, gemma-4-26b-a4b-q4-gguf, '
        'gemma-4-31b-q4-gguf, gemma-4-e4b-q4-gguf, inkling-2-mini-iq2-gguf, '
        'inkling-2-mini-q4-gguf, inkling-2-mini-q8-gguf, muse-glimmer-30b-q2-gguf, '
        'muse-glimmer-30b-q4-gguf, qwen-image-2.1-fp8, qwen-image-2.1-q4km, qwen3.8-27b-256k, '
        'qwen3.8-27b-32k, qwen3.8-27b-q3-gguf, qwen3.8-27b-q4-gguf\n'
        '\n'
        'Fonte VRAM: Google Cloud, Compute Engine > GPU machine types > General comparison chart '
        '(docs.cloud.google.com/compute/docs/gpus), consultado em 2026-11-23\n'
        'Nota Colab: FAQ oficial do Colab (research.google.com/colaboratory/faq.html, consultado '
        'em 2026-11-23): tipos de GPU variam com o tempo; a FAQ não publica uma tabela de VRAM.\n'
    )
    assert res.output == esperado


def test_snapshot_cli_models_all_gpus_e_prompt():
    for gpu_id in ("T4", "L4", "A100-40", "A100-80", "H100-80"):
        res = runner.invoke(cli.app, ["models", "--gpu", gpu_id])
        assert res.exit_code == 0
        assert f"GPU: {gpu_id} " in res.output
        assert "Summary:" in res.output

    res_l4 = runner.invoke(cli.app, ["models", "--gpu", "L4"])
    assert "Summary: 11 compatible variant(s); 5 discarded for insufficient VRAM." in res_l4.output

    res_prompt = runner.invoke(cli.app, ["models"], input="2\n")
    assert res_prompt.exit_code == 0
    assert "Enter the number or ID of the GPU to filter models" in res_prompt.output
    assert "GPU: L4 (L4) — 24 GiB VRAM aproximada." in res_prompt.output


def test_snapshot_cli_configure():
    res_32k = runner.invoke(cli.app, ["configure", "--profile", "qwen3.8-27b-32k"])
    assert res_32k.exit_code == 0
    cfg_file = config.config_path()
    assert res_32k.output == (
        "Applied profile: qwen3.8-27b-32k — Qwen3.8 27B (32k) (runtime vllm)\n"
        "Contexto servido: 32,768 tokens; GPU: A100 (standard memory)\n"
        "Contexto documentado: 262,144 tokens (modelo qwen3.8-27b-awq:q4-awq); "
        "A100 40 GB sem high_mem: 32K limita o KV a 2 GiB.\n"
        "Reasoning padrão do modelo: xhigh\n"
        f"Configuration saved to {cfg_file}\n"
        "In OpenCode, select `byom/qwen3.8-27b-32k` to use the matching context limit\n"
    )

    res_img = runner.invoke(cli.app, ["configure", "--profile", "qwen-image-2.1-q4km"])
    assert res_img.exit_code == 0
    assert res_img.output == (
        "Applied profile: qwen-image-2.1-q4km — Qwen-Image-2.1 (GGUF Q4_K_M) (runtime image)\n"
        "Contexto servido: 1 tokens; GPU: T4 (standard memory)\n"
        "Contexto documentado: 0 tokens (modelo qwen-image-2.1:q4km); "
        "Modelo de difusão não tem janela autoregressiva; 1 é sentinela.\n"
        "Reasoning padrão do modelo: not documented\n"
        f"Configuration saved to {cfg_file}\n"
        "In OpenCode, select `byom/qwen-image-2.1-q4km` to use the matching context limit\n"
    )


def test_snapshot_cli_init(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/ssh")
    monkeypatch.setattr(cli.colab, "version", lambda: "Version: 0.7.5")
    monkeypatch.setattr(
        cli.colab,
        "run",
        lambda args, check=False: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    def fake_ssh_keygen(cmd, check=False):
        idx = cmd.index("-f")
        Path(cmd[idx + 1]).write_text("PRIVATE", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_ssh_keygen)

    res = runner.invoke(cli.app, ["init"])
    assert res.exit_code == 0
    cfg = config.load_config()
    assert res.output == (
        "Colab CLI: Version: 0.7.5\n"
        f"Created SSH key at {cfg.ssh_key_path}\n"
        f"Configuration saved to {config.config_path()}\n"
        "Ready\n"
    )


def test_snapshot_cli_status_e_down(monkeypatch):
    monkeypatch.setattr(colab, "orphan_assignments", lambda: [])
    monkeypatch.setattr(colab, "usage", lambda: "")

    res_status_empty = runner.invoke(cli.app, ["status"])
    assert res_status_empty.exit_code == 0
    assert res_status_empty.output == "No active session.\n"

    res_down_empty = runner.invoke(cli.app, ["down"])
    assert res_down_empty.exit_code == 0
    assert res_down_empty.output == "No active session.\n"

    state = config.State(
        session="byom-1234abcd",
        endpoint="m-s-abc",
        nonce="n" * 32,
        created_at="2026-11-23T19:00:00+00:00",
    )
    config.save_state(state)
    monkeypatch.setattr(
        colab,
        "status_line",
        lambda name: colab.SessionLine(
            name=name, endpoint="m-s-abc", hardware="L4", shape="standard", variant="GPU"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_daemon_status",
        lambda cfg, expected_session=None: {
            "session": "byom-1234abcd",
            "worker": {
                "state": "ready",
                "watchdog_silence_s": 5,
                "idle_s": 10,
                "idle_timeout_min": 20,
            },
            "tunnel": {"state": "connected", "reconnects": 0, "latency_ms": 42},
        },
    )

    res_status_active = runner.invoke(cli.app, ["status"])
    assert res_status_active.exit_code == 0
    assert (
        "Session: byom-1234abcd  (created at 2026-11-23T19:00:00+00:00)" in res_status_active.output
    )
    assert "Colab session: active (requested hardware: L4)" in res_status_active.output
    assert "Worker: ready" in res_status_active.output
    assert "Tunnel: connected (reconnects: 0, latency: 42 ms)" in res_status_active.output

    monkeypatch.setattr(colab, "stop", lambda name: None)
    monkeypatch.setattr(colab, "is_allocated", lambda name, endpoint="": False)
    res_down_active = runner.invoke(cli.app, ["down"])
    assert res_down_active.exit_code == 0
    assert (
        res_down_active.output == "Stopping session 'byom-1234abcd'\nStopped (confirmed by Colab)\n"
    )
