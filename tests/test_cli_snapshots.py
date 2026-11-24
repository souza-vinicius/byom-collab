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
        "gemma-4-12b-q4-gguf — Gemma 4 12b-q4 (32,768 tokens servidos; reasoning not d"
        "ocumented)\n"
        "  Fornecedor: Google DeepMind\n"
        "  Pacote: Unsloth\n"
        "  gemma-4:12b-q4 — contexto doc 262,144 / servido 32,768\n"
        "  8 GiB de pesos em T4: 32K mantêm 2 GiB de KV.\n"
        "  Gemma 4 12B Unified 4 bit; thinking ligado por <|think|>.\n"
        "gemma-4-26b-a4b-q4-gguf — Gemma 4 26b-a4b-q4 (32,768 tokens servidos; reasoni"
        "ng not documented)\n"
        "  Fornecedor: Google DeepMind\n"
        "  Pacote: Unsloth\n"
        "  gemma-4:26b-a4b-q4 — contexto doc 262,144 / servido 32,768\n"
        "  18 GiB pesos + 1,11 visão + 2 runtime + 2 KV = 23,11, mais 1,2 de folga = 2"
        "4,31 > 24 GiB da L4; a A100-40 comporta 32K com 15,69 GiB de margem.\n"
        "  Gemma 4 26B-A4B 4 bit com mmproj de visão; A100 de 40 GB (o projetor de 1,1"
        "1 GiB tira a margem da L4).\n"
        "gemma-4-31b-q4-gguf — Gemma 4 31b-q4 (131,072 tokens servidos; reasoning not "
        "documented)\n"
        "  Fornecedor: Google DeepMind\n"
        "  Pacote: Unsloth\n"
        "  gemma-4:31b-q4 — contexto doc 262,144 / servido 131,072\n"
        "  20 GiB pesos + 0,65 visão + 2 runtime + 8 KV = 30,65 na A100-40, com 2 GiB "
        "de margem.\n"
        "  Gemma 4 31B 4 bit; requer A100.\n"
        "gemma-4-e4b-q4-gguf — Gemma 4 e4b-q4 (65,536 tokens servidos; reasoning not d"
        "ocumented)\n"
        "  Fornecedor: Google DeepMind\n"
        "  Pacote: Unsloth\n"
        "  gemma-4:e4b-q4 — contexto doc 131,072 / servido 65,536\n"
        "  6 GiB de pesos em T4: 65.536 tokens mantêm 4 GiB de KV.\n"
        "  Gemma 4 E4B 4 bit; thinking ligado por <|think|>.\n"
        "inkling-2-mini-ad-iq2xxs — Inkling 2 Mini (AD-IQ2_XXS, GGUF) (32,768 tokens s"
        "ervidos; reasoning high)\n"
        "  Fornecedor: Thinking Machines Lab\n"
        "  Pacote: AtomicChat\n"
        "  inkling-2-mini:ad-iq2xxs — contexto doc 1,000,000 / servido 32,768\n"
        "  9,71 GiB pesos + 0,65 visão + 2 runtime + 0,58 de KV em 32K = 12,94, mais 1"
        ",0 de folga = 13,94 <= 15 GiB da T4. O KV medido pela arquitetura (4 camadas "
        "completas, 20 com janela de 1024) é o que torna 32K possível: pelo heurístico"
        " seriam 15,36 e o perfil ficaria em 16K.\n"
        "  GGUF AD-IQ2_XXS da AtomicChat para T4: 88,5% de top-1 contra o BF16; com o "
        "KV medido pela arquitetura (0,58 GiB em 32K) os 32K cabem nos 15 GiB com 2,06"
        " GiB de sobra.\n"
        "inkling-2-mini-ad-q4k — Inkling 2 Mini (AD-Q4_K, GGUF) (32,768 tokens servido"
        "s; reasoning high)\n"
        "  Fornecedor: Thinking Machines Lab\n"
        "  Pacote: AtomicChat\n"
        "  inkling-2-mini:ad-q4k — contexto doc 1,000,000 / servido 32,768\n"
        "  15,39 GiB pesos + 0,65 visão + 2 runtime + 0,58 de KV em 32K = 18,62, mais "
        "1,2 de folga = 19,82 <= 24 GiB da L4 (5,38 de sobra). 32K é o contexto do exe"
        "mplo de execução do publicador (`-c 32768`).\n"
        "  GGUF AD-Q4_K da AtomicChat, o quant recomendado pelo publicador: 96,3% de t"
        "op-1 (qualidade de Q5_K_M com 3,3 GB menos); 32K cabe na L4.\n"
        "inkling-2-mini-ad-q6k — Inkling 2 Mini (AD-Q6_K, GGUF) (131,072 tokens servid"
        "os; reasoning high)\n"
        "  Fornecedor: Thinking Machines Lab\n"
        "  Pacote: AtomicChat\n"
        "  inkling-2-mini:ad-q6k — contexto doc 1,000,000 / servido 131,072\n"
        "  22,15 GiB pesos + 0,65 visão + 2 runtime + 2,08 de KV em 128K = 26,88, mais"
        " 2,0 de folga = 28,88 <= 40 GiB da A100-40. A L4 não comporta nem em 32K (25,"
        "38 + 1,2).\n"
        "  GGUF AD-Q6_K da AtomicChat: 97,6% de top-1 contra o BF16 (qualidade de Q8_0"
        " com 4 GB menos); 128K exige A100 de 40 GB.\n"
        "inkling-2-mini-iq2-gguf — Inkling 2 Mini (2 bit, GGUF) (32,768 tokens servido"
        "s; reasoning high)\n"
        "  Fornecedor: Thinking Machines Lab\n"
        "  Pacote: Unsloth\n"
        "  inkling-2-mini:q2 — contexto doc 1,000,000 / servido 32,768\n"
        "  7,90 GiB de pesos + 0,65 visão + 2 runtime + 0,58 de KV em 32K (medido pela"
        " arquitetura) = 11,13, mais 1,0 de folga = 12,13 <= 15 GiB utilizáveis da T4."
        "\n"
        "  2 bit para T4 (7,90 GiB de pesos + 0,65 de visão + 2 de runtime + 0,58 de K"
        "V em 32K = 11,13, com 1,0 de folga ainda sobram 3,87 GiB); build mainline b12"
        "611 com arquivos Mainline_Support.\n"
        "inkling-2-mini-q4-gguf — Inkling 2 Mini (4 bit, GGUF) (65,536 tokens servidos"
        "; reasoning high)\n"
        "  Fornecedor: Thinking Machines Lab\n"
        "  Pacote: Unsloth\n"
        "  inkling-2-mini:q4 — contexto doc 1,000,000 / servido 65,536\n"
        "  18 GiB pesos + 0,65 visão + 2 runtime + 1,08 de KV em 64K = 21,73, mais 1,2"
        " de folga = 22,93 <= 24 GiB da L4: a A100-40 deixou de ser necessária.\n"
        "  Quant recomendado UD-Q4_K_XL com visão; com o KV medido pela arquitetura (1"
        ",08 GiB em 64K) os 64K cabem na L4, com 1,07 GiB além da folga.\n"
        "inkling-2-mini-q8-gguf — Inkling 2 Mini (8 bit, GGUF) (131,072 tokens servido"
        "s; reasoning high)\n"
        "  Fornecedor: Thinking Machines Lab\n"
        "  Pacote: Unsloth\n"
        "  inkling-2-mini:q8 — contexto doc 1,000,000 / servido 131,072\n"
        "  24 GiB pesos + 0,65 visão + 2 runtime + 2,08 de KV em 128K = 28,73 na A100-"
        "40, com 11,27 GiB de margem; a L4 não comporta os pesos.\n"
        "  8 bit sem perdas; build mainline b12611 com arquivos Mainline_Support; requ"
        "er A100 de 40 GB ou mais.\n"
        "muse-glimmer-30b-q2-gguf — Muse Glimmer 30B (q2) (32,768 tokens servidos; rea"
        "soning high)\n"
        "  Fornecedor: Meta Superintelligence Lab\n"
        "  Pacote: Unsloth\n"
        "  muse-glimmer-30b:q2 — contexto doc 131,072 / servido 32,768\n"
        "  12 GiB pesos + 0,65 visão + 2 runtime + 2 KV = 16,65: a T4 de 15 GiB utiliz"
        "áveis não comporta.\n"
        "  Muse Glimmer 2 bit para L4 (a T4 não comporta o projetor de visão); build l"
        "lama.cpp fixado.\n"
        "muse-glimmer-30b-q4-gguf — Muse Glimmer 30B (q4) (65,536 tokens servidos; rea"
        "soning high)\n"
        "  Fornecedor: Meta Superintelligence Lab\n"
        "  Pacote: Unsloth\n"
        "  muse-glimmer-30b:q4 — contexto doc 131,072 / servido 65,536\n"
        "  17 GiB pesos + 0,65 visão + 2 runtime + 4 KV = 23,65: não cabe em L4 com a "
        "folga de 1,2 GiB.\n"
        "  Muse Glimmer 4 bit com visão; 64K exige A100 de 40 GB (L4 fica 1,2 GiB abai"
        "xo da folga).\n"
        "qwen-image-2.1-fp8 — Qwen-Image-2.1 (FP8/INT8-ConvRot) (1 tokens servidos; re"
        "asoning not documented)\n"
        "  Fornecedor: Qwen (Alibaba Cloud)\n"
        "  Pacote: Comfy-Org\n"
        "  qwen-image-2.1:fp8-int8 — contexto doc 0 / servido 1\n"
        "  Modelo de difusão não tem janela autoregressiva; 1 é sentinela. 12,07 GiB d"
        "e pesos + 2 de runtime + 1,0 de folga = 15,07 > 15 GiB da T4; a L4 (24 GiB) c"
        "omporta.\n"
        "  Pesos INT8-ConvRot para GPU de 24 GB+; endpoint de imagem, não chat.\n"
        "qwen-image-2.1-q4km — Qwen-Image-2.1 (GGUF Q4_K) (1 tokens servidos; reasonin"
        "g not documented)\n"
        "  Fornecedor: Qwen (Alibaba Cloud)\n"
        "  Pacote: leejet\n"
        "  qwen-image-2.1:q4km — contexto doc 0 / servido 1\n"
        "  Modelo de difusão não tem janela autoregressiva; 1 é sentinela. 9,22 GiB de"
        " pesos + 2 de runtime + 1,0 de folga = 12,22 <= 15 GiB utilizáveis da T4, ent"
        "ão a GPU barata serve.\n"
        "  Geração de imagens em T4 (15 GiB); POST /v1/images/generations, não chat.\n"
        "qwen3.8-27b-256k — Qwen3.8 27B (256k) (262,144 tokens servidos; reasoning xhi"
        "gh)\n"
        "  Fornecedor: Qwen (Alibaba Cloud)\n"
        "  Pacote: philbert440\n"
        "  qwen3.8-27b-awq:q4-awq — contexto doc 262,144 / servido 262,144\n"
        "  Contexto máximo documentado mantido em A100 80 GB (high_mem).\n"
        "  Full native context for a high-memory A100.\n"
        "qwen3.8-27b-32k — Qwen3.8 27B (32k) (32,768 tokens servidos; reasoning xhigh)"
        "\n"
        "  Fornecedor: Qwen (Alibaba Cloud)\n"
        "  Pacote: philbert440\n"
        "  qwen3.8-27b-awq:q4-awq — contexto doc 262,144 / servido 32,768\n"
        "  A100 40 GB sem high_mem: 32K limita o KV a 2 GiB.\n"
        "  Lower-context profile for an A100 without a high-memory requirement.\n"
        "qwen3.8-27b-q3-gguf — Qwen3.8 27B (3 bit, GGUF) (32,768 tokens servidos; reas"
        "oning xhigh)\n"
        "  Fornecedor: Qwen (Alibaba Cloud)\n"
        "  Pacote: Unsloth\n"
        "  qwen3.8-27b-gguf:q3 — contexto doc 262,144 / servido 32,768\n"
        "  12 GiB pesos + 2 runtime + 2 KV = 16 GiB: não cabe nos 15 GiB utilizáveis d"
        "a T4.\n"
        "  GGUF 3 bit para L4 (pesos + runtime + KV passam dos 15 GiB da T4); alternat"
        "iva ao AWQ em vLLM.\n"
        "qwen3.8-27b-q4-gguf — Qwen3.8 27B (4 bit, GGUF) (32,768 tokens servidos; reas"
        "oning xhigh)\n"
        "  Fornecedor: Qwen (Alibaba Cloud)\n"
        "  Pacote: Unsloth\n"
        "  qwen3.8-27b-gguf:q4 — contexto doc 262,144 / servido 32,768\n"
        "  17 GiB de pesos em L4: 32K mantém 2 GiB de KV.\n"
        "  GGUF 4 bit recomendado; requer L4 ou melhor.\n"    )
    assert res.output == esperado


def test_snapshot_cli_gpus():
    res = runner.invoke(cli.app, ["gpus"])
    assert res.exit_code == 0
    esperado = (
        "GPU | VRAM (GiB) | bf16 | VRAM aproximada | Perfis que cabem\n"
        "\n"
        "T4 (Tesla T4) | 15 | nao | sim | gemma-4-12b-q4-gguf, gemma-4-e4b-q4-gguf, in"
        "kling-2-mini-ad-iq2xxs, inkling-2-mini-iq2-gguf, qwen-image-2.1-q4km\n"
        "L4 (L4) | 24 | sim | sim | gemma-4-12b-q4-gguf, gemma-4-e4b-q4-gguf, inkling-"
        "2-mini-ad-iq2xxs, inkling-2-mini-ad-q4k, inkling-2-mini-iq2-gguf, inkling-2-m"
        "ini-q4-gguf, muse-glimmer-30b-q2-gguf, qwen-image-2.1-fp8, qwen-image-2.1-q4k"
        "m, qwen3.8-27b-32k, qwen3.8-27b-q3-gguf, qwen3.8-27b-q4-gguf\n"
        "A100-40 (A100 40 GB) | 40 | sim | sim | gemma-4-12b-q4-gguf, gemma-4-26b-a4b-"
        "q4-gguf, gemma-4-31b-q4-gguf, gemma-4-e4b-q4-gguf, inkling-2-mini-ad-iq2xxs, "
        "inkling-2-mini-ad-q4k, inkling-2-mini-ad-q6k, inkling-2-mini-iq2-gguf, inklin"
        "g-2-mini-q4-gguf, inkling-2-mini-q8-gguf, muse-glimmer-30b-q2-gguf, muse-glim"
        "mer-30b-q4-gguf, qwen-image-2.1-fp8, qwen-image-2.1-q4km, qwen3.8-27b-256k, q"
        "wen3.8-27b-32k, qwen3.8-27b-q3-gguf, qwen3.8-27b-q4-gguf\n"
        "A100-80 (A100 80 GB) | 80 | sim | sim | gemma-4-12b-q4-gguf, gemma-4-26b-a4b-"
        "q4-gguf, gemma-4-31b-q4-gguf, gemma-4-e4b-q4-gguf, inkling-2-mini-ad-iq2xxs, "
        "inkling-2-mini-ad-q4k, inkling-2-mini-ad-q6k, inkling-2-mini-iq2-gguf, inklin"
        "g-2-mini-q4-gguf, inkling-2-mini-q8-gguf, muse-glimmer-30b-q2-gguf, muse-glim"
        "mer-30b-q4-gguf, qwen-image-2.1-fp8, qwen-image-2.1-q4km, qwen3.8-27b-256k, q"
        "wen3.8-27b-32k, qwen3.8-27b-q3-gguf, qwen3.8-27b-q4-gguf\n"
        "H100-80 (H100 80 GB) | 80 | sim | sim | gemma-4-12b-q4-gguf, gemma-4-26b-a4b-"
        "q4-gguf, gemma-4-31b-q4-gguf, gemma-4-e4b-q4-gguf, inkling-2-mini-ad-iq2xxs, "
        "inkling-2-mini-ad-q4k, inkling-2-mini-ad-q6k, inkling-2-mini-iq2-gguf, inklin"
        "g-2-mini-q4-gguf, inkling-2-mini-q8-gguf, muse-glimmer-30b-q2-gguf, muse-glim"
        "mer-30b-q4-gguf, qwen-image-2.1-fp8, qwen-image-2.1-q4km, qwen3.8-27b-256k, q"
        "wen3.8-27b-32k, qwen3.8-27b-q3-gguf, qwen3.8-27b-q4-gguf\n"
        "\n"
        "Fonte VRAM: Google Cloud, Compute Engine > GPU machine types > General compar"
        "ison chart (docs.cloud.google.com/compute/docs/gpus), consultado em 2026-11-2"
        "3\n"
        "Nota Colab: FAQ oficial do Colab (research.google.com/colaboratory/faq.html, "
        "consultado em 2026-11-23): tipos de GPU variam com o tempo; a FAQ não publica"
        " uma tabela de VRAM.\n"    )
    assert res.output == esperado


def test_snapshot_cli_models_all_gpus_e_prompt():
    for gpu_id in ("T4", "L4", "A100-40", "A100-80", "H100-80"):
        res = runner.invoke(cli.app, ["models", "--gpu", gpu_id])
        assert res.exit_code == 0
        assert f"GPU: {gpu_id} " in res.output
        assert "Summary:" in res.output

    res_l4 = runner.invoke(cli.app, ["models", "--gpu", "L4"])
    # O q4 do Inkling passou a caber na L4 com o KV medido pela arquitetura; as
    # três variantes do Bonsai 2.5 também cabem (a T4 é a GPU que elas exigem).
    assert "Summary: 16 compatible variant(s); 6 discarded for insufficient VRAM." in (
        res_l4.output
    )

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
        "Applied profile: qwen-image-2.1-q4km — Qwen-Image-2.1 (GGUF Q4_K) (runtime im"
        "age)\n"
        "Contexto servido: 1 tokens; GPU: T4 (standard memory)\n"
        "Contexto documentado: 0 tokens (modelo qwen-image-2.1:q4km); Modelo de difusã"
        "o não tem janela autoregressiva; 1 é sentinela. 9,22 GiB de pesos + 2 de runt"
        "ime + 1,0 de folga = 12,22 <= 15 GiB utilizáveis da T4, então a GPU barata se"
        "rve.\n"
        "Reasoning padrão do modelo: not documented\n"
        f"Configuration saved to {cfg_file}\n"
        "In OpenCode, select `byom/qwen-image-2.1-q4km` to use the matching context li"
        "mit\n"    )


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
    assert res_down_active.output == (
        "Stopping session 'byom-1234abcd'\n"
        "Warning: nothing recorded to restore; the harness default may still point to this "
        "byom endpoint (run `byom up` then `byom down`, or fix it by hand)\n"
        "Stopped (confirmed by Colab)\n"
    )
