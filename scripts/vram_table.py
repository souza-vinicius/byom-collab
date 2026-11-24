"""Tabela de orçamento de VRAM por perfil (sonda de conferência).

Uso: uv run python scripts/vram_table.py
Lê os perfis TOML embarcados e imprime, para cada um, GPU, contexto, required,
folga e fits. Nenhum efeito colateral: só lê o catálogo e os perfis.

A coluna `origem` diz de onde saiu o KV: `arq` quando o modelo declara
`KvProfile` (conta pela arquitetura, com fonte datada em `catalog.py`) e `heur`
quando vale o heurístico de 2 GiB por 32K.
"""

from __future__ import annotations

import sys

from byom import catalog, config


def main() -> int:
    rows = []
    for profile in config.available_profiles():
        if not profile.catalog_model:
            continue
        entry = catalog.model_by_id(profile.catalog_model)
        gpu = catalog.resolve_gpu(
            str(profile.settings["gpu"]), bool(profile.settings.get("gpu_high_mem"))
        )
        context = int(profile.settings["max_model_len"])
        drafter = catalog.drafter_gib(profile.worker)
        fit = catalog.check_fit(
            profile.catalog_model,
            profile.catalog_variant,
            gpu.id,
            context,
            kv_cache_dtype=profile.settings.get("kv_cache_dtype"),
            drafter_gib=drafter,
        )
        variant = catalog.variant_by_id(profile.catalog_model, profile.catalog_variant)
        kv_dtype = catalog.effective_kv_dtype(
            profile.runtime, profile.settings.get("kv_cache_dtype")
        )
        kv_gib = (
            0.0
            if profile.runtime == "image"
            else catalog.kv_cache_gib(
                context,
                entry,
                kv_cache_dtype=profile.settings.get("kv_cache_dtype"),
                runtime=profile.runtime,
            )
        )
        rows.append(
            (
                profile.id,
                gpu.id,
                context,
                variant.memory_gib,
                variant.vision_gib,
                kv_gib,
                fit.required_gib,
                kv_dtype,
                fit.headroom_gib,
                gpu.vram_gib - fit.required_gib,
                fit.fits,
                "arq" if entry.kv_shape is not None else "heur",
                drafter,
            )
        )
    header = (
        f"{'perfil':30s} {'gpu':8s} {'ctx':>7s} {'pesos':>6s} {'visao':>6s} "
        f"{'kvGiB':>5s} {'req':>6s} {'dt':>4s} {'folga':>6s} {'sobra':>6s} "
        f"fits origem draft"
    )
    print(header)
    for row in rows:
        (profile_id, gpu_id, context, weights, vision, kv_gib, required, kv_dtype,
         headroom, spare, fits, origin, drafter_gib) = row
        print(
            f"{profile_id:30s} {gpu_id:8s} {context:7d} {weights:6.2f} {vision:6.2f} "
            f"{kv_gib:5.2f} {required:6.2f} {kv_dtype or '-':>4s} {headroom:6.2f} "
            f"{spare:6.2f} {'sim' if fits else 'NAO':4s} {origin} {drafter_gib:5.2f}"
        )
    print(f"\n{len(rows)} perfis; {sum(1 for row in rows if not row[10])} não cabem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
