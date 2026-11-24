"""Tabela de orçamento de VRAM por perfil (sonda de conferência).

Uso: uv run python scripts/vram_table.py
Lê os perfis TOML embarcados e imprime, para cada um, GPU, contexto, required,
folga e fits. Nenhum efeito colateral: só lê o catálogo e os perfis.
"""

from __future__ import annotations

import sys

from byom import catalog, config


def main() -> int:
    rows = []
    for profile in config.available_profiles():
        if not profile.catalog_model:
            continue
        gpu = catalog.resolve_gpu(
            str(profile.settings["gpu"]), bool(profile.settings.get("gpu_high_mem"))
        )
        context = int(profile.settings["max_model_len"])
        fit = catalog.check_fit(
            profile.catalog_model, profile.catalog_variant, gpu.id, context
        )
        variant = catalog.variant_by_id(profile.catalog_model, profile.catalog_variant)
        rows.append(
            (
                profile.id,
                gpu.id,
                context,
                variant.memory_gib,
                variant.vision_gib,
                fit.required_gib,
                fit.headroom_gib,
                gpu.vram_gib - fit.required_gib,
                fit.fits,
            )
        )
    header = (
        f"{'perfil':30s} {'gpu':8s} {'ctx':>7s} {'pesos':>6s} {'visao':>6s} "
        f"{'req':>6s} {'folga':>6s} {'sobra':>6s} fits"
    )
    print(header)
    for row in rows:
        profile_id, gpu_id, context, weights, vision, required, headroom, spare, fits = row
        print(
            f"{profile_id:30s} {gpu_id:8s} {context:7d} {weights:6.2f} {vision:6.2f} "
            f"{required:6.2f} {headroom:6.2f} {spare:6.2f} {'sim' if fits else 'NAO'}"
        )
    print(f"\n{len(rows)} perfis; {sum(1 for row in rows if not row[-1])} não cabem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
