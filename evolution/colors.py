"""Identidad por color hexadecimal (#rrggbb): generación, mutación y unicidad.

El color es una etiqueta visual de linaje, NO una medida de calidad ni de
semejanza genética. El linaje real se guarda con una referencia explícita al padre.
"""
from __future__ import annotations

import colorsys
import random
import re
from dataclasses import dataclass, field
from typing import Iterable

HEX_RE = re.compile(r"^#[0-9a-f]{6}$")


def is_valid_hex(c: str) -> bool:
    return bool(HEX_RE.match(c))


def to_rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def to_hex(rgb: Iterable[int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass(frozen=True)
class ColorTier:
    """Magnitud de la variación por canal (|delta| en [min_delta, max_delta], 0-255)
    y cuántos canales se alteran."""
    min_delta: int
    max_delta: int
    min_channels: int = 1
    max_channels: int = 3


DEFAULT_TIERS = {
    "small": ColorTier(1, 3, 1, 3),      # #ff031f -> #ff041f
    "medium": ColorTier(4, 24, 1, 3),
    "large": ColorTier(25, 80, 1, 3),    # #ff031f -> #af141f
}
TIER_ORDER = ["small", "medium", "large"]


@dataclass
class ColorResult:
    color: str
    meta: dict = field(default_factory=dict)


def random_color(rng: random.Random, taken: set[str]) -> ColorResult:
    """Color inicial (fundadores sin padre): uniforme, distinto de los existentes."""
    for attempt in range(1, 10_000):
        c = to_hex(rng.randrange(256) for _ in range(3))
        if c not in taken:
            return ColorResult(c, {"strategy": "random", "attempts": attempt})
    return _scan_free(rng.choice(sorted(taken)) if taken else "#808080", taken, "random_exhausted")


def spread_colors(n: int, rng: random.Random, taken: set[str]) -> list[ColorResult]:
    """``n`` colores de la población inicial repartidos por TODO el círculo cromático.

    Cada color tiene un matiz propio (separados 360/n grados, con un pequeño desvío) y
    saturación/brillo altos, así las IA de la primera generación se distinguen a simple
    vista y ninguna es demasiado oscura para verse en una terminal con fondo negro.
    Los descendientes luego se parecen a su padre (``mutate_color``): cada familia conserva su tono.
    """
    offset = rng.random()
    used = set(taken)
    out: list[ColorResult] = []
    for k in range(n):
        for attempt in range(1, 1000):
            hue = (offset + (k + rng.uniform(-0.2, 0.2)) / n) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, rng.uniform(0.55, 0.95), rng.uniform(0.70, 1.0))
            c = to_hex(round(v * 255) for v in (r, g, b))
            if c not in used:
                used.add(c)
                out.append(ColorResult(c, {"strategy": "spread", "slot": k, "of": n, "attempts": attempt}))
                break
        else:  # no debería ocurrir: espacio casi agotado
            out.append(random_color(rng, used))
            used.add(out[-1].color)
    return out


def _try_variation(parent: tuple[int, int, int], tier: ColorTier, rng: random.Random):
    k = rng.randint(tier.min_channels, tier.max_channels)
    channels = rng.sample(range(3), k)
    child = list(parent)
    deltas = {}
    for ch in channels:
        mag = rng.randint(tier.min_delta, tier.max_delta)
        sign = rng.choice((-1, 1))
        new = child[ch] + sign * mag
        if not 0 <= new <= 255:           # si se sale del rango, invertir el signo
            new = child[ch] - sign * mag
            sign = -sign
        new = min(255, max(0, new))
        if new != child[ch]:
            deltas["rgb"[ch]] = new - child[ch]
            child[ch] = new
    return tuple(child), deltas


def _scan_free(parent_hex: str, taken: set[str], strategy: str) -> ColorResult:
    """Estrategia de reserva: busca el color libre más cercano al padre recorriendo
    cubos RGB concéntricos (distancia de Chebyshev creciente) en orden fijo.
    Es determinista y solo falla si TODO el espacio de 16,7 M de colores está ocupado."""
    pr, pg, pb = to_rgb(parent_hex)
    for radius in range(1, 256):
        for dr in range(-radius, radius + 1):
            for dg in range(-radius, radius + 1):
                for db in range(-radius, radius + 1):
                    if max(abs(dr), abs(dg), abs(db)) != radius:
                        continue
                    r, g, b = pr + dr, pg + dg, pb + db
                    if not (0 <= r < 256 and 0 <= g < 256 and 0 <= b < 256):
                        continue
                    c = to_hex((r, g, b))
                    if c not in taken:
                        return ColorResult(c, {"strategy": strategy, "radius": radius})
    raise RuntimeError("Espacio de colores RGB agotado: no hay identificadores libres")


def mutate_color(
    parent_hex: str,
    level: str,
    rng: random.Random,
    taken: set[str],
    tiers: dict[str, ColorTier] | None = None,
    max_attempts: int = 200,
) -> ColorResult:
    """Variación del color del padre, con magnitud según ``level``.

    Si colisiona, reintenta; tras ``max_attempts`` fallos escala a la magnitud
    siguiente; si aún así no hay hueco, usa la búsqueda de reserva (``_scan_free``).
    """
    tiers = tiers or DEFAULT_TIERS
    parent = to_rgb(parent_hex)
    start = TIER_ORDER.index(level)
    attempts_total = 0
    for tier_name in TIER_ORDER[start:]:
        for _ in range(max_attempts):
            attempts_total += 1
            child, deltas = _try_variation(parent, tiers[tier_name], rng)
            c = to_hex(child)
            if c != parent_hex and c not in taken:
                return ColorResult(c, {
                    "strategy": "variation", "tier": tier_name, "requested_level": level,
                    "deltas": deltas, "attempts": attempts_total,
                    "escalated": tier_name != level,
                })
    res = _scan_free(parent_hex, taken, "nearest_free_scan")
    res.meta.update({"requested_level": level, "attempts": attempts_total})
    return res