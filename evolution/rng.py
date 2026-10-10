"""Semillas derivadas: todo el azar del torneo sale de la semilla de la temporada."""
from __future__ import annotations

import hashlib
import random


def derive_seed(base: int, *labels: object) -> int:
    """Semilla de 63 bits derivada de ``base`` y etiquetas. Estable entre procesos/plataformas."""
    h = hashlib.sha256(("|".join([str(base), *map(str, labels)])).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def derive_rng(base: int, *labels: object) -> random.Random:
    return random.Random(derive_seed(base, *labels))
