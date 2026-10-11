"""Base de los bots ESTÁTICOS (jugadores fijos con lógica propia, sin red neuronal ni mutación).

Cada bot vive en su propio archivo dentro de ``models_statics/`` y declara una clase que hereda de
``StaticBot``. El NOMBRE se muestra pintado de a ``SEGMENT`` letras alternando dos colores:

    WALL-E  ->  "WA" (color 1) · "LL" (color 2) · "-E" (color 1)

``COLOR1`` es además la identidad del bot en la base de datos (única en toda la historia, como el
color de las redes); ``COLOR2`` solo se usa para pintar el nombre.

El azar de un bot sale SIEMPRE de ``self.rng``, que el torneo siembra por partida (``reseed``): así
el bot juega distinto en cada partida pero una partida interrumpida se repite idéntica al reanudar.
"""
from __future__ import annotations

import random
import re

SEGMENT = 2
_HEX = re.compile(r"^#[0-9a-f]{6}$")


def split_name(name: str, color1: str, color2: str, size: int = SEGMENT) -> list[tuple[str, str]]:
    """[(texto, color)] alternando color1/color2 de a ``size`` letras: WALL-E -> WA·LL·-E."""
    colors = (color1, color2)
    return [(name[i:i + size], colors[(i // size) % 2]) for i in range(0, len(name), size)]


class StaticBot:
    NAME: str = ""
    COLOR1: str = ""
    COLOR2: str = ""
    SEGMENT: int = SEGMENT

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # -- identidad / presentación ---------------------------------------
    @classmethod
    def segments(cls) -> list[tuple[str, str]]:
        """[(texto, color)] alternando COLOR1/COLOR2 de a ``SEGMENT`` letras."""
        return split_name(cls.NAME, cls.COLOR1, cls.COLOR2, cls.SEGMENT)

    @classmethod
    def validate(cls) -> None:
        if not cls.NAME.strip():
            raise ValueError(f"{cls.__name__}: falta NAME")
        for attr in ("COLOR1", "COLOR2"):
            v = getattr(cls, attr)
            if not _HEX.match(v):
                raise ValueError(f"{cls.__name__}.{attr}={v!r}: debe ser #rrggbb en minúsculas")
        if cls.COLOR1 == cls.COLOR2:
            raise ValueError(f"{cls.__name__}: COLOR1 y COLOR2 deben ser distintos")

    # -- juego -----------------------------------------------------------
    def reseed(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def choose(self, game) -> tuple[int, int]:      # pragma: no cover - lo implementa cada bot
        raise NotImplementedError
