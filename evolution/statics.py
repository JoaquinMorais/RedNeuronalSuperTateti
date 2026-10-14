"""Bots ESTÁTICOS (carpeta ``models_statics/``): jugadores con lógica propia, sin red ni mutación.

Compiten en las temporadas como las redes, pero no se multiplican: añaden cupos extra y siguen en
la temporada siguiente mientras terminen dentro de los primeros ``7 + (nº de estáticos)`` puestos.
Se muestran por su NOMBRE (pintado en dos colores), nunca por un código de color.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import models_statics
from models_statics.base import StaticBot, split_name

STATIC_TYPE, NEURAL_TYPE = "estatico", "neuronal"
TYPE_LABEL = {NEURAL_TYPE: "red", STATIC_TYPE: "estático"}


class StaticError(Exception):
    """Error que se le muestra a la persona."""


def normalize(name: str) -> str:
    """'Wall_E' == 'WALL-E' == 'wall e': solo letras y dígitos, en minúsculas."""
    return re.sub(r"[^0-9a-z]", "", name.lower())


def _scan(directory: Path) -> tuple[dict, dict]:
    """Carga cada archivo por separado: un bot con errores no impide usar los demás."""
    bots: dict[str, type[StaticBot]] = {}
    errors: dict[str, str] = {}
    d = Path(directory)
    if not d.exists():
        return bots, errors
    for p in sorted(d.glob("*.py")):
        if p.name.startswith("_") or p.stem == "base":
            continue
        slug = models_statics.slug_of(p)
        try:
            bots[slug] = models_statics._load_file(p)
        except Exception as e:  # noqa: BLE001  (sintaxis, falta de clase, colores inválidos, ...)
            errors[slug] = f"{type(e).__name__}: {e}"
    return bots, errors


def available(directory: Path) -> dict[str, type[StaticBot]]:
    """{slug: clase} de los bots VÁLIDOS del directorio."""
    return _scan(directory)[0]


def broken(directory: Path) -> dict[str, str]:
    """{slug: error} de los archivos que no se pudieron cargar."""
    return _scan(directory)[1]


def load_one(directory: Path, slug: str) -> type[StaticBot] | None:
    """Clase del bot ``slug`` (carga solo ese archivo); None si no existe. Lanza StaticError si tiene errores."""
    path = Path(directory) / f"{slug}.py"
    if not path.exists():
        return None
    try:
        return models_statics._load_file(path)
    except Exception as e:  # noqa: BLE001
        raise StaticError(f"El archivo {path.name} tiene errores: {type(e).__name__}: {e}") from e


def find(directory: Path, text: str) -> tuple[str, type[StaticBot]]:
    """Busca un bot por NOMBRE (o por nombre de archivo), sin distinguir mayúsculas ni '-'/'_'."""
    bots, errors = _scan(directory)
    key = normalize(text)
    hits = [(slug, c) for slug, c in bots.items() if key in (normalize(c.NAME), normalize(slug))]
    if not hits:
        names = ", ".join(c.NAME for c in bots.values()) or "ninguno"
        bad = "; ".join(f"{slug}.py: {err}" for slug, err in errors.items())
        raise StaticError(f"No hay ningún bot estático llamado '{text}' en {directory} (disponibles: {names})."
                          + (f" Archivos con errores: {bad}" if bad else ""))
    if len(hits) > 1:
        raise StaticError(f"'{text}' coincide con varios bots estáticos: {', '.join(c.NAME for _, c in hits)}.")
    return hits[0]


def check_unique(directory: Path) -> None:
    """Dos bots no pueden tener el mismo nombre ni el mismo COLOR1 (es su identidad)."""
    seen_names, seen_colors = {}, {}
    for slug, c in available(directory).items():
        for table, key, what in ((seen_names, normalize(c.NAME), "nombre"), (seen_colors, c.COLOR1, "COLOR1")):
            if key in table:
                raise StaticError(f"Los bots '{table[key]}' y '{slug}' repiten el {what}.")
            table[key] = slug


def code_hash(directory: Path, slug: str) -> str:
    return hashlib.sha256((Path(directory) / f"{slug}.py").read_bytes()).hexdigest()


def make_player(cls: type[StaticBot], seed: int = 0) -> StaticBot:
    return cls(seed=seed)


def plain_name(name: str) -> str:
    return name


def segments(name: str, color1: str, color2: str) -> list[tuple[str, str]]:
    return split_name(name, color1, color2)