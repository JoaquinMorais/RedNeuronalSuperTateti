"""Registro de bots estáticos: descubre ``models_statics/*.py`` (el nombre del archivo puede tener guiones,
p. ej. ``wall-e.py``, por eso se cargan por ruta y no con ``import``)."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from .base import StaticBot

STATICS_DIR = Path(__file__).resolve().parent


def slug_of(path: Path) -> str:
    return path.stem.lower()


def _load_file(path: Path) -> type[StaticBot]:
    mod_name = "models_statics._dyn_" + slug_of(path).replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found = [c for _, c in inspect.getmembers(mod, inspect.isclass)
             if issubclass(c, StaticBot) and c is not StaticBot and c.__module__ == mod_name]
    if len(found) != 1:
        raise ValueError(f"{path.name}: debe definir exactamente una clase que herede de StaticBot (hay {len(found)})")
    found[0].validate()
    return found[0]


def discover(directory: Path | None = None) -> dict[str, type[StaticBot]]:
    """{slug: clase} de todos los bots válidos."""
    d = directory or STATICS_DIR
    out: dict[str, type[StaticBot]] = {}
    for p in sorted(d.glob("*.py")):
        if p.name.startswith("_") or p.stem == "base":
            continue
        out[slug_of(p)] = _load_file(p)
    return out


def get(slug: str, directory: Path | None = None) -> type[StaticBot]:
    bots = discover(directory)
    if slug.lower() not in bots:
        raise KeyError(f"No hay ningún bot estático '{slug}' en {directory or STATICS_DIR} (disponibles: {', '.join(bots) or 'ninguno'})")
    return bots[slug.lower()]
