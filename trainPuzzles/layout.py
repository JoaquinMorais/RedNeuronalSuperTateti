"""Layout de checkpoints de ``trainPuzzles`` (ordenamiento de archivos).

Reglas:

* El **canon** (``puzzle_canon.pth`` + ``puzzle_canon_report.json``) vive en la raíz
  del directorio de checkpoints y guarda **siempre el último modelo terminado**, para
  que un entrenamiento nuevo reanude desde ahí (handoff).
* Cada corrida archiva sus artefactos en ``<checkpoints>/runs/NNNN_<tag>/``:
  el modelo final, su reporte y el mejor estado por nivel en ``levels/``.

No depende de ``evolution/`` ni de ningún otro módulo del repo.
"""
from __future__ import annotations

import re
from pathlib import Path

CANON_STEM = "puzzle_canon"
CANON_PTH = f"{CANON_STEM}.pth"
CANON_REPORT = f"{CANON_STEM}_report.json"
RUNS_DIRNAME = "runs"
LEVELS_DIRNAME = "levels"

_RUN_RE = re.compile(r"^(\d+)_")


def sanitize_tag(name: str | None) -> str:
    """Normaliza un nombre de corrida a algo seguro para nombre de carpeta."""
    tag = (name or "puzzle").strip()
    tag = re.sub(r"[^0-9A-Za-z._-]+", "-", tag).strip("-._")
    return tag or "puzzle"


def canon_paths(out: Path) -> tuple[Path, Path]:
    """Rutas ``(modelo, reporte)`` del canon, derivadas del directorio de ``out``.

    Si ``out`` ya apunta a ``puzzle_canon.pth`` (default), el canon coincide con
    ``out``. Un ``--out`` alternativo sigue teniendo su canon en la misma carpeta.
    """
    base = Path(out).parent
    return base / CANON_PTH, base / CANON_REPORT


def runs_root(out: Path) -> Path:
    """Directorio ``runs/`` que contiene las corridas archivadas."""
    return Path(out).parent / RUNS_DIRNAME


def next_run_dir(out: Path, tag: str) -> Path:
    """Devuelve ``runs/NNNN_<tag>`` con el siguiente número secuencial.

    Escanea los subdirectorios de ``runs/`` cuyo nombre empieza con dígitos y toma
    el mayor + 1. Si no hay ninguno, arranca en ``0001``.
    """
    root = runs_root(out)
    n = 0
    if root.is_dir():
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            m = _RUN_RE.match(entry.name)
            if m:
                n = max(n, int(m.group(1)))
    return root / f"{n + 1:04d}_{sanitize_tag(tag)}"


def run_paths(run_dir: Path, tag: str) -> tuple[Path, Path, Path]:
    """Rutas ``(modelo_final, reporte, levels_dir)`` dentro de una corrida."""
    tag = sanitize_tag(tag)
    return run_dir / f"{tag}.pth", run_dir / f"{tag}_report.json", run_dir / LEVELS_DIRNAME
