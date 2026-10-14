"""Entrada canónica y simetrías para entrenar con puzzles (solo numpy, sin torch).

Por qué existe
--------------
``game.get_state()`` codifica el tablero en términos ABSOLUTOS (X = 1, O = -1). Con eso la red
tiene que aprender cada patrón táctico dos veces (una como X y otra como O) y, con pocos puzzles,
termina memorizándolos: en entrenamiento acierta ~99 % y en puzzles nuevos ~62 %.

Dos arreglos, ambos acá:

1. **Perspectiva de quien mueve** (``canonical_batch``): se multiplica el tablero por +1 (mueve X)
   o -1 (mueve O) y lo mismo con los tableros chicos ganados, de modo que "mis fichas" son siempre
   +1 y las del rival -1. El 0.5 de "tablero en juego" no se toca.
2. **Simetrías D4** (``augment``): rotar/reflejar el tablero con la MISMA transformación en el
   tablero grande y dentro de cada casilla conserva las reglas (la casilla ``c`` jugada manda al
   tablero ``c``, el historial de tableros también se transforma). Son 8 versiones gratis de cada puzzle.

El quién-mueve se toma de la paridad del turno (``state[99] * 81``): mueve X con turno par.
En posiciones reales coincide con la suma de las fichas (#X - #O = turno % 2).

COMPATIBILIDAD: la perspectiva canónica la aplica ``SuperTatetiNet.forward`` (``model.py``), así que
al jugar/entrenar se pasa el estado CRUDO (``game.get_state()``) y la red lo transforma sola. Acá
``canonical_batch``/``canonical_state`` quedan como referencia y para tests; ``augment`` (D4) sigue
usándose en el entrenamiento de puzzles.
"""
from __future__ import annotations

import numpy as np

INPUT_ENCODING = "canonical"


# ----------------------------------------------------------------------
# Perspectiva
# ----------------------------------------------------------------------
def mover_sign(states: np.ndarray) -> np.ndarray:
    """+1.0 si mueve X (turno par), -1.0 si mueve O. ``states``: (B, 100)."""
    turns = np.rint(states[:, 99] * 81.0).astype(np.int64)
    return np.where(turns % 2 == 0, 1.0, -1.0).astype(np.float32)


def canonical_batch(states: np.ndarray) -> np.ndarray:
    """Copia del lote con todo visto desde quien mueve (mis fichas = +1). (B, 100) -> (B, 100)."""
    states = np.asarray(states, dtype=np.float32)
    me = mover_sign(states)[:, None]
    out = states.copy()
    out[:, 0:81] *= me
    small = states[:, 81:90]
    out[:, 81:90] = np.where(np.abs(small) == 1.0, small * me, small)   # solo cambia lo ganado
    return out


def canonical_state(state: np.ndarray) -> np.ndarray:
    """Versión de un solo estado (100,) para usar al jugar con una red entrenada con puzzles."""
    return canonical_batch(np.asarray(state, dtype=np.float32)[None, :])[0]


# ----------------------------------------------------------------------
# Simetrías D4
# ----------------------------------------------------------------------
def _gather_maps() -> list[np.ndarray]:
    """8 mapas ``r`` de 9 elementos: la posición nueva ``j`` toma lo que había en ``r[j]``."""
    base = np.arange(9).reshape(3, 3)
    maps = []
    for k in range(4):
        rot = np.rot90(base, k)
        maps.append(rot.flatten())
        maps.append(np.fliplr(rot).flatten())
    return maps


SYM_MAPS: list[np.ndarray] = _gather_maps()
# Índices de 81: la jugada nueva (b', c') = (r[b'], r[c']) de antes (misma r en ambos niveles).
IDX81: list[np.ndarray] = [np.array([r[b] * 9 + r[c] for b in range(9) for c in range(9)]) for r in SYM_MAPS]
IDX9: list[np.ndarray] = [np.asarray(r) for r in SYM_MAPS]
INV9: list[np.ndarray] = [np.argsort(r) for r in SYM_MAPS]       # tablero viejo -> posición nueva


def apply_sym(states: np.ndarray, masks: np.ndarray, targets: np.ndarray, k: int):
    """Aplica la simetría ``k`` (0..7; 0 = identidad) a lotes de estados, máscaras y objetivos."""
    s = states.copy()
    s[:, 0:81] = states[:, 0:81][:, IDX81[k]]
    s[:, 81:90] = states[:, 81:90][:, IDX9[k]]
    s[:, 90:99] = states[:, 90:99][:, IDX9[k]]
    return s, masks[:, IDX81[k]], targets[:, IDX81[k]]


def augment(states: np.ndarray, masks: np.ndarray, targets: np.ndarray, rng: np.random.Generator):
    """Una simetría D4 al azar POR muestra. Devuelve copias (el original no se toca)."""
    ks = rng.integers(0, len(SYM_MAPS), size=len(states))
    s, m, t = states.copy(), masks.copy(), targets.copy()
    for k in np.unique(ks):
        sel = np.nonzero(ks == k)[0]
        s[sel], m[sel], t[sel] = apply_sym(states[sel], masks[sel], targets[sel], int(k))
    return s, m, t