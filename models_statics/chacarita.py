"""Chacarita: el bot estático MÁS FÁCIL. Lógica mínima: casi no razona.

Su lógica (solo mira la jugada propia, ni siquiera lo que el rival podría hacer después):
  1. Si puede ganar la partida, la gana.
  2. Si puede ganar un tablero chico, lo gana (prefiere los de más valor).
  3. Si no, juega por posición (centro > esquinas > bordes) con mucha variedad.

Lo que NO hace: no bloquea al rival, no mira adónde lo manda, no arma nada. Esas limitaciones son de su lógica, no
jugadas malas a propósito: entre las opciones que su criterio ve parecidas, sortea (no es determinista).
"""
from __future__ import annotations

import math

from models_statics.base import StaticBot

WIN_COMBOS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
LINES_THROUGH = [[tuple(x for x in line if x != i) for line in WIN_COMBOS if i in line] for i in range(9)]
MACRO_W = [3, 2, 3, 2, 4, 2, 3, 2, 3]
CELL_W = [0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5]


def _completes(cells, idx, p) -> bool:
    for a, b in LINES_THROUGH[idx]:
        if cells[a] == p and cells[b] == p:
            return True
    return False


def _macro_wins(small, b, p) -> bool:
    for x, y in LINES_THROUGH[b]:
        if small[x] == p and small[y] == p:
            return True
    return False


class Chacarita(StaticBot):
    NAME = "Chacarita"
    COLOR1 = "#c8102e"     # rojo
    COLOR2 = "#111111"     # negro

    TEMP = 1.6             # alto: mucha variedad entre jugadas posicionalmente parecidas

    def choose(self, game) -> tuple[int, int]:
        me = 1 if game.current_player == 1 else -1
        board = [[int(v) for v in row] for row in game.board.tolist()]
        small = game.small_boards.tolist()
        active = None if game.active_board is None else int(game.active_board)
        boards = [active] if active is not None else [i for i in range(9) if small[i] == 0.5]

        take, rest = [], []
        for b in boards:
            cells = board[b]
            for c in range(9):
                if cells[c] != 0:
                    continue
                if _completes(cells, c, me):
                    if _macro_wins(small, b, me):
                        return b, c                                   # 1. ganar la partida
                    take.append((float(MACRO_W[b]), (b, c)))          # 2. ganar un tablero chico
                else:
                    rest.append((CELL_W[c] + 0.15 * MACRO_W[b], (b, c)))   # 3. posición
        return self._pick(take or rest)

    def _pick(self, pool):
        best = max(s for s, _ in pool)
        cand = sorted(pool, key=lambda t: t[1])
        weights = [math.exp((s - best) / self.TEMP) for s, _ in cand]
        r = self.rng.random() * sum(weights)
        acc = 0.0
        for w, (_, m) in zip(weights, cand):
            acc += w
            if r <= acc:
                return m
        return cand[-1][1]