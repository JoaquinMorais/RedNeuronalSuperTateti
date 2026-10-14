"""Talleres: bot estático de nivel "pasable". Es el escalón más bajo: razona poco a propósito.

Su lógica (solo mira la jugada propia, nunca más de una jugada por delante):
  1. Si puede ganar la partida, la gana.
  2. No le regala la partida al rival (no lo manda a un tablero donde la gana de inmediato), si hay otra opción.
  3. Si puede ganar un tablero chico, lo gana (prefiere los de más valor: centro > esquinas > bordes).
  4. Si el rival está por ganar un tablero chico, lo bloquea.
  5. Si no, juega por posición: casilla central > esquinas > bordes, con preferencia suave por tableros importantes.

Lo que NO hace (por eso es fácil de superar): no calcula adónde manda al rival más allá del punto 2, no arma
amenazas macro, no hace horquillas ni mira jugadas futuras. Cuando varias jugadas valen casi lo mismo, sortea
entre ellas (softmax): no es determinista, pero nunca elige a propósito algo peor que lo que su lógica ve.
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


def _next_active(small, sq, c):
    """Tablero obligatorio tras jugar en la casilla c (regla de este juego: destino cerrado -> último abierto del historial)."""
    if small[c] == 0.5:
        return c
    for x in reversed(sq):
        if small[x] == 0.5:
            return x
    return None


class Talleres(StaticBot):
    NAME = "Talleres"
    COLOR1 = "#2f6fe4"     # azul
    COLOR2 = "#ffffff"     # blanco

    TEMP = 0.7             # más alto = más variedad entre jugadas parecidas

    def choose(self, game) -> tuple[int, int]:
        me = 1 if game.current_player == 1 else -1
        board = [[int(v) for v in row] for row in game.board.tolist()]
        small = game.small_boards.tolist()
        sq = [int(x) for x in game.squares_played]
        active = None if game.active_board is None else int(game.active_board)
        boards = [active] if active is not None else [i for i in range(9) if small[i] == 0.5]
        legal = [(b, c) for b in boards for c in range(9) if board[b][c] == 0]

        take, block, rest = [], [], []
        for b, c in legal:
            cells = board[b]
            if _completes(cells, c, me):
                if _macro_wins(small, b, me):
                    return b, c                                   # 1. ganar la partida
                take.append((float(MACRO_W[b]), (b, c)))          # 3. ganar un tablero chico
            elif _completes(cells, c, -me):
                block.append((float(MACRO_W[b]) + (10.0 if _macro_wins(small, b, -me) else 0.0), (b, c)))   # 4. bloquear
            else:
                rest.append((2.0 * CELL_W[c] + 0.25 * MACRO_W[b], (b, c)))                                  # 5. posición

        # 2. no regalar la partida: descartar jugadas que mandan al rival a un tablero donde la gana de inmediato
        def safe(move):
            b, c = move
            cells = board[b]
            ns = small
            if _completes(cells, c, me) or cells.count(0) == 1:   # el tablero b se cierra (ganado o lleno)
                ns = small[:]
                ns[b] = me if _completes(cells, c, me) else 0
            tgt = _next_active(ns, sq if (sq and sq[-1] == b) else sq + [b], c)
            targets = [tgt] if tgt is not None else [i for i in range(9) if ns[i] == 0.5]
            for t in targets:
                row = board[t]
                for cc in range(9):
                    if row[cc] == 0 and not (t == b and cc == c) and _completes(row, cc, -me) and _macro_wins(ns, t, -me):
                        return False
            return True

        for tier in (take, block, rest):
            if not tier:
                continue
            ok = [t for t in tier if safe(t[1])]
            pool = ok or tier
            return self._pick(pool)
        return legal[0]

    def _pick(self, pool):
        best = max(s for s, _ in pool)
        cand = sorted(((s, m) for s, m in pool if s >= best - 1.5), key=lambda t: t[1])
        weights = [math.exp((s - best) / self.TEMP) for s, _ in cand]
        r = self.rng.random() * sum(weights)
        acc = 0.0
        for w, (_, m) in zip(weights, cand):
            acc += w
            if r <= acc:
                return m
        return cand[-1][1]