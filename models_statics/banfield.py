"""Banfield: bot estático de nivel medio (entre RiverPt y WALL-E).

Razona una jugada hacia adelante, pero sin las capas finas de WALL-E:
  1. Gana la partida si puede y nunca se la regala al rival (mira su mejor respuesta, incluida la regla de este
     juego: destino cerrado => el rival vuelve al último tablero abierto del historial).
  2. Puntaje = valor táctico propio (ganar/bloquear tableros según su peso macro) + amenazas macro propias
     MENOS la mejor respuesta táctica del rival.
Lo que NO hace (a diferencia de WALL-E/BocaJrs): no valora potencial macro ni horquillas, no mira una tercera jugada
y no tiene análisis de final. Sortea con softmax (bastante abierto) entre jugadas parecidas: no es determinista.
"""
from __future__ import annotations

import math

from models_statics.base import StaticBot

WIN_COMBOS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
LINES_THROUGH = [[tuple(x for x in line if x != i) for line in WIN_COMBOS if i in line] for i in range(9)]
MACRO_W = [3, 2, 3, 2, 4, 2, 3, 2, 3]
CELL_W = [0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5]
WIN = 1000.0
LOSS_GUARD = -400.0


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


def _macro_threats(small, p) -> int:
    n = 0
    for a, b, c in WIN_COMBOS:
        line = (small[a], small[b], small[c])
        if line.count(p) == 2 and line.count(0.5) == 1:
            n += 1
    return n


def _legal(board, small, active):
    boards = (active,) if active is not None else [i for i in range(9) if small[i] == 0.5]
    return [(b, c) for b in boards for c in range(9) if board[b][c] == 0]


def _after(board, small, sq, b, c, p):
    """(small', active') tras jugar p en (b, c); winner None/±1/0. Solo lo necesario para mirar la respuesta del rival."""
    cells = board[b]
    won = _completes(cells, c, p)
    ns = small
    if won:
        if _macro_wins(small, b, p):
            return small, None, p
        ns = small[:]
        ns[b] = p
    elif cells.count(0) == 1:
        ns = small[:]
        ns[b] = 0
    if 0.5 not in ns:
        return ns, None, 0
    nsq = sq if (sq and sq[-1] == b) else sq + [b]
    if ns[c] == 0.5:
        return ns, c, None
    for x in reversed(nsq):
        if ns[x] == 0.5:
            return ns, x, None
    return ns, None, None


def _gain(board, small, b, c, p) -> float:
    cells = board[b]
    g = CELL_W[c] + (0.5 if b == 4 else 0.0)
    if _completes(cells, c, p):
        if _macro_wins(small, b, p):
            return WIN
        g += 3.0 * MACRO_W[b]
    elif _completes(cells, c, -p):
        g += 1.5 * MACRO_W[b]
        if _macro_wins(small, b, -p):
            g += 20.0
    return g


class Banfield(StaticBot):
    NAME = "Banfield"
    COLOR1 = "#1f9d55"     # verde
    COLOR2 = "#ffffff"     # blanco

    OPP_W = 0.6            # cuánto pesa la mejor respuesta del rival
    TEMP, MARGIN = 0.9, 3.0

    def choose(self, game) -> tuple[int, int]:
        me = 1 if game.current_player == 1 else -1
        board = [[int(v) for v in row] for row in game.board.tolist()]
        small = game.small_boards.tolist()
        sq = [int(x) for x in game.squares_played]
        active = None if game.active_board is None else int(game.active_board)

        scored = []
        for b, c in _legal(board, small, active):
            g = _gain(board, small, b, c, me)
            if g >= WIN:
                return b, c
            ns, nact, win = _after(board, small, sq, b, c, me)
            if win == 0:
                scored.append((0.0, (b, c)))
                continue
            score = g + 8.0 * (_macro_threats(ns, me) - _macro_threats(small, me))
            nb = board[:]
            row = board[b][:]
            row[c] = me
            nb[b] = row
            replies = _legal(nb, ns, nact)
            if replies:
                worst = max(_gain(nb, ns, rb, rc, -me) for rb, rc in replies)
                if worst >= WIN:
                    scored.append((LOSS_GUARD, (b, c)))          # no regalar la partida
                    continue
                score -= self.OPP_W * worst
            scored.append((score, (b, c)))
        return self._pick(scored)

    def _pick(self, scored):
        best = max(s for s, _ in scored)
        if best <= LOSS_GUARD / 2:
            return max(scored, key=lambda t: t[0])[1]
        pool = sorted(((s, m) for s, m in scored if s >= best - self.MARGIN and s > LOSS_GUARD / 2), key=lambda t: t[1])
        weights = [math.exp((s - best) / self.TEMP) for s, _ in pool]
        r = self.rng.random() * sum(weights)
        acc = 0.0
        for w, (_, m) in zip(weights, pool):
            acc += w
            if r <= acc:
                return m
        return pool[-1][1]