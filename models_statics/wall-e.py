"""WALL-E: bot estático metódico (sin red, sin mutación) con azar controlado.

Capas, en orden de prioridad (ver el plan en la conversación):
  1. Reglas absolutas: ganar la partida si se puede; nunca regalarla (se detecta mirando la mejor
     respuesta del rival tras cada jugada, incluido el tablero al que lo obligo a jugar).
  2. Evaluación posicional: ganar/bloquear tableros chicos según su valor macro (centro > esquinas >
     bordes), amenazas macro, horquillas, casillas centrales y -sobre todo- DESTINO DEL RIVAL.
     Regla de este juego: si mando al rival a un tablero cerrado, debe jugar en el último tablero
     abierto del historial (casi siempre donde acabo de jugar), no tiene libertad.
  3. Búsqueda: en el final (pocos tableros abiertos) minimax con poda alfa-beta y tope de nodos.
  4. Azar controlado: entre las jugadas dentro de un margen de la mejor se sortea con softmax; las
     jugadas tácticamente forzadas (ganar, evitar perder) no se sortean.
"""
from __future__ import annotations

import math

from models_statics.base import StaticBot

WIN_COMBOS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
LINES_THROUGH = [[tuple(x for x in line if x != i) for line in WIN_COMBOS if i in line] for i in range(9)]
MACRO_W = [3, 2, 3, 2, 4, 2, 3, 2, 3]          # nº de líneas que cruza cada tablero (valor macro)
CELL_W = [0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5]

WIN = 1000.0          # ganar la partida
LOSS_GUARD = -400.0   # jugada que regala la partida
NODE_BUDGET = 3500    # tope de nodos de la búsqueda de final (por jugada)


# --------------------------------------------------------------------------- utilidades (listas, rápido)
def _completes(cells, idx, p) -> bool:
    return any(cells[a] == p and cells[b] == p for a, b in LINES_THROUGH[idx])


def _macro_wins(small, b, p) -> bool:
    return any(small[x] == p and small[y] == p for x, y in LINES_THROUGH[b])


def _macro_threats(small, p) -> int:
    """Líneas macro con 2 tableros de p y el tercero todavía en juego."""
    n = 0
    for a, b, c in WIN_COMBOS:
        line = (small[a], small[b], small[c])
        if line.count(p) == 2 and line.count(0.5) == 1:
            n += 1
    return n


def _macro_potential(small, p) -> int:
    """Líneas macro todavía alcanzables por p (sin fichas rivales) con al menos un tablero ya suyo."""
    n = 0
    for a, b, c in WIN_COMBOS:
        line = (small[a], small[b], small[c])
        if line.count(-p) == 0 and line.count(0.0) == 0 and line.count(p) >= 1:
            n += line.count(p)
    return n


def _small_threats(cells, p) -> int:
    """Líneas del tablero chico con 2 de p y la tercera vacía."""
    n = 0
    for a, b, c in WIN_COMBOS:
        line = (cells[a], cells[b], cells[c])
        if line.count(p) == 2 and line.count(0) == 1:
            n += 1
    return n


def _legal(board, small, active):
    boards = [active] if active is not None else [i for i in range(9) if small[i] == 0.5]
    return [(b, c) for b in boards for c in range(9) if board[b][c] == 0]


def _gain(board, small, b, c, p) -> float:
    """Valor táctico inmediato de que ``p`` juegue en (b, c). Sin mirar la respuesta."""
    cells = board[b]
    g = CELL_W[c] + (0.5 if b == 4 else 0.0)
    if _completes(cells, c, p):
        if _macro_wins(small, b, p):
            return WIN
        g += 3.0 * MACRO_W[b]
        for x, y in LINES_THROUGH[b]:                       # crea amenaza macro (2 en línea + tercero abierto)
            if (small[x] == p and small[y] == 0.5) or (small[x] == 0.5 and small[y] == p):
                g += 5.0
    elif _completes(cells, c, -p):
        g += 1.5 * MACRO_W[b]                                # bloquea un tablero chico del rival
        if _macro_wins(small, b, -p):
            g += 20.0                                        # ...que además le daba la partida
    elif cells.count(0) == 1:
        g -= 3.0                                             # llena el tablero sin ganarlo: se muere (empata y apaga líneas macro)
    return g


def _evaluate(board, small, me) -> float:
    """Evaluación estática desde el punto de vista de ``me`` (hojas de la búsqueda)."""
    s = 0.0
    for i in range(9):
        if small[i] == me:
            s += 3.0 * MACRO_W[i]
        elif small[i] == -me:
            s -= 3.0 * MACRO_W[i]
        elif small[i] == 0.5:
            cells = board[i]
            s += 0.6 * (cells[4] * me)
            s += (MACRO_W[i] / 3.0) * (_small_threats(cells, me) - _small_threats(cells, -me))
    for a, b, c in WIN_COMBOS:
        line = (small[a], small[b], small[c])
        m, o, u = line.count(me), line.count(-me), line.count(0.5)
        if o == 0 and m == 2 and u == 1:
            s += 8.0
        elif m == 0 and o == 2 and u == 1:
            s -= 8.0
        elif o == 0 and m == 1 and u == 2:
            s += 1.5
        elif m == 0 and o == 1 and u == 2:
            s -= 1.5
    return s


class WallE(StaticBot):
    NAME = "WALL-E"
    COLOR1 = "#e8b923"     # amarillo óxido
    COLOR2 = "#3b6fb6"     # azul acero

    # ------------------------------------------------------------------ API del torneo
    def choose(self, game) -> tuple[int, int]:
        me = game.current_player
        board, small = game.board.tolist(), game.small_boards.tolist()
        legal = _legal(board, small, game.active_board)
        open_boards = sum(1 for x in small if x == 0.5)

        scored: list[tuple[float, tuple[int, int]]] = []
        for b, c in legal:
            g = _gain(board, small, b, c, me)
            if g >= WIN:                                     # capa 1: ganar ya
                return b, c
            scored.append((self._score_move(game, board, small, b, c, me, g), (b, c)))

        endgame = open_boards <= 3 or len(legal) <= 9 and game.turn >= 45
        if endgame:
            scored = self._refine_endgame(game, scored, me)

        best = max(s for s, _ in scored)
        if best <= LOSS_GUARD / 2:                           # todo pierde: da igual, no hay nada que sortear
            return max(scored, key=lambda t: t[0])[1]
        return self._sample(scored, best, game.turn, endgame)

    # ------------------------------------------------------------------ puntuación de una jugada
    def _score_move(self, game, board, small, b, c, me, g) -> float:
        sim = game.clone()
        sim.make_move(b, c)
        if sim.winner == 0:
            return 0.0
        nb, ns = sim.board.tolist(), sim.small_boards.tolist()
        score = g
        score += 8.0 * (_macro_threats(ns, me) - _macro_threats(small, me))
        score += 7.0 * (_macro_threats(small, -me) - _macro_threats(ns, -me))   # cerrar una amenaza macro del rival
        score += 1.2 * (_macro_potential(ns, me) - _macro_potential(small, me))
        score += 1.2 * (_macro_potential(small, -me) - _macro_potential(ns, -me))   # apagar líneas del rival
        t = _small_threats(nb[b], me)
        if t >= 2:
            score += 5.0                                    # horquilla en el tablero chico
        elif t == 1 and ns[b] == 0.5:
            score += 1.5
        # Destino del rival: su mejor respuesta (incluye el tablero forzado por la regla del historial)
        replies = _legal(nb, ns, sim.active_board)
        if replies:
            gains = [_gain(nb, ns, rb, rc, -me) for rb, rc in replies]
            worst = max(gains)
            if worst >= WIN:
                return LOSS_GUARD                           # capa 1: no regalar la partida
            score -= 0.85 * worst + 0.15 * (sum(gains) / len(gains))
            if len(replies) <= 2 and worst < 1.0:
                score += 2.0                                # lo dejo sin nada útil que hacer
        return score

    # ------------------------------------------------------------------ final: alfa-beta
    def _refine_endgame(self, game, scored, me):
        depth = 4 if len(scored) <= 9 else 3
        self._nodes = 0
        out = []
        for base, (b, c) in scored:
            if base <= LOSS_GUARD:
                out.append((base, (b, c)))
                continue
            sim = game.clone()
            sim.make_move(b, c)
            val = -self._negamax(sim, depth - 1, -math.inf, math.inf)
            out.append((0.35 * base + val, (b, c)))
        return out

    def _negamax(self, g, depth, alpha, beta) -> float:
        me = g.current_player
        if g.winner is not None:
            return 0.0 if g.winner == 0 else (-WIN - depth if g.winner == -me else WIN + depth)
        board, small = g.board.tolist(), g.small_boards.tolist()
        self._nodes += 1
        if depth <= 0 or self._nodes > NODE_BUDGET:
            return _evaluate(board, small, me)
        moves = _legal(board, small, g.active_board)
        moves.sort(key=lambda m: -_gain(board, small, m[0], m[1], me))
        best = -math.inf
        for b, c in moves:
            s = g.clone()
            s.make_move(b, c)
            v = -self._negamax(s, depth - 1, -beta, -alpha)
            if v > best:
                best = v
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    # ------------------------------------------------------------------ azar controlado
    def _sample(self, scored, best, turn, endgame):
        if best >= 40.0 and not endgame:
            temp, margin = 0.15, 1.0                        # táctica clara: casi determinista
        elif endgame:
            temp, margin = 0.4, 1.5
        elif turn < 8:
            temp, margin = 1.0, 3.0                         # apertura: variedad
        else:
            temp, margin = 0.4, 1.8
        pool = [(s, m) for s, m in scored if s >= best - margin and s > LOSS_GUARD / 2]
        pool.sort(key=lambda t: t[1])                       # orden estable => reproducible
        weights = [math.exp((s - best) / temp) for s, _ in pool]
        r = self.rng.random() * sum(weights)
        acc = 0.0
        for w, (_, m) in zip(weights, pool):
            acc += w
            if r <= acc:
                return m
        return pool[-1][1]
