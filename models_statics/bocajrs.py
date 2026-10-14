"""BocaJrs: bot estático rápido (sin red, sin MCTS, sin árbol recursivo), evolución de WALL-E.

Cómo decide (≈ 0,5 ms por jugada, máx. ≈ 9 ms):
  1. Reglas absolutas: si puede ganar la partida, la gana; si una jugada le regala la partida al rival, la descarta.
  2. Puntaje de cada jugada = valor táctico propio (tableros ganados/bloqueados según su peso macro) + amenazas y
     potencial en la grilla grande + horquillas, MENOS lo que puede hacer el rival en el tablero al que lo mando
     (incluida la regla de este juego: destino cerrado => debe volver al último tablero abierto del historial),
     MÁS lo que podría hacer yo después de sus dos respuestas más peligrosas ("lo que puede llegar a pasar").
  3. En el final (pocos tableros abiertos) hace un lookahead PLANO y acotado de 3 jugadas (mi jugada, sus mejores
     réplicas, mis mejores seguimientos) solo con las candidatas más fuertes. Se apaga con ENDGAME_LOOKAHEAD = False
     (sin él queda al nivel de WALL-E sin búsqueda). No hay profundización, ni deshacer jugadas, ni MCTS.
  4. Azar controlado: softmax entre jugadas casi igual de buenas (más variedad en la apertura, poca en el medio
     juego); las jugadas forzadas (ganar ya, no perder) no se sortean.

Trabaja con un estado liviano en listas (sin ``game.clone()``): cada hipótesis cuesta microsegundos.
Los pesos de arriba (GAIN_BOARD, OPP_MEAN, FOLLOW...) se ajustaron jugando contra WALL-E y el bot táctico.
"""
from __future__ import annotations

import math

from models_statics.base import StaticBot

WIN_COMBOS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
LINES_THROUGH = [[tuple(x for x in line if x != i) for line in WIN_COMBOS if i in line] for i in range(9)]
MACRO_W = [3, 2, 3, 2, 4, 2, 3, 2, 3]
CELL_W = [0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5]

WIN = 1000.0

# Pesos de la heurística (ajustados jugando contra otros bots; ver notas al final del archivo).
GAIN_BOARD = 1.8      # ganar un tablero chico: GAIN_BOARD x peso macro del tablero
BLOCK = 1.5           # bloquear un tablero chico del rival
MACRO_MAKE = 5.0      # ganar un tablero que deja una amenaza macro
THR_MY = 8.0          # crear amenaza macro propia (2 tableros en línea + tercero libre)
THR_OPP = 7.0         # cerrar una amenaza macro del rival
POT = 1.2             # potencial de líneas macro (propias a favor, del rival en contra)
FORK = 5.0            # horquilla en el tablero chico
OPP_WORST = 0.85      # peso de la mejor respuesta del rival
OPP_MEAN = 0.09       # peso de la respuesta media del rival
FOLLOW = 0.455         # peso de lo que puedo hacer yo después de su mejor respuesta
EV_BOARD = 3.0        # (hojas) tablero ganado x peso macro
EV_LINE2 = 8.0        # (hojas) línea macro con 2 propios y el tercero libre
EV_LINE1 = 1.5        # (hojas) línea macro con 1 propio y 2 libres
EV_CENTER = 0.6       # (hojas) centro de tableros abiertos
EV_THREAT = 1.0       # (hojas) amenazas dentro de tableros abiertos
EV_TEMPO = 0.8        # (hojas) lo que puede hacer el que mueve a continuación


# --------------------------------------------------------------------------- estado liviano
# (board, small, sq, active): board = 9 listas de 9 (0 vacía, 1/-1 fichas), small = 9 (0.5 en juego, 0 empate, ±1),
# sq = historial de tableros jugados, active = tablero obligatorio o None.
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


def _legal(board, small, active):
    boards = (active,) if active is not None else [i for i in range(9) if small[i] == 0.5]
    return [(b, c) for b in boards for c in range(9) if board[b][c] == 0]


def _apply(board, small, sq, b, c, p):
    """Juega p en (b, c). Devuelve (board, small, sq, active, winner) con winner None / ±1 / 0 (empate)."""
    row = board[b][:]
    won = _completes(row, c, p)
    row[c] = p
    nb = board[:]
    nb[b] = row
    ns = small
    if won:
        if _macro_wins(small, b, p):
            return nb, small, sq, None, p
        ns = small[:]
        ns[b] = p
    elif 0 not in row:
        ns = small[:]
        ns[b] = 0
    nsq = sq if (sq and sq[-1] == b) else sq + [b]
    if ns[c] == 0.5:
        act = c
    else:
        act = None
        for x in reversed(nsq):
            if ns[x] == 0.5:
                act = x
                break
    if 0.5 not in ns:
        return nb, ns, nsq, None, 0
    return nb, ns, nsq, act, None


def _gain(board, small, b, c, p) -> float:
    """Valor táctico inmediato de que p juegue en (b, c) (barato, sin aplicar la jugada)."""
    cells = board[b]
    g = CELL_W[c] + (0.5 if b == 4 else 0.0)
    if _completes(cells, c, p):
        if _macro_wins(small, b, p):
            return WIN
        g += GAIN_BOARD * MACRO_W[b]
        for x, y in LINES_THROUGH[b]:
            if (small[x] == p and small[y] == 0.5) or (small[x] == 0.5 and small[y] == p):
                g += MACRO_MAKE
    elif _completes(cells, c, -p):
        g += BLOCK * MACRO_W[b]
        if _macro_wins(small, b, -p):
            g += 20.0
    elif cells.count(0) == 1:
        g -= 3.0
    return g


def _best_gain(board, small, active, p) -> float:
    best = 0.0
    for b, c in _legal(board, small, active):
        g = _gain(board, small, b, c, p)
        if g > best:
            best = g
    return best


def _small_threats(cells, p) -> int:
    n = 0
    for a, b, c in WIN_COMBOS:
        line = (cells[a], cells[b], cells[c])
        if line.count(p) == 2 and line.count(0) == 1:
            n += 1
    return n


def _evaluate(board, small, active, me, mover) -> float:
    """Evaluación estática desde el punto de vista de ``me``; ``mover`` es quien juega a continuación."""
    s = 0.0
    for i in range(9):
        v = small[i]
        if v == me:
            s += EV_BOARD * MACRO_W[i]
        elif v == -me:
            s -= EV_BOARD * MACRO_W[i]
        elif v == 0.5:
            cells = board[i]
            s += EV_CENTER * cells[4] * me
            s += EV_THREAT * (MACRO_W[i] / 3.0) * (_small_threats(cells, me) - _small_threats(cells, -me))
    for a, b, c in WIN_COMBOS:
        line = (small[a], small[b], small[c])
        m, o, u = line.count(me), line.count(-me), line.count(0.5)
        if o == 0 and m == 2 and u == 1:
            s += EV_LINE2
        elif m == 0 and o == 2 and u == 1:
            s -= EV_LINE2
        elif o == 0 and m == 1 and u == 2:
            s += EV_LINE1
        elif m == 0 and o == 1 and u == 2:
            s -= EV_LINE1
    # lo que puede hacer el que mueve ahora (ventana de oportunidad del siguiente turno)
    tempo = _best_gain(board, small, active, mover)
    s += EV_TEMPO * tempo if mover == me else -EV_TEMPO * tempo
    return s


def _macro_threats(small, p) -> int:
    n = 0
    for a, b, c in WIN_COMBOS:
        line = (small[a], small[b], small[c])
        if line.count(p) == 2 and line.count(0.5) == 1:
            n += 1
    return n


def _macro_potential(small, p) -> int:
    n = 0
    for a, b, c in WIN_COMBOS:
        line = (small[a], small[b], small[c])
        if line.count(-p) == 0 and line.count(0.0) == 0 and line.count(p) >= 1:
            n += line.count(p)
    return n


class BocaJrs(StaticBot):
    NAME = "BocaJrs"
    COLOR1 = "#0a3d91"     # azul
    COLOR2 = "#ffd400"     # amarillo

    LOSS_GUARD = -400.0
    ENDGAME_LOOKAHEAD = True
    ENDGAME_BOARDS = 3     # el final empieza cuando quedan tantos tableros abiertos o menos
    TEMP_MID, MARGIN_MID = 0.2, 1.0      # medio juego: pool de jugadas casi iguales
    TEMP_OPEN, MARGIN_OPEN = 1.0, 3.0    # apertura (primeras 8 jugadas): más variedad
    END_ROOT, END_REPLIES, END_FOLLOW = 6, 6, 4

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
                return b, c                                      # capa 1: ganar ya
            scored.append((self._score_move(board, small, sq, b, c, me, g), (b, c)))
        open_boards = sum(1 for x in small if x == 0.5)
        if self.ENDGAME_LOOKAHEAD and (open_boards <= self.ENDGAME_BOARDS or (len(scored) <= 9 and game.turn >= 45)):
            scored = self._endgame(board, small, sq, me, scored)
        best = max(s for s, _ in scored)
        return self._sample(scored, best, game.turn)

    # -- final: lookahead PLANO y acotado (mi jugada -> sus mejores réplicas -> mis mejores seguimientos) ----------
    def _endgame(self, board, small, sq, me, scored):
        ranked = sorted(scored, key=lambda t: -t[0])
        out = []
        for k, (base, (b, c)) in enumerate(ranked):
            if k >= self.END_ROOT or base <= self.LOSS_GUARD:
                out.append((base - 30.0 if base > self.LOSS_GUARD else base, (b, c)))
                continue
            nb, ns, nsq, nact, win = _apply(board, small, sq, b, c, me)
            if win == 0:
                out.append((0.0, (b, c)))
                continue
            worst = math.inf
            replies = _legal(nb, ns, nact)
            replies.sort(key=lambda m: -_gain(nb, ns, m[0], m[1], -me))
            for rb, rc in replies[:self.END_REPLIES]:
                b2, s2, q2, a2, w2 = _apply(nb, ns, nsq, rb, rc, -me)
                if w2 == -me:
                    worst = -WIN
                    break
                if w2 == 0:
                    worst = min(worst, 0.0)
                    continue
                best_f = -math.inf
                fl = _legal(b2, s2, a2)
                fl.sort(key=lambda m: -_gain(b2, s2, m[0], m[1], me))
                for fb, fc in fl[:self.END_FOLLOW]:
                    b3, s3, q3, a3, w3 = _apply(b2, s2, q2, fb, fc, me)
                    if w3 == me:
                        v = WIN
                    elif w3 == 0:
                        v = 0.0
                    else:
                        v = _evaluate(b3, s3, a3, me, -me)
                    if v > best_f:
                        best_f = v
                worst = min(worst, best_f if best_f != -math.inf else 0.0)
            val = 0.0 if worst == math.inf else worst
            out.append((0.35 * base + val, (b, c)))
        return out

    def _score_move(self, board, small, sq, b, c, me, g) -> float:
        nb, ns, nsq, nact, win = _apply(board, small, sq, b, c, me)
        if win == 0:
            return 0.0
        score = g
        score += THR_MY * (_macro_threats(ns, me) - _macro_threats(small, me))
        score += THR_OPP * (_macro_threats(small, -me) - _macro_threats(ns, -me))
        score += POT * (_macro_potential(ns, me) - _macro_potential(small, me))
        score += POT * (_macro_potential(small, -me) - _macro_potential(ns, -me))
        t = _small_threats(nb[b], me)
        if t >= 2:
            score += FORK
        elif t == 1 and ns[b] == 0.5:
            score += 1.5
        replies = _legal(nb, ns, nact)
        if replies:
            gl = [(_gain(nb, ns, rb, rc, -me), (rb, rc)) for rb, rc in replies]
            gl.sort(key=lambda t: -t[0])
            worst = gl[0][0]
            if worst >= WIN:
                return self.LOSS_GUARD
            score -= OPP_WORST * worst + OPP_MEAN * (sum(x for x, _ in gl) / len(gl))
            if len(replies) <= 2 and worst < 1.0:
                score += 2.0
            # 3.ª jugada: tras sus 2 réplicas más peligrosas, ¿qué puedo hacer yo? (la peor de las dos)
            follow = math.inf
            for gv, (rb, rc) in gl[:2]:
                b2, s2, q2, a2, w2 = _apply(nb, ns, nsq, rb, rc, -me)
                if w2 is None:
                    follow = min(follow, _best_gain(b2, s2, a2, me))
            if follow != math.inf:
                score += FOLLOW * follow
        return score

    def _sample(self, scored, best, turn):
        if best <= self.LOSS_GUARD / 2:
            return max(scored, key=lambda t: t[0])[1]
        if best >= 40.0:
            temp, margin = 0.15, 1.0
        elif turn < 8:
            temp, margin = self.TEMP_OPEN, self.MARGIN_OPEN
        else:
            temp, margin = self.TEMP_MID, self.MARGIN_MID
        pool = [(s, m) for s, m in scored if s >= best - margin and s > self.LOSS_GUARD / 2]
        pool.sort(key=lambda t: t[1])
        weights = [math.exp((s - best) / temp) for s, _ in pool]
        r = self.rng.random() * sum(weights)
        acc = 0.0
        for w, (_, m) in zip(weights, pool):
            acc += w
            if r <= acc:
                return m
        return pool[-1][1]