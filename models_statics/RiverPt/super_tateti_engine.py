"""
[Motor de un amigo, SIN la parte de partidas/exportación/consola: ver el original para eso.]
[Cambio respecto del original: MCTSBot recibe ``rng`` (random.Random) para que su azar sea reproducible por partida.]

Super Tateti - motor con módulo de análisis de tableros chicos + minimax (alfa-beta).

Convenciones:
  - Jugadores: 1 (X) y 2 (O).
  - Un movimiento es (b, c): b = tablero chico (0-8), c = casilla dentro de ese tablero (0-8).
  - Las casillas se numeran así en cada tablero 3x3:
        0 1 2
        3 4 5
        6 7 8
  - Jugar en la casilla c manda al rival al tablero chico c.
  - Tablero chico empatado/lleno: no cuenta para nadie.
"""
import time
import random

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]

DRAW = 3
WIN_SCORE = 100_000


# ---------------------------------------------------------------------------
# 1) MÓDULO DE ANÁLISIS DE TABLEROS CHICOS
# ---------------------------------------------------------------------------
class SmallBoardAnalyzer:
    """Evalúa un tateti chico. Devuelve puntaje desde el punto de vista de X
    (positivo = bueno para X, negativo = bueno para O). Usa caché: solo hay
    3^9 = 19683 posiciones posibles."""

    CELL_W = [3, 2, 3, 2, 4, 2, 3, 2, 3]  # esquinas y centro valen más

    def __init__(self):
        self._cache = {}

    @staticmethod
    def winner(cells):
        for a, b, c in LINES:
            if cells[a] and cells[a] == cells[b] == cells[c]:
                return cells[a]
        return 0

    def score(self, cells):
        key = tuple(cells)
        s = self._cache.get(key)
        if s is None:
            s = self._compute(key)
            self._cache[key] = s
        return s

    def _compute(self, cells):
        s = 0
        # valor posicional
        for i, v in enumerate(cells):
            if v == 1:
                s += self.CELL_W[i]
            elif v == 2:
                s -= self.CELL_W[i]
        # líneas abiertas: 2 en línea con hueco = amenaza, 1 solo = potencial
        for a, b, c in LINES:
            line = (cells[a], cells[b], cells[c])
            x, o = line.count(1), line.count(2)
            if x and o:
                continue
            if x == 2:
                s += 12
            elif x == 1:
                s += 2
            if o == 2:
                s -= 12
            elif o == 1:
                s -= 2
        return s

    @staticmethod
    def winning_cells(cells, player):
        """Casillas donde 'player' completa tres en línea de inmediato."""
        out = set()
        for a, b, c in LINES:
            line = (cells[a], cells[b], cells[c])
            if line.count(player) == 2 and line.count(0) == 1:
                out.add((a, b, c)[line.index(0)])
        return out


# ---------------------------------------------------------------------------
# 2) ESTADO DEL JUEGO (con make/undo para que la búsqueda sea rápida)
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.cells = [[0] * 9 for _ in range(9)]
        self.status = [0] * 9      # 0 en juego, 1/2 ganado, 3 empatado
        self.next = -1             # -1 = libre
        self.player = 1
        self.winner = 0            # 0 en juego, 1/2 ganó, 3 empate
        self._hist = []

    def legal_moves(self):
        if self.winner:
            return []
        boards = [self.next] if self.next != -1 else \
            [b for b in range(9) if self.status[b] == 0]
        return [(b, c) for b in boards for c in range(9) if self.cells[b][c] == 0]

    def play(self, b, c):
        p = self.player
        self._hist.append((b, c, self.status[b], self.next, self.winner))
        self.cells[b][c] = p
        if self.status[b] == 0:
            if SmallBoardAnalyzer.winner(self.cells[b]):
                self.status[b] = p
            elif all(self.cells[b]):
                self.status[b] = DRAW
        if self.status[b] == p:
            for x, y, z in LINES:
                if self.status[x] == self.status[y] == self.status[z] == p:
                    self.winner = p
                    break
        if not self.winner and all(self.status):
            self.winner = DRAW
        self.next = c if self.status[c] == 0 else -1
        self.player = 3 - p

    def to_dict(self):
        return {"cells": [row[:] for row in self.cells], "status": self.status[:],
                "next": self.next, "player": self.player, "winner": self.winner}

    def undo(self):
        b, c, st, nxt, win = self._hist.pop()
        self.cells[b][c] = 0
        self.status[b] = st
        self.next = nxt
        self.winner = win
        self.player = 3 - self.player


# ---------------------------------------------------------------------------
# 3) EVALUACIÓN GLOBAL (usa el analizador de tableros chicos)
# ---------------------------------------------------------------------------
BOARD_W = [3, 2, 3, 2, 4, 2, 3, 2, 3]  # importancia de cada tablero en el grande


class Evaluator:
    def __init__(self):
        self.small = SmallBoardAnalyzer()

    def evaluate(self, st):
        """Puntaje desde el punto de vista de X."""
        s = 0
        for b in range(9):
            t = st.status[b]
            if t == 1:
                s += 60 * BOARD_W[b]
            elif t == 2:
                s -= 60 * BOARD_W[b]
            elif t == 0:
                # tablero abierto: su estado interno vale, ponderado por su importancia
                s += self.small.score(st.cells[b]) * BOARD_W[b] / 3
        # líneas del tablero grande
        for a, b, c in LINES:
            line = (st.status[a], st.status[b], st.status[c])
            x, o = line.count(1), line.count(2)
            if x and o:
                continue
            if line.count(DRAW):
                continue
            if x == 2:
                s += 150
            elif x == 1:
                s += 20
            if o == 2:
                s -= 150
            elif o == 1:
                s -= 20
        # mover con tablero libre es ventaja; con amenaza inmediata en el tablero
        # al que te mandan, el que mueve gana un tablero
        mover = 1 if st.player == 1 else -1
        if st.next == -1:
            s += 25 * mover
        else:
            cells = st.cells[st.next]
            if self.small.winning_cells(cells, st.player):
                s += 30 * mover
            if self.small.winning_cells(cells, 3 - st.player):
                s -= 15 * mover
        return s


def move_heuristic(st, b, c, small):
    """Puntaje rápido de qué tan prometedora es la jugada (b, c). Sirve para
    ordenar en alfa-beta y como probabilidad previa (prior) en MCTS."""
    p = st.player
    s = SmallBoardAnalyzer.CELL_W[c]
    if c in small.winning_cells(st.cells[b], p):
        s += 100                       # gano un tablero chico
        # si además ese tablero completa una línea grande, es casi la partida
        for x, y, z in LINES:
            if b in (x, y, z):
                o = [i for i in (x, y, z) if i != b]
                if st.status[o[0]] == p and st.status[o[1]] == p:
                    s += 400
    elif c in small.winning_cells(st.cells[b], 3 - p):
        s += 40                        # bloqueo
    if c != b:
        if st.status[c] != 0:
            s -= 15                    # le regalo jugada libre
        elif small.winning_cells(st.cells[c], 3 - p):
            s -= 60                    # lo mando a donde gana un tablero
    return s


# ---------------------------------------------------------------------------
# 4) BÚSQUEDA: negamax + alfa-beta + profundización iterativa
# ---------------------------------------------------------------------------
class TimeUp(Exception):
    pass


class Bot:
    def __init__(self, time_limit=2.0, max_depth=12, full_scores=False):
        # full_scores=True: busca cada jugada raíz con ventana completa, así el
        # valor de TODAS las candidatas es exacto (más lento, útil para entrenar).
        self.full_scores = full_scores
        self.last_info = {}
        self.ev = Evaluator()
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.nodes = 0

    # ---- ordenamiento de jugadas: lo prometedor primero (poda mucho mejor)
    def _order(self, st, moves, pv=None):
        small = self.ev.small

        def key(m):
            if m == pv:
                return 10_000
            return move_heuristic(st, m[0], m[1], small)

        return sorted(moves, key=key, reverse=True)

    def _negamax(self, st, depth, alpha, beta, ply):
        self.nodes += 1
        if self.nodes & 1023 == 0 and time.time() > self.deadline:
            raise TimeUp
        if st.winner:
            if st.winner == DRAW:
                return 0
            return -(WIN_SCORE - ply)          # el que movió antes ganó -> malo para mí
        if depth == 0:
            e = self.ev.evaluate(st)
            return e if st.player == 1 else -e
        best = -float("inf")
        for m in self._order(st, st.legal_moves()):
            st.play(*m)
            try:
                v = -self._negamax(st, depth - 1, -beta, -alpha, ply + 1)
            finally:
                st.undo()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def choose(self, st, verbose=False):
        moves = st.legal_moves()
        t0 = time.time()
        if len(moves) == 1:
            self.last_info = {"depth": 0, "value": None, "nodes": 0, "time": 0.0,
                              "forced": True,
                              "candidates": [{"move": list(moves[0]), "value": None}]}
            return moves[0]
        self.deadline = t0 + self.time_limit
        self.nodes = 0
        best_move = random.choice(self._order(st, moves)[:2])
        pv = None
        info = {"depth": 0, "value": None, "forced": False, "candidates": []}
        for depth in range(1, self.max_depth + 1):
            try:
                alpha, beta = -float("inf"), float("inf")
                cur_best, cur_val = None, -float("inf")
                cands = []
                for m in self._order(st, moves, pv):
                    st.play(*m)
                    try:
                        v = -self._negamax(st, depth - 1, -beta, -alpha, 1)
                    finally:
                        st.undo()
                    cands.append({"move": list(m), "value": v})
                    if v > cur_val:
                        cur_val, cur_best = v, m
                    if not self.full_scores:
                        alpha = max(alpha, cur_val)
                best_move, pv = cur_best, cur_best
                info.update(depth=depth, value=cur_val, candidates=cands)
                if verbose:
                    print(f"  prof {depth}: {best_move} val={cur_val:.0f} nodos={self.nodes}")
                if abs(cur_val) > WIN_SCORE - 100:
                    break                       # ya encontró victoria/derrota forzada
            except TimeUp:
                break
        info["engine"] = "alphabeta"
        info["nodes"] = self.nodes
        info["time"] = round(time.time() - t0, 3)
        self.last_info = info
        return best_move


# ---------------------------------------------------------------------------
# 4b) MCTS con PUCT (estilo AlphaZero, pero con la heurística en lugar de red)
# ---------------------------------------------------------------------------
import math


class _Node:
    __slots__ = ("N", "W", "P", "children")

    def __init__(self, p):
        self.N = 0          # visitas
        self.W = 0.0        # suma de valores, desde la perspectiva de quien movió hacia este nodo
        self.P = p          # probabilidad previa de elegir este nodo desde su padre
        self.children = None


class MCTSBot:
    """Monte Carlo Tree Search guiado por la heurística.

    - Prior de cada jugada: softmax(move_heuristic / prior_temp).
    - Valor de hoja: tanh(evaluación / value_scale) en [-1, 1] (sin rollouts al azar).
    - Selección PUCT: Q + c_puct * P * sqrt(N_padre) / (1 + N_hijo).
    - La salida es una distribución de probabilidad (visitas normalizadas).

    Variedad:
    - temperature > 0: en las primeras temp_plies jugadas se *muestrea* la jugada
      según visitas^(1/temperature) en lugar de elegir la más visitada.
    - dirichlet_alpha > 0: ruido Dirichlet en la raíz (exploración para entrenar).
    """

    def __init__(self, time_limit=1.0, sims=None, c_puct=1.4, prior_temp=40.0,
                 value_scale=350.0, temperature=0.0, temp_plies=10,
                 dirichlet_alpha=0.0, noise_frac=0.25, rng=None):
        self.time_limit = time_limit
        self.sims = sims                  # si se da, usa esa cantidad fija en vez de tiempo
        self.c_puct = c_puct
        self.prior_temp = prior_temp
        self.value_scale = value_scale
        self.temperature = temperature
        self.temp_plies = temp_plies
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self.rng = rng if rng is not None else random   # azar propio (reproducible) o el global
        self.ev = Evaluator()
        self.last_info = {}

    # -- expansión: crea hijos con sus probabilidades previas
    def _expand(self, st, node):
        moves = st.legal_moves()
        small = self.ev.small
        scores = [move_heuristic(st, b, c, small) / self.prior_temp for b, c in moves]
        mx = max(scores)
        exps = [math.exp(x - mx) for x in scores]
        tot = sum(exps)
        node.children = {m: _Node(e / tot) for m, e in zip(moves, exps)}

    def _leaf_value(self, st):
        """Valor en [-1,1] desde la perspectiva de quien mueve en st."""
        if st.winner:
            return 0.0 if st.winner == DRAW else -1.0   # ganó el que movió antes
        e = self.ev.evaluate(st)
        if st.player == 2:
            e = -e
        return math.tanh(e / self.value_scale)

    def _select(self, node):
        sqrt_n = math.sqrt(node.N + 1)
        # first-play urgency: lo no visitado vale un poco menos que el promedio del padre
        fpu = (-node.W / node.N - 0.15) if node.N else 0.0
        best, best_s, best_m = None, -1e9, None
        c = self.c_puct
        for m, ch in node.children.items():
            q = ch.W / ch.N if ch.N else fpu
            sc = q + c * ch.P * sqrt_n / (1 + ch.N)
            if sc > best_s:
                best, best_s, best_m = ch, sc, m
        return best_m, best

    def _simulate(self, st, root):
        path = [root]
        node = root
        while node.children is not None and not st.winner:
            m, node = self._select(node)
            st.play(*m)
            path.append(node)
        if not st.winner and node.children is None:
            self._expand(st, node)
        v = self._leaf_value(st)           # perspectiva de quien mueve en la hoja
        for _ in range(len(path) - 1):
            st.undo()
        val = -v                           # perspectiva de quien movió hacia la hoja
        for n in reversed(path):
            n.N += 1
            n.W += val
            val = -val

    def choose(self, st):
        moves = st.legal_moves()
        t0 = time.time()
        root = _Node(1.0)
        self._expand(st, root)
        if self.dirichlet_alpha > 0:
            noise = [self.rng.gammavariate(self.dirichlet_alpha, 1) for _ in root.children]
            tot = sum(noise)
            for ch, n in zip(root.children.values(), noise):
                ch.P = (1 - self.noise_frac) * ch.P + self.noise_frac * n / tot
        n_sims = 0
        if len(moves) > 1:
            deadline = t0 + self.time_limit
            while True:
                if self.sims is not None:
                    if n_sims >= self.sims:
                        break
                elif (n_sims & 15) == 0 and time.time() > deadline:
                    break
                self._simulate(st, root)
                n_sims += 1
        total = sum(ch.N for ch in root.children.values()) or 1
        cand = []
        for m, ch in root.children.items():
            cand.append({"move": list(m), "visits": ch.N,
                         "p": round(ch.N / total, 5),
                         "q": round(ch.W / ch.N, 4) if ch.N else None})
        cand.sort(key=lambda d: -d["visits"])
        # elección: argmax o muestreo con temperatura
        ply = len(st._hist)
        if self.temperature > 0 and ply < self.temp_plies and len(moves) > 1:
            w = [max(d["visits"], 1e-9) ** (1.0 / self.temperature) for d in cand]
            pick = self.rng.choices(cand, weights=w)[0]
        else:
            pick = cand[0]
        q_root = (sum(ch.W for ch in root.children.values()) / total)
        self.last_info = {"engine": "mcts", "sims": n_sims,
                          "time": round(time.time() - t0, 3),
                          "value": round(q_root, 4),
                          "win_prob": round((q_root + 1) / 2, 4),
                          "sampled": pick is not cand[0],
                          "policy": cand}
        return tuple(pick["move"])