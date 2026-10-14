"""Oráculo de "mate en N" y generador de puzzles tácticos.

Un puzzle es una posición de ``game.SuperTateti`` a la que le toca mover y en la
que el jugador que mueve tiene **victoria forzada en exactamente N jugadas
propias** (contra cualquier defensa). El nivel se etiqueta con el *mínimo* N; si
hay victoria más rápida, el puzzle pertenece a ese nivel menor.

Nada de este módulo depende de torch: es Python + numpy puro sobre ``game.py``.

Definiciones
------------
* Nodo "mío": le toca mover al jugador raíz. Basta **una** jugada que gane.
* Nodo "rival": le toca al rival. **Todas** sus respuestas deben seguir ganando
  para mí (búsqueda AND/OR).
* ``can_win`` usa a lo sumo ``remaining`` jugadas *mías* (cada jugada mía consume
  una; las del rival no cuentan).

La búsqueda lleva memoización por estado + ``remaining`` y un presupuesto de
nodos (``max_nodes``) para no colgarse en posiciones sin mate.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from game import WIN_COMBOS, SuperTateti

MAX_MOVES = 81


def legal_moves(game: SuperTateti) -> list[int]:
    """Índices planos (0..80) de los movimientos legales."""
    mask = game.get_valid_moves()
    return [i for i in range(MAX_MOVES) if mask[i] == 1.0]


def state_key(game: SuperTateti) -> tuple:
    """Clave hasheable y estable de la posición (sin el contador de jugadas)."""
    return (game.board.tobytes(), game.small_boards.tobytes(),
            game.current_player, game.active_board)


def _completes_line(cells, idx: int, player: float) -> bool:
    """¿Poner ``player`` en la casilla ``idx`` completa tres en línea en ese tablero?"""
    for a, b, c in WIN_COMBOS:
        if idx in (a, b, c):
            o1, o2 = [q for q in (a, b, c) if q != idx]
            if cells[o1] == player and cells[o2] == player:
                return True
    return False


def immediate_win_moves(game: SuperTateti) -> list[int]:
    """Movimientos que ganan la partida de inmediato (definen el nivel 1).

    Optimizado: una jugada gana el macro-tablero si cierra un tablero chico que a su
    vez completa una línea grande ya con dos tableros míos. Solo mira esos tableros.
    """
    if game.winner is not None:
        return []
    me = game.current_player
    sb = game.small_boards
    targets = set()
    for a, b, c in WIN_COMBOS:
        vals = (float(sb[a]), float(sb[b]), float(sb[c]))
        if vals.count(float(me)) == 2 and vals.count(0.5) == 1:
            targets.add((a, b, c)[vals.index(0.5)])
    if not targets:
        return []
    mask = game.get_valid_moves()
    out = []
    for b in targets:
        cells = game.board[b]
        for cell in range(9):
            mv = b * 9 + cell
            if cells[cell] == 0 and mask[mv] == 1 and _completes_line(cells, cell, me):
                out.append(mv)
    return out


class NodeBudgetExceeded(Exception):
    """La búsqueda superó el presupuesto de nodos (posición demasiado difícil)."""


class MateSolver:
    """Busca victorias forzadas. Reutilizable: llama a ``reset()`` entre posiciones."""

    def __init__(self, max_nodes: int = 0):
        self.max_nodes = max_nodes          # 0 = sin límite
        self.nodes = 0
        self.memo: dict = {}

    def reset(self) -> None:
        self.nodes = 0
        self.memo = {}

    def _bump(self) -> None:
        self.nodes += 1
        if self.max_nodes and self.nodes > self.max_nodes:
            raise NodeBudgetExceeded()

    # -- búsqueda AND/OR ------------------------------------------------
    def can_win(self, game: SuperTateti, remaining: int) -> bool:
        """¿El jugador que mueve fuerza la victoria en <= ``remaining`` jugadas propias?"""
        self._bump()
        if game.winner is not None:
            return game.winner == game.current_player
        if remaining <= 0:
            return False
        key = (game.board.tobytes(), game.small_boards.tobytes(),
               game.current_player, game.active_board, remaining, 0)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        if remaining == 1:                  # una sola jugada: ¿hay mate inmediato?
            result = bool(immediate_win_moves(game))
            self.memo[key] = result
            return result
        me = game.current_player
        result = False
        for mv in legal_moves(game):
            child = game.clone()
            child.make_move(mv // 9, mv % 9)
            if child.winner is not None:
                if child.winner == me:      # gano ya
                    result = True
                    break
                continue                    # empate/derrota: esta jugada no sirve
            if self._all_replies_lose(child, remaining - 1):
                result = True
                break
        self.memo[key] = result
        return result

    def _all_replies_lose(self, game: SuperTateti, remaining: int) -> bool:
        """Nodo del rival: True si TODA respuesta del rival deja la victoria forzada."""
        self._bump()
        key = (game.board.tobytes(), game.small_boards.tobytes(),
               game.current_player, game.active_board, remaining, 1)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        me = -game.current_player
        result = True
        for mv in legal_moves(game):
            child = game.clone()
            child.make_move(mv // 9, mv % 9)
            if child.winner is not None:
                if child.winner != me:      # el rival escapa (gana o empata)
                    result = False
                    break
                continue                    # extraño pero inofensivo: sigo ganando
            if not self.can_win(child, remaining):
                result = False
                break
        self.memo[key] = result
        return result

    # -- consultas ------------------------------------------------------
    def min_mate(self, game: SuperTateti, max_n: int = 4) -> int | None:
        """Menor N de victoria forzada (<= ``max_n``), o ``None`` si no hay."""
        if game.winner is not None:
            return None
        for n in range(1, max_n + 1):
            if self.can_win(game, n):
                return n
        return None

    def optimal_moves(self, game: SuperTateti, n: int) -> list[int]:
        """Jugadas que conservan el mate en N (todas mantienen la victoria forzada)."""
        me = game.current_player
        out = []
        for mv in legal_moves(game):
            child = game.clone()
            child.make_move(mv // 9, mv % 9)
            if child.winner is not None:
                if child.winner == me:
                    out.append(mv)
                continue
            if self._all_replies_lose(child, n - 1):
                out.append(mv)
        return out


# ----------------------------------------------------------------------
# Generación de puzzles
# ----------------------------------------------------------------------
@dataclass(eq=False)
class Puzzle:
    """Estado listo para entrenar + etiqueta de nivel y jugadas óptimas."""
    state: np.ndarray      # (100,) float32 (game.get_state)
    mask: np.ndarray       # (81,) float32  (game.get_valid_moves)
    level: int             # N mínimo de mate
    best_moves: tuple      # índices planos legales que conservan el mate en N
    turn: int
    key: tuple             # state_key() para deduplicar/evitar solapamientos

    def __len__(self) -> int:
        return int(self.state.shape[0])


def make_puzzle(game: SuperTateti, level: int, best_moves) -> Puzzle:
    return Puzzle(game.get_state(), game.get_valid_moves(), level,
                  tuple(sorted(best_moves)), game.turn, state_key(game))


def _random_move(game: SuperTateti, rng: random.Random) -> int | None:
    moves = legal_moves(game)
    return rng.choice(moves) if moves else None


def play_random_game(rng: random.Random, stop_at_turn: int | None = None):
    """Partida aleatoria reproducible; devuelve la lista de posiciones (clones) jugadas.

    Si ``stop_at_turn`` se indica, corta cuando se alcanza ese turno (para no gastar
    tiempo en fases iniciales sin mates posibles).
    """
    game = SuperTateti()
    positions = []
    while game.winner is None:
        positions.append(game.clone())
        if stop_at_turn is not None and game.turn >= stop_at_turn:
            break
        mv = _random_move(game, rng)
        if mv is None:
            break
        game.make_move(mv // 9, mv % 9)
    return positions


def harvest(
    level_targets: dict[int, int],
    seed: int,
    max_n: int = 4,
    exclude: set | None = None,
    min_turn: int = 16,
    solver_max_nodes: int = 0,
    games_budget: int = 8000,
    log=None,
    log_interval: float = 10.0,
    should_stop=None,
) -> dict[int, list[Puzzle]]:
    """Genera puzzles por nivel por muestreo aleatorio de partidas + filtro del solver.

    ``level_targets``: {nivel: cantidad objetivo}. ``exclude``: claves de estado a
    evitar (p. ej. el set de evaluación). ``should_stop``: callable opcional que, si
    devuelve True, corta la generación (para paradas prolijas). Devuelve
    {nivel: [Puzzle, ...]}.
    """
    level_targets = {int(k): int(v) for k, v in level_targets.items() if v > 0}
    if not level_targets:
        return {}
    rng = random.Random(seed)
    exclude = set() if exclude is None else set(exclude)
    found: dict[int, dict] = {n: {} for n in level_targets}
    solver = MateSolver(max_nodes=solver_max_nodes)
    top_n = max(level_targets)

    def pending() -> list[int]:
        return [n for n, t in level_targets.items() if len(found[n]) < t]

    games = 0
    last_log = time.time()
    while games < games_budget and pending():
        if should_stop is not None and should_stop():
            break
        games += 1
        game = SuperTateti()
        while game.winner is None and pending():
            need = pending()
            key = state_key(game)
            if key not in exclude:
                if 1 in need:
                    im = immediate_win_moves(game)
                    if im:
                        found[1].setdefault(key, make_puzzle(game, 1, im))
                deep = [n for n in need if n >= 2]
                if deep and game.turn >= min_turn:
                    solver.reset()
                    try:
                        n = solver.min_mate(game, top_n)
                        if n is not None and n >= 2 and n in found and len(found[n]) < level_targets[n]:
                            best = solver.optimal_moves(game, n)
                            if best:
                                found[n].setdefault(key, make_puzzle(game, n, best))
                    except NodeBudgetExceeded:
                        pass
            mv = _random_move(game, rng)
            if mv is None:
                break
            game.make_move(mv // 9, mv % 9)
        if log is not None and time.time() - last_log >= log_interval:
            last_log = time.time()
            log({n: len(found[n]) for n in level_targets}, games)

    return {n: list(found[n].values()) for n in level_targets}
