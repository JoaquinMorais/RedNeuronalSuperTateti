"""RiverPt: bot estático basado en el motor MCTS de un amigo (``_super_tateti_engine.py``).

Este archivo es solo la CAPA DE COMUNICACIÓN entre el torneo y el motor:
  * convierte la partida (``game.SuperTateti``: jugadores 1/-1, tableros 0.5/0/1/-1) al ``State`` del motor (1/2, 0/1/2/3);
  * aplica la regla de ESTE juego (si el tablero de destino está cerrado, el rival debe jugar en el último tablero
    abierto del historial; el motor original daba jugada libre);
  * siembra el azar del motor por partida (reproducible al reanudar) y fija un presupuesto de simulaciones fijo
    (no por tiempo, para que el resultado no dependa de la carga de la máquina).

Con SIMS=5 el bot casi no "piensa": es el MCTS en su versión mínima (la heurística del motor decide casi todo).
Para subirle el nivel basta aumentar SIMS (300 ≈ 13 ms por jugada; 600 ≈ 26 ms).
"""
from __future__ import annotations

from models_statics.RiverPt.super_tateti_engine import MCTSBot, State
from models_statics.base import StaticBot


class _OurState(State):
    """``State`` del motor con la regla de envío de este repo (ver ``game.SuperTateti.get_next_active_board``)."""

    def __init__(self):
        super().__init__()
        self._sq = []       # tableros jugados, en orden (historial que usa la regla)
        self._sq_pushed = []

    @classmethod
    def from_game(cls, game) -> "_OurState":
        s = cls()
        board = game.board.tolist()
        small = game.small_boards.tolist()
        for b in range(9):
            s.cells[b] = [0 if v == 0 else (1 if v == 1 else 2) for v in board[b]]
            s.status[b] = 0 if small[b] == 0.5 else 3 if small[b] == 0 else (1 if small[b] == 1 else 2)
        s.player = 1 if game.current_player == 1 else 2
        s.next = -1 if game.active_board is None else int(game.active_board)
        s._sq = [int(x) for x in game.squares_played]
        s._hist = [None] * game.turn        # el MCTS usa len(_hist) como nº de jugada; solo se deshace lo que juega la búsqueda
        return s

    def play(self, b, c):
        pushed = not self._sq or self._sq[-1] != b
        if pushed:
            self._sq.append(b)
        self._sq_pushed.append(pushed)
        super().play(b, c)
        if self.status[c] == 0:
            self.next = c
        else:                                # destino cerrado: vuelve al último tablero abierto del historial
            self.next = -1
            for x in reversed(self._sq):
                if self.status[x] == 0:
                    self.next = x
                    break

    def undo(self):
        super().undo()
        if self._sq_pushed.pop():
            self._sq.pop()


class RiverPt(StaticBot):
    NAME = "RiverPt"
    COLOR1 = "#ffffff"      # blanco (identidad en la base de datos: no puede repetirse con otra IA)
    COLOR2 = "#ff0000"      # rojo

    SIMS = 5                # simulaciones MCTS por jugada
    TEMPERATURE = 1.0       # variedad: en las primeras jugadas se sortea según las visitas
    TEMP_PLIES = 10
    DIRICHLET_ALPHA = 0.3   # ruido en la raíz (0 = sin ruido)
    NOISE_FRAC = 0.25

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._engine = None

    def reseed(self, seed: int) -> None:
        super().reseed(seed)
        self._engine = None                 # se recrea con el rng nuevo

    def _mcts(self) -> MCTSBot:
        if self._engine is None:
            self._engine = MCTSBot(sims=self.SIMS, temperature=self.TEMPERATURE, temp_plies=self.TEMP_PLIES,
                                   dirichlet_alpha=self.DIRICHLET_ALPHA, noise_frac=self.NOISE_FRAC, rng=self.rng)
        return self._engine

    def choose(self, game) -> tuple[int, int]:
        b, c = self._mcts().choose(_OurState.from_game(game))
        return int(b), int(c)