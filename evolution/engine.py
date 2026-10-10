"""Ejecución de partidas: la IA elige el movimiento directamente con la red.

SIN MCTS. Selección de jugada: la red devuelve una política (softmax) ya enmascarada
con los movimientos legales (``game.get_valid_moves()``); se juega el legal de mayor
probabilidad (argmax; en empate exacto, el de menor índice). Es determinista, por lo
que una partida interrumpida puede repetirse y da exactamente el mismo resultado.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from game import SuperTateti

torch.set_num_threads(1)  # redes diminutas: más rápido y reproducible con 1 hilo

X, O, DRAW = "x", "o", "draw"


class IllegalMoveError(RuntimeError):
    pass


def select_move(net, game: SuperTateti) -> int:
    """Índice plano (0..80) del mejor movimiento legal según la red."""
    mask = game.get_valid_moves()
    state = torch.from_numpy(game.get_state()).unsqueeze(0)
    with torch.no_grad():
        policy, _value = net(state, torch.from_numpy(mask).unsqueeze(0))
    move = int(torch.argmax(policy, dim=-1).item())
    if mask[move] != 1:  # no debería ocurrir: la política está enmascarada
        raise IllegalMoveError(f"La red eligió un movimiento ilegal: {move}")
    return move


@dataclass
class MatchOutcome:
    result: str          # 'x' | 'o' | 'draw'
    moves: str           # jugadas planas separadas por comas (auditoría/reproducción)
    num_moves: int
    duration_ms: float


def play_match(net_x, net_o) -> MatchOutcome:
    """``net_x`` comienza (X, jugador 1). Devuelve resultado, jugadas y duración."""
    t0 = time.perf_counter()
    game = SuperTateti()
    moves: list[int] = []
    while game.winner is None:
        net = net_x if game.current_player == 1 else net_o
        mv = select_move(net, game)
        if not game.make_move(mv // 9, mv % 9):
            raise IllegalMoveError(f"make_move rechazó {mv}")
        moves.append(mv)
    w = float(game.winner)
    result = X if w == 1.0 else O if w == -1.0 else DRAW
    return MatchOutcome(result, ",".join(map(str, moves)), len(moves),
                        (time.perf_counter() - t0) * 1000.0)
