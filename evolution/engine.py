"""Ejecución de partidas: la IA elige el movimiento directamente con la red.

SIN MCTS. Selección de jugada: la red devuelve una política (softmax) ya enmascarada
con los movimientos legales (``game.get_valid_moves()``); se juega el legal de mayor
probabilidad (argmax; en empate exacto, el de menor índice). Es determinista, por lo
que una partida interrumpida puede repetirse y da exactamente el mismo resultado.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from game import SuperTateti

torch.set_num_threads(1)  # redes diminutas: más rápido y reproducible con 1 hilo

X, O, DRAW = "x", "o", "draw"


log = logging.getLogger("evolution.engine")
_WARNED: set[int] = set()


class IllegalMoveError(RuntimeError):
    pass


def net_is_finite(net) -> bool:
    """¿La política de la red es finita en la posición inicial? (pesos enormes desbordan float32 -> NaN).
    Solo cuenta la política: es lo único que se usa para jugar (la cabeza de valor no interviene)."""
    g = SuperTateti()
    with torch.no_grad():
        policy, _value = net(torch.from_numpy(g.get_state()).unsqueeze(0), torch.from_numpy(g.get_valid_moves()).unsqueeze(0))
    return bool(torch.isfinite(policy).all())


def select_move(net, game: SuperTateti) -> int:
    """Índice plano (0..80) del mejor movimiento legal según la red.

    Si la red produce valores no finitos (NaN/inf por pesos gigantes), ``softmax`` da NaN en TODA la fila y un
    ``argmax`` ingenuo devolvería la casilla 0 aunque sea ilegal. En ese caso se juega el primer movimiento
    legal (respaldo determinista): la red "rota" pierde y es eliminada, pero el entrenamiento no se detiene."""
    mask = game.get_valid_moves()
    state = torch.from_numpy(game.get_state()).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0)
    with torch.no_grad():
        policy, _value = net(state, mask_t)
    if not torch.isfinite(policy).all():
        if id(net) not in _WARNED:
            _WARNED.add(id(net))
            log.warning("Una red produce valores no finitos (pesos demasiado grandes): juega con el movimiento de respaldo.")
        policy = torch.nan_to_num(policy, nan=0.0, posinf=1.0, neginf=0.0)
    # argmax SOLO entre movimientos legales (las probabilidades de los legales son >= 0; las ilegales quedan en -1)
    move = int(torch.argmax(policy.masked_fill(mask_t == 0, -1.0), dim=-1).item())
    if mask[move] != 1:  # no debería ocurrir
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