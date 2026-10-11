"""Rivales FIJOS para medir progreso de forma absoluta (no dependen de la población que evoluciona).

* ``RandomBot``: elige al azar entre los movimientos legales. Es el piso: una IA que no le gana
  claramente no ha aprendido nada del juego.
* ``TacticalBot``: mira 2 jugadas (la suya y la respuesta del rival): gana si puede, no regala la
  victoria, gana tableros chicos, bloquea los del rival y evita enviarlo a un tablero donde cierre uno.
  Es un jugador "decente" sin ninguna red: sirve de vara intermedia.
Los dos usan su propio ``random.Random`` (semilla guardada) para desempatar: resultados reproducibles.
"""
from __future__ import annotations

import random

from game import WIN_COMBOS, SuperTateti


def legal_moves(game: SuperTateti) -> list[tuple[int, int]]:
    mask = game.get_valid_moves()
    return [(i // 9, i % 9) for i in range(81) if mask[i] == 1]


class RandomBot:
    name = "random"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, game: SuperTateti) -> tuple[int, int]:
        return self.rng.choice(legal_moves(game))


def _completes_line(cells, idx: int, player: float) -> bool:
    """¿Poner ``player`` en ``idx`` completa una línea de 3 en ese tablero chico?"""
    for a, b, c in WIN_COMBOS:
        if idx in (a, b, c):
            o1, o2 = [x for x in (a, b, c) if x != idx]
            if cells[o1] == player and cells[o2] == player:
                return True
    return False


class TacticalBot:
    name = "tactical"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, game: SuperTateti) -> tuple[int, int]:
        me = game.current_player
        best, best_score = [], -1e9
        for b, c in legal_moves(game):
            if game.small_boards[b] != 0.5:
                continue
            score = 0.0
            sim = game.clone()
            sim.make_move(b, c)
            if sim.winner == me:
                return b, c                                   # gana la partida ya
            if _completes_line(game.board[b], c, me):
                score += 10                                    # gana un tablero chico
            if _completes_line(game.board[b], c, -me):
                score += 6                                     # bloquea uno del rival
            if sim.winner is None:
                reply_wins, reply_small = False, False
                for rb, rc in legal_moves(sim):
                    if _completes_line(sim.board[rb], rc, -me):
                        reply_small = True
                        s2 = sim.clone(); s2.make_move(rb, rc)
                        if s2.winner == -me:
                            reply_wins = True
                            break
                if reply_wins:
                    score -= 100                               # regalaría la partida
                elif reply_small:
                    score -= 5                                 # le deja cerrar un tablero chico
            if c == 4:
                score += 1.0                                   # centro del tablero chico
            if b == 4:
                score += 0.5
            if score > best_score:
                best, best_score = [(b, c)], score
            elif score == best_score:
                best.append((b, c))
        return self.rng.choice(best)


BOTS = {"random": RandomBot, "tactical": TacticalBot}