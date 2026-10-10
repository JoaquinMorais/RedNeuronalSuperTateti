"""Reglas puras del torneo (sin base de datos ni torch).

Todo lo que está aquí es determinista dado el sorteo (``rng``) que se le pase,
lo que permite probarlo de forma aislada.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

POPULATION = 20
SURVIVORS = 7
CHILDREN = POPULATION - SURVIVORS  # 13
LEAGUE_MATCHDAYS = 38
RECOPA_GAMES = 10
RECOPA_BONUS = 5

# Puntos de PARTIDO (tabla de Liga y grupos de Copa)
WIN_POINTS, DRAW_POINTS, LOSS_POINTS = 3, 1, 0

# Recompensa evolutiva de la Liga según posición final (índice 0 = 1.º)
LEAGUE_REWARD = [30, 24, 21, 18, 15, 13, 11, 9, 7, 6, 5, 4, 3, 2, 1, 0, 0, -3, -3, -3]

# Recompensa evolutiva de la Copa según la fase alcanzada (no se acumula)
CUP_REWARD = {
    "champion": 24,
    "runner_up": 17,
    "SF": 12,   # eliminado en semifinales
    "QF": 7,    # eliminado en cuartos
    "R16": 3,   # eliminado en octavos
    "group": 0,  # eliminado en fase de grupos
}

# Fechas de Liga en las que se juega cada ronda eliminatoria
KNOCKOUT_MATCHDAYS = {"R16": 18, "QF": 23, "SF": 30, "F": 37}
KNOCKOUT_ROUNDS = ["R16", "QF", "SF", "F"]
# Las 10 rondas de grupos se reparten en las fechas 1-12 dejando 6 y 12 de margen
GROUP_ROUND_MATCHDAYS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

# Cruces iniciales de octavos (posición+grupo), en orden de cuadro
R16_CROSSINGS = [
    ("1A", "4B"), ("2A", "3B"), ("3A", "2B"), ("4A", "1B"),
    ("1C", "4D"), ("2C", "3D"), ("3C", "2D"), ("4C", "1D"),
]

X, O, DRAW = "x", "o", "draw"  # resultado de una partida (X = quien comienza)


def match_points(result: str, side: str) -> int:
    """Puntos de partido para el lado ``side`` ('x' u 'o')."""
    if result == DRAW:
        return DRAW_POINTS
    return WIN_POINTS if result == side else LOSS_POINTS


# --------------------------------------------------------------------------
# Desempates por sorteo reproducible
# --------------------------------------------------------------------------
@dataclass
class DrawRecord:
    """Un sorteo de desempate que hay que persistir."""
    context: str
    members: list[str]       # identificadores (colores) en orden canónico
    result_order: list[str]  # orden resultante tras el sorteo


def shuffled_canonical(members: Sequence[str], rng: random.Random) -> list[str]:
    """Baraja ``members`` partiendo de un orden canónico (alfabético).

    Ordenar antes de barajar evita que el resultado dependa del orden de inserción
    en la base de datos o en memoria: mismo conjunto + misma semilla = mismo orden.
    """
    canon = sorted(members)
    rng.shuffle(canon)
    return canon


def rank_with_draws(
    items: Sequence[str],
    key: Callable[[str], Hashable],
    context: str,
    rng_factory: Callable[[str], random.Random],
    draws: list[DrawRecord],
) -> list[str]:
    """Ordena ``items`` por ``key`` ascendente (menor = mejor) y resuelve por sorteo
    los grupos que sigan empatados (parejas o empates múltiples).

    Cada grupo empatado se sortea una sola vez, con una semilla derivada de
    ``context`` y de la clave de empate, y se agrega a ``draws``.
    """
    groups: dict[Hashable, list[str]] = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    ordered: list[str] = []
    for k in sorted(groups):
        grp = groups[k]
        if len(grp) > 1:
            ctx = f"{context}|{k}"
            order = shuffled_canonical(grp, rng_factory(ctx))
            draws.append(DrawRecord(ctx, sorted(grp), order))
            grp = order
        ordered.extend(grp)
    return ordered


# --------------------------------------------------------------------------
# Tablas (Liga y grupos)
# --------------------------------------------------------------------------
@dataclass
class Row:
    pid: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0

    def add(self, result: str, side: str) -> None:
        self.played += 1
        if result == DRAW:
            self.draws += 1
        elif result == side:
            self.wins += 1
        else:
            self.losses += 1
        self.points += match_points(result, side)


def build_table(pids: Sequence[str], games: Sequence[tuple[str, str, str]]) -> dict[str, Row]:
    """``games`` = (id_X, id_O, resultado). Devuelve una fila por participante."""
    table = {p: Row(p) for p in pids}
    for x, o, res in games:
        table[x].add(res, X)
        table[o].add(res, O)
    return table


def league_final_order(
    pids: Sequence[str],
    final_points: dict[str, int],
    history: dict[str, list[int]],
    rng_factory: Callable[[str], random.Random],
    draws: list[DrawRecord],
) -> list[str]:
    """Clasificación final de Liga.

    1. Más puntos.
    2. Entre empatados: quien alcanzó ANTES ese total (primera fecha en que su
       acumulado igualó su total final; ``history[pid][d-1]`` = puntos tras la fecha d).
    3. Si sigue el empate (misma fecha): sorteo reproducible.
    """
    def first_reach(pid: str) -> int:
        target = final_points[pid]
        for day, pts in enumerate(history[pid], start=1):
            if pts >= target:
                return day
        return len(history[pid]) + 1

    return rank_with_draws(
        list(pids),
        key=lambda p: (-final_points[p], first_reach(p)),
        context="league",
        rng_factory=rng_factory,
        draws=draws,
    )


# --------------------------------------------------------------------------
# Eliminatorias de Copa
# --------------------------------------------------------------------------
def leg_winner(result: str, x: str, o: str) -> str | None:
    if result == X:
        return x
    if result == O:
        return o
    return None


def evaluate_two_legs(a: str, b: str, legs: Sequence[tuple[str, str, str]]):
    """``legs`` = [(x, o, resultado), ...] de las dos partidas.

    Devuelve (ganador|None, resolución). None => eliminatoria empatada.
    """
    wins = {a: 0, b: 0}
    for x, o, res in legs:
        w = leg_winner(res, x, o)
        if w is not None:
            wins[w] += 1
    if wins[a] == 2:
        return a, "two_wins"
    if wins[b] == 2:
        return b, "two_wins"
    if wins[a] == 1 and wins[b] == 0:
        return a, "win_and_draw"
    if wins[b] == 1 and wins[a] == 0:
        return b, "win_and_draw"
    return None, "tied"  # 1-1 o 0-0


def evaluate_third_game(x: str, o: str, result: str):
    """Desempate: si empata, gana quien comenzó en segundo lugar (O)."""
    if result == X:
        return x, "third_game"
    if result == O:
        return o, "third_game"
    return o, "third_game_draw_second_starter"


def knockout_next_round(round_name: str) -> str | None:
    i = KNOCKOUT_ROUNDS.index(round_name)
    return KNOCKOUT_ROUNDS[i + 1] if i + 1 < len(KNOCKOUT_ROUNDS) else None


def cup_reward_for_stage(stage: str) -> int:
    return CUP_REWARD[stage]


# --------------------------------------------------------------------------
# Recopa
# --------------------------------------------------------------------------
def recopa_outcome(cup_champ: str, league_champ: str, games: Sequence[tuple[str, str, str]]):
    """Cuenta victorias de cada campeón. Los empates de partida valen 0 victorias
    para ambos (misma regla para los dos): solo se comparan victorias.

    Devuelve (wins_cup, wins_league, draws, ganador|None).
    """
    w = {cup_champ: 0, league_champ: 0}
    d = 0
    for x, o, res in games:
        win = leg_winner(res, x, o)
        if win is None:
            d += 1
        else:
            w[win] += 1
    winner = None
    if w[cup_champ] > w[league_champ]:
        winner = cup_champ
    elif w[league_champ] > w[cup_champ]:
        winner = league_champ
    return w[cup_champ], w[league_champ], d, winner


# --------------------------------------------------------------------------
# Clasificación global
# --------------------------------------------------------------------------
def global_order(
    pids: Sequence[str],
    league_score: dict[str, int],
    cup_score: dict[str, int],
    recopa_score: dict[str, int],
    league_position: dict[str, int],
    rng_factory: Callable[[str], random.Random],
    draws: list[DrawRecord],
) -> list[str]:
    """Orden final por puntos totales (Copa + Liga + Recopa), de mayor a menor.

    Desempates deterministas: 1) mejor posición final en Liga; 2) más puntos de Copa;
    3) sorteo reproducible (solo si todo lo anterior coincide).
    """
    def total(p: str) -> int:
        return league_score[p] + cup_score[p] + recopa_score[p]

    return rank_with_draws(
        list(pids),
        key=lambda p: (-total(p), league_position[p], -cup_score[p]),
        context="global",
        rng_factory=rng_factory,
        draws=draws,
    )
