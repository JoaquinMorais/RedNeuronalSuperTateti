"""Calendarios (funciones puras): Liga, grupos de Copa y cuadro eliminatorio."""
from __future__ import annotations

import random
from typing import Sequence

from . import rules


def round_robin(n_teams: int) -> list[list[tuple[int, int]]]:
    """Método del círculo. Devuelve rondas de pares (a, b) con índices 0..n-1.

    El primer elemento de cada par es quien comienza (X) en la IDA. Con n impar se
    añade un descanso y los pares con el descanso se descartan.
    El inicio se alterna entre rondas/posiciones para repartir quién comienza.
    """
    teams: list[int | None] = list(range(n_teams))
    if n_teams % 2:
        teams.append(None)
    n = len(teams)
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = teams[i], teams[n - 1 - i]
            if a is None or b is None:
                continue
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        teams = [teams[0], teams[-1], *teams[1:-1]]  # rota todos menos el primero
    return rounds


def double_round_robin(n_teams: int) -> list[list[tuple[int, int]]]:
    """Ida y vuelta: la vuelta repite los cruces con el inicio invertido."""
    first = round_robin(n_teams)
    second = [[(b, a) for a, b in rnd] for rnd in first]
    return first + second


def league_fixtures() -> list[list[tuple[int, int]]]:
    """38 fechas x 10 partidos. Pares de slots 1..20: (slot que comienza, slot que responde)."""
    rounds = double_round_robin(rules.POPULATION)
    return [[(a + 1, b + 1) for a, b in rnd] for rnd in rounds]


def group_fixtures() -> list[tuple[int, list[tuple[int, int]]]]:
    """Fixture de un grupo de 5 (posiciones 0..4): 10 rondas de 2 partidos.

    Devuelve [(fecha_de_liga, [(x, o), ...]), ...] usando ``rules.GROUP_ROUND_MATCHDAYS``.
    """
    rounds = double_round_robin(5)
    assert len(rounds) == len(rules.GROUP_ROUND_MATCHDAYS) == 10
    return [(md, rnd) for md, rnd in zip(rules.GROUP_ROUND_MATCHDAYS, rounds)]


def draw_groups(pids: Sequence[str], rng: random.Random) -> dict[str, list[str]]:
    """Sorteo de 4 grupos de 5. Parte de orden canónico para ser reproducible."""
    order = sorted(pids)
    rng.shuffle(order)
    return {name: order[i * 5:(i + 1) * 5] for i, name in enumerate("ABCD")}


def knockout_bracket() -> list[dict]:
    """Los 15 cruces del cuadro. Octavos con los cruces exactos de la especificación;
    cada cruce posterior enfrenta a los ganadores de dos cruces consecutivos
    (1v2, 3v4, ...) en el orden del cuadro."""
    ties = []
    for i, (a, b) in enumerate(rules.R16_CROSSINGS, start=1):
        ties.append({"round": "R16", "tie_no": i, "slot_a": a, "slot_b": b})
    sizes = {"QF": 4, "SF": 2, "F": 1}
    prev = "R16"
    for rnd in ("QF", "SF", "F"):
        for k in range(1, sizes[rnd] + 1):
            ties.append({
                "round": rnd, "tie_no": k,
                "slot_a": f"W:{prev}-{2 * k - 1}", "slot_b": f"W:{prev}-{2 * k}",
            })
        prev = rnd
    for t in ties:
        t["matchday"] = rules.KNOCKOUT_MATCHDAYS[t["round"]]
    return ties
