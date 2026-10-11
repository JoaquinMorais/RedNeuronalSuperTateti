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


def league_fixtures(n_teams: int = rules.POPULATION) -> list[list[tuple[int, int]]]:
    """Fechas de Liga. Pares de slots 1..n: (slot que comienza, slot que responde).

    n=20 -> 38 fechas x 10 partidos. Con n impar descansa una IA por fecha (2·n fechas)."""
    rounds = double_round_robin(n_teams)
    return [[(a + 1, b + 1) for a, b in rnd] for rnd in rounds]


def group_round_matchdays(rounds_per_leg: int) -> list[int]:
    """Fechas de Liga de las rondas de un grupo: la ida en 1..L, una fecha de margen, la vuelta en L+2..2L+1
    (y el resto, hasta la fecha 17, queda de margen antes de los octavos). Con L=5 son las fechas 1-5 y 7-11."""
    L = rounds_per_leg
    return list(range(1, L + 1)) + list(range(L + 2, 2 * L + 2))


def group_fixtures(size: int = 5) -> list[tuple[int, list[tuple[int, int]]]]:
    """Fixture de un grupo de ``size`` equipos (posiciones 0..size-1), ida y vuelta.

    Devuelve [(fecha_de_liga, [(x, o), ...]), ...]. Con 5 o 6 equipos son 10 rondas (fechas 1-12);
    con 7 u 8, 14 rondas (fechas 1-15)."""
    rounds = double_round_robin(size)
    mds = group_round_matchdays(len(rounds) // 2)
    assert len(mds) == len(rounds) and mds[-1] < rules.KNOCKOUT_MATCHDAYS["R16"]
    return list(zip(mds, rounds))


def draw_groups(pids: Sequence[str], rng: random.Random) -> dict[str, list[str]]:
    """Sorteo de 4 grupos (A-D) con los tamaños de ``rules.season_format``. Parte de orden canónico (reproducible)."""
    sizes = rules.season_format(len(pids))["group_sizes"]
    order = sorted(pids)
    rng.shuffle(order)
    out, i = {}, 0
    for name, size in zip("ABCD", sizes):
        out[name] = order[i:i + size]
        i += size
    return out


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