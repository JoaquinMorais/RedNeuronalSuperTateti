"""Estadística mínima (solo numpy) para juzgar si hay progreso o solo ruido."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

SPARK = "▁▂▃▄▅▆▇█"


def outcome_stats(w: int, d: int, l: int) -> dict:
    """Puntaje (victoria 1, empate 0,5, derrota 0) con intervalo de confianza del 95 %."""
    n = w + d + l
    if n == 0:
        return {"n": 0, "score": float("nan"), "ci": float("nan")}
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    return {"n": n, "score": s, "ci": 1.96 * math.sqrt(var / n)}


def _ranks(a: Sequence[float]) -> np.ndarray:
    """Rangos 1..n con promedio en los empates."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def trend(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """¿La serie ``ys`` sube con ``xs``? Correlación de Spearman con umbral ~95 % (|rho| > 1.96/sqrt(n-1))
    y pendiente por mínimos cuadrados (en puntos de la serie por cada 100 unidades de ``xs``).
    También compara el primer tercio con el último."""
    n = len(xs)
    out = {"n": n, "verdict": "pocos datos", "rho": float("nan"), "threshold": float("nan"),
           "slope_per_100": float("nan"), "first": float("nan"), "last": float("nan")}
    if n < 6:
        return out
    rho = spearman(xs, ys)
    thr = 1.96 / math.sqrt(n - 1)
    slope = float(np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)[0] * 100)
    k = max(2, n // 3)
    first, last = float(np.mean(ys[:k])), float(np.mean(ys[-k:]))
    out.update(rho=rho, threshold=thr, slope_per_100=slope, first=first, last=last)
    out["verdict"] = ("MEJORA" if rho > thr and slope > 0 else "EMPEORA" if rho < -thr and slope < 0
                      else "SIN TENDENCIA CLARA")
    return out


def sparkline(values: Sequence[float], width: int = 50, lo: float = 0.0, hi: float = 1.0) -> str:
    """Mini gráfico de barras en texto (promedia en bloques si hay más valores que ``width``)."""
    vals = [v for v in values if v == v]
    if not vals:
        return ""
    if len(vals) > width:
        edges = np.linspace(0, len(vals), width + 1).astype(int)
        vals = [float(np.mean(vals[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, max(0, int((v - lo) / span * (len(SPARK) - 1) + 0.5)))] for v in vals)


def bradley_terry(n: int, scores: dict[tuple[int, int], float], games: dict[tuple[int, int], float],
                  iters: int = 500) -> list[float]:
    """Fuerza de Bradley-Terry (escala Elo, media 1000) a partir de puntajes por pareja.

    ``scores[(i, j)]`` = puntos que obtuvo i contra j (victoria 1, empate 0,5); ``games[(i, j)]`` = partidas.
    Se agrega un rival virtual de fuerza 1 (media partida) para evitar fuerzas 0 o infinitas."""
    p = np.ones(n)
    for _ in range(iters):
        new = np.empty(n)
        for i in range(n):
            wins = 0.5 + sum(scores.get((i, j), 0.0) for j in range(n) if j != i)
            den = 1.0 / (p[i] + 1.0) + sum(games.get((i, j), 0.0) / (p[i] + p[j]) for j in range(n) if j != i)
            new[i] = wins / den
        new /= math.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < 1e-9:
            p = new
            break
        p = new
    return [float(1000 + 400 * math.log10(x)) for x in p]