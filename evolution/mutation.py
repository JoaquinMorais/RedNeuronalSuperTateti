"""Mutación de pesos y selección de padres."""
from __future__ import annotations

import random

import torch

from .config import Config, MutationLevel
from .rules import CHILDREN


def is_mutable(name: str, mutate_value_head: bool) -> bool:
    """Por defecto solo mutan las capas que influyen en la jugada (shared + policy_head)."""
    if name.startswith("value_head."):
        return mutate_value_head
    return True


def mutate_state_dict(parent: dict, level: MutationLevel, seed: int, mutate_value_head: bool = False) -> dict:
    """Devuelve una COPIA INDEPENDIENTE del padre con mutaciones (el padre no se toca)."""
    gen = torch.Generator().manual_seed(seed)
    child = {k: v.detach().clone() for k, v in parent.items()}
    for name in sorted(child):
        t = child[name]
        if not is_mutable(name, mutate_value_head) or not torch.is_floating_point(t):
            continue
        scale = max(float(t.std()) if t.numel() > 1 else abs(float(t)), 1e-3)
        noise = torch.randn(t.shape, generator=gen) * (level.sigma * scale)
        mask = (torch.rand(t.shape, generator=gen) < level.rate).to(t.dtype)
        child[name] = t + noise * mask
    return child


def pick_level(probs: dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for name in ("small", "medium", "large"):
        acc += probs[name]
        if r < acc:
            return name
    return "large"


def select_parents(
    ranked: list[str],
    totals: dict[str, int],
    survivors: set[str],
    rng: random.Random,
    power: float,
    min_weight: float,
    max_per_parent: int,
    min_from_non_survivors: int,
    n_children: int = CHILDREN,
) -> list[str]:
    """Elige exactamente ``n_children`` padres (con repetición, tope por padre).

    Probabilidad proporcional a max(puntos_totales,0)**power. Se garantizan al menos
    ``min_from_non_survivors`` hijos de IA que NO están entre los supervivientes, para
    que la descendencia no sea exclusiva del top conservado.
    ``ranked`` es la clasificación final (mejor primero) y fija el orden canónico.
    """
    weight = {p: max(totals[p], 0) ** power + min_weight for p in ranked}
    counts = {p: 0 for p in ranked}
    chosen: list[str] = []

    def draw(pool: list[str]) -> str | None:
        pool = [p for p in pool if counts[p] < max_per_parent]
        if not pool:
            return None
        total = sum(weight[p] for p in pool)
        r = rng.random() * total
        acc = 0.0
        for p in pool:
            acc += weight[p]
            if r < acc:
                return p
        return pool[-1]

    outsiders = [p for p in ranked if p not in survivors]
    for _ in range(min(min_from_non_survivors, n_children)):
        p = draw(outsiders)
        if p is None:
            break
        counts[p] += 1
        chosen.append(p)
    while len(chosen) < n_children:
        p = draw(ranked)
        if p is None:
            raise ValueError("No hay capacidad para asignar los descendientes")
        counts[p] += 1
        chosen.append(p)
    return chosen
