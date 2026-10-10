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


def parents_by_rank(ranked: list[str], children_per_rank, n_children: int = CHILDREN) -> list[str]:
    """Padre de cada descendiente según el puesto en la tabla anual (``ranked``: 1.º primero).

    Con el criterio por defecto (5, 3, 2, 2, 1, 0, 0): el 1.º tiene 5 hijos, el 2.º 3, etc.
    Es determinista (sin azar) y siempre devuelve exactamente ``n_children`` padres.
    """
    parents: list[str] = []
    for color, n in zip(ranked, children_per_rank):
        parents += [color] * n
    if len(parents) != n_children:
        raise ValueError(f"La política produce {len(parents)} descendientes y debe producir {n_children}")
    return parents