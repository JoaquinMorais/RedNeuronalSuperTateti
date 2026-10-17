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


def _ref_layers(sd: dict) -> list:
    """Capas de referencia ``(nombre, fan_in)`` derivadas del propio ``state_dict``.

    Reconoce las capas que influyen en la jugada: ``shared.*`` y ``policy_head`` (MLP),
    ``conv*`` y ``policy_head`` (CNN). ``fan_in`` es la dimensión de entrada, incluyendo
    el área del kernel en convoluciones. Así el reescalado funciona con cualquier ancho,
    profundidad o arquitectura, sin listas hardcodeadas.
    """
    out = []
    for name in sorted(sd):
        if not name.endswith(".weight"):
            continue
        prefix = name[:-len(".weight")]
        if not (prefix.startswith("shared.") or prefix == "policy_head" or prefix.startswith("conv")):
            continue
        w = sd[name]
        fan_in = int(w.shape[1])
        if w.ndim == 4:
            fan_in *= int(w.shape[2]) * int(w.shape[3])
        out.append((prefix, fan_in))

    def _order(item):
        prefix = item[0]
        if prefix.startswith("shared."):
            return (0, int(prefix.split(".")[1]))
        if prefix.startswith("conv"):
            return (1, prefix)
        return (2, prefix)

    out.sort(key=_order)
    return out


def normalize_state_dict(sd: dict, threshold: float = 4.0) -> tuple[dict, bool]:
    """Reescala los pesos SIN cambiar lo que juega la red (la política), si crecieron demasiado.

    Problema: la mutación suma ruido proporcional al desvío ACTUAL de cada tensor, así que los pesos de un linaje
    crecen ~x1,019 por generación (x1e8 en 1000 generaciones) hasta desbordar float32: la red devuelve NaN.
    Solución: una red ReLU es homogénea positiva, así que multiplicar los pesos de una capa por m y sus
    sesgos por la escala acumulada (y compensar en la capa siguiente) deja la jugada idéntica. Si algún tensor
    de pesos supera ``threshold`` veces su desvío natural, TODAS las capas se llevan a su desvío natural.
    La cabeza de valor NO se toca: no interviene en la jugada y compensarla la haría crecer sin límite.
    Devuelve (nuevo_state_dict, ¿se reescaló?). ``threshold <= 0`` desactiva (comportamiento anterior)."""
    if threshold <= 0:
        return sd, False
    ref_layers = _ref_layers(sd)
    if not ref_layers:
        return sd, False
    d = {k: torch.nan_to_num(v.detach().double(), nan=0.0, posinf=0.0, neginf=0.0) if torch.is_floating_point(v) else v
         for k, v in sd.items()}
    stds = {n: float(d[f"{n}.weight"].std()) for n, _ in ref_layers}
    refs = {n: 1.0 / (fan_in ** 0.5) for n, fan_in in ref_layers}
    broken = any(not torch.isfinite(v).all() for v in sd.values() if torch.is_floating_point(v))
    if not broken and max(stds[n] / refs[n] for n, _ in ref_layers) <= threshold:
        return sd, False
    out = dict(d)
    scale = 1.0                                   # escala acumulada de la salida de cada capa
    for n, _ in ref_layers:
        m = refs[n] / max(stds[n], 1e-300)
        scale *= m
        out[f"{n}.weight"] = d[f"{n}.weight"] * m
        out[f"{n}.bias"] = d[f"{n}.bias"] * scale
    return {k: (v.float() if torch.is_floating_point(v) else v) for k, v in out.items()}, True


def mutate_state_dict(parent: dict, level: MutationLevel, seed: int, mutate_value_head: bool = False,
                      norm_threshold: float = 0.0) -> dict:
    """Devuelve una COPIA INDEPENDIENTE del padre con mutaciones (el padre no se toca).

    Con ``norm_threshold > 0`` el padre se reescala antes (ver ``normalize_state_dict``) para evitar la
    explosión de pesos a lo largo de los linajes; con 0 se comporta exactamente como antes."""
    parent, _ = normalize_state_dict(parent, norm_threshold)
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