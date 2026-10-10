"""Títulos (Liga / Copa / Recopa) por IA, a lo largo de las temporadas. Funciones puras."""
from __future__ import annotations

from collections import defaultdict


def build_title_index(rows) -> dict[str, dict[str, list[int]]]:
    """``rows``: dicts con ``number``, ``league``, ``cup``, ``recopa`` (colores o None).

    Devuelve {color: {"L": [temporadas], "C": [...], "RC": [...]}} solo para quien ganó algo.
    """
    idx: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"L": [], "C": [], "RC": []})
    for r in sorted(rows, key=lambda r: r["number"]):
        for key, color in (("L", r["league"]), ("C", r["cup"]), ("RC", r["recopa"])):
            if color:
                idx[color][key].append(r["number"])
    return dict(idx)


def badge(t: dict[str, list[int]] | None) -> str:
    """':01L:00C:00RC' (vacío si la IA nunca ganó nada)."""
    if not t or not (t["L"] or t["C"] or t["RC"]):
        return ""
    return f":{len(t['L']):02d}L:{len(t['C']):02d}C:{len(t['RC']):02d}RC"


def rank_champions(idx: dict[str, dict[str, list[int]]]) -> list[tuple[str, dict]]:
    """Orden de prioridad: más Ligas, luego más Copas, luego más Recopas.
    Desempate: quien ganó su primer título antes, y por último el color (determinista)."""
    def key(item):
        color, t = item
        first = min(min(v) for v in t.values() if v)
        return (-len(t["L"]), -len(t["C"]), -len(t["RC"]), first, color)
    return sorted(idx.items(), key=key)
