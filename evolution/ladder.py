"""Propuesta 2: torneo maestro entre CAMPEONES de distintas temporadas.

Se toman los 1.º de la tabla anual de temporadas repartidas en el tiempo (sus .pth nunca se borran) y se
enfrentan todos contra todos (ida y vuelta). Si el entrenamiento progresa, los campeones nuevos le
ganan a los viejos. Mide progreso RELATIVO a la propia historia (y detecta ciclos tipo piedra-papel-tijera),
sin depender de ningún rival externo.
"""
from __future__ import annotations

import math

from . import modelstore, stats, statics
from .rng import derive_seed
from models_statics.base import StaticBot
from .repo import BEST_NEURAL_SQL
from .display import Display
from .engine import play_match


def pick_champions(conn, models_dir, step: int | None = None, max_n: int = 20) -> list[dict]:
    """1.º de la tabla anual de temporadas repartidas (siempre incluye la primera y la última).

    Una IA que fue 1.ª varias temporadas aparece una sola vez (con su primera temporada)."""
    rows = conn.execute(
        """SELECT s.number, i.color, i.model_file, i.model_sha256 FROM seasons s
           JOIN season_participants p ON p.season_id = s.id AND """ + BEST_NEURAL_SQL + """
           JOIN individuals i ON i.id = p.individual_id WHERE s.status = 'completed' ORDER BY s.number""").fetchall()
    if not rows:
        return []
    if step is None:
        step = max(1, math.ceil(len(rows) / max_n))
    chosen = rows[::step]
    if rows[-1] not in chosen:
        chosen.append(rows[-1])
    out, seen = [], set()
    for r in chosen:
        if r["color"] in seen:
            continue
        seen.add(r["color"])
        path = models_dir / r["model_file"]
        if path.exists():
            out.append({"season": r["number"], "color": r["color"], "path": path, "sha256": r["model_sha256"]})
    return out[:max_n] if len(out) > max_n else out


def run_ladder(champs: list[dict], progress=None) -> dict:
    n = len(champs)
    nets = [(statics.make_player(c["cls"], derive_seed(0, "ladder", c["color"])) if c.get("cls") else
             modelstore.load_net(c["path"], c["sha256"])) for c in champs]
    score = {}      # (i, j) -> puntos de i contra j (victoria 1, empate 0,5)
    games = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k, (first, second) in enumerate(((i, j), (j, i))):
                for idx in (first, second):
                    if isinstance(nets[idx], StaticBot):
                        nets[idx].reseed(derive_seed(0, "ladder", idx, first, second))
                r = play_match(nets[first], nets[second])
                a = 1.0 if r.result == "x" else 0.5 if r.result == "draw" else 0.0
                score[(first, second)] = score.get((first, second), 0.0) + a
                score[(second, first)] = score.get((second, first), 0.0) + (1.0 - a)
            games[(i, j)] = games[(j, i)] = 2.0
        if progress:
            progress(i + 1, n)
    elo = stats.bradley_terry(n, score, games)
    # Las redes (con temporada) se ordenan por antigüedad; los estáticos no tienen temporada y NO cuentan como
    # "generación más nueva" (si no, inflarían el veredicto): se miden aparte, contra todas las redes.
    nets_idx = [i for i, c in enumerate(champs) if c.get("season") is not None]
    vs_older, vs_newer = [float("nan")] * n, [float("nan")] * n
    for pos, i in enumerate(nets_idx):
        old = [(score[(i, j)], games[(i, j)]) for j in nets_idx[:pos]]
        new = [(score[(i, j)], games[(i, j)]) for j in nets_idx[pos + 1:]]
        vs_older[i] = sum(a for a, _ in old) / sum(g for _, g in old) if old else float("nan")
        vs_newer[i] = sum(a for a, _ in new) / sum(g for _, g in new) if new else float("nan")
    for i in (k for k in range(n) if k not in nets_idx):
        vs = [(score[(i, j)], games[(i, j)]) for j in nets_idx]
        vs_older[i] = sum(a for a, _ in vs) / sum(g for _, g in vs) if vs else float("nan")    # = contra todas las redes
    # ¿el más nuevo de cada pareja de REDES gana? (0,5 = sin progreso)
    newer_pts = sum(score[(nets_idx[b], nets_idx[a])] for a in range(len(nets_idx)) for b in range(a + 1, len(nets_idx)))
    pairs = len(nets_idx) * (len(nets_idx) - 1) / 2 * 2
    # ciclos: ternas A>B, B>C, C>A
    def beats(a, b):
        return score[(a, b)] > score[(b, a)]
    cycles = sum(1 for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)
                 if (beats(a, b) and beats(b, c) and beats(c, a)) or (beats(b, a) and beats(c, b) and beats(a, c)))
    triples = n * (n - 1) * (n - 2) // 6
    seasons = [champs[i]["season"] for i in nets_idx]
    return {"champs": champs, "elo": elo, "vs_older": vs_older, "vs_newer": vs_newer, "score": score,
            "newer_share": newer_pts / pairs if pairs else float("nan"),
            "rho": stats.spearman(seasons, [elo[i] for i in nets_idx]), "threshold": 1.96 / math.sqrt(max(len(nets_idx) - 1, 1)),
            "cycles": cycles, "triples": triples}


def render_ladder(out: Display, res: dict) -> None:
    champs = res["champs"]
    n = len(champs)
    out.banner(f"TORNEO MAESTRO · {n} campeones de distintas temporadas, todos contra todos (ida y vuelta)")
    if n < 3:
        out.emit("  Hacen falta al menos 3 campeones distintos (más temporadas completadas).")
        return
    for c in champs:
        if c.get("cls"):
            out.statics[c["color"]] = {"name": c["cls"].NAME, "color2": c["cls"].COLOR2, "slug": c.get("slug", "")}
    iw = max(9, out.id_width())
    out.emit(f"  {'Temp':>5}  {'IA':<{iw}} {'Elo':>6} {'vs anteriores':>14} {'vs posteriores':>15}")
    for i, c in enumerate(champs):
        vo = "—" if res["vs_older"][i] != res["vs_older"][i] else f"{res['vs_older'][i]:.0%}"
        vn = "—" if res["vs_newer"][i] != res["vs_newer"][i] else f"{res['vs_newer'][i]:.0%}"
        season = "—" if c.get("season") is None else c["season"]
        out.emit(f"  {season:>5}  {out.cell(c['color'], iw, False)} {res['elo'][i]:>6.0f} {vo:>14} {vn:>15}")
    out.emit(out.dim("  Elo en escala de 400 puntos, media 1000. 'vs anteriores' = puntaje de ese campeón contra los de temporadas previas "
                     "(en un estático, contra todas las redes; los estáticos no cuentan como 'generación más nueva')."))
    ns, rho, thr = res["newer_share"], res["rho"], res["threshold"]
    verdict = "MEJORA" if rho > thr and ns > 0.55 else "EMPEORA" if rho < -thr and ns < 0.45 else "SIN PROGRESO CLARO"
    out.section("Lectura")
    out.emit(f"  La generación más NUEVA de cada pareja se lleva el {ns:.0%} de los puntos (50 % = ninguna mejora).")
    out.emit(f"  Correlación temporada↔Elo: ρ={rho:+.2f} (umbral ±{thr:.2f}).")
    out.emit(f"  Ciclos (A gana a B, B a C y C a A): {res['cycles']} de {res['triples']} ternas "
             f"({res['cycles'] / res['triples']:.0%}) — muchos ciclos = la población 'da vueltas' sin mejorar.")
    out.emit(out.bold(f"  Veredicto: {verdict}"))