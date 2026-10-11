"""Pruebas sueltas con las IA (no se guardan en la base de datos):

* ``fight``: dos IA se enfrentan en una partida de ida y vuelta.
* ``run_random_league``: una Liga de 20 con las IA indicadas y el resto rellenado con redes aleatorias.

Las IA se indican con un número de temporada (el 1.º de su tabla anual) o con su color (#rrggbb).
Siempre juegan como en los torneos: mejor movimiento legal de su red, sin MCTS.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from model import SuperTatetiNet

from . import colors as colormod
from . import modelstore, reports, rules, schedule
from . import titles as titlemod
from .display import Display
from .engine import play_match
from .play import PlayError, find_opponent
from .rng import derive_rng, derive_seed

_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")


# ---------------------------------------------------------------------------
# Referencias a IA
# ---------------------------------------------------------------------------
def parse_ia_ref(token: str) -> tuple[str, object]:
    """'12' -> ('season', 12) · '#a1b2c3' o 'a1b2c3' -> ('color', '#a1b2c3').

    Un token de exactamente 6 caracteres hexadecimales se toma como color (la shell trata '#...' como
    comentario, por eso se admite sin '#'); los números más cortos son temporadas."""
    t = token.strip()
    if t.startswith("#") or (len(t) == 6 and _HEX.match(t)):
        if not _HEX.match(t):
            raise PlayError(f"'{token}' no es un color válido (formato #rrggbb).")
        return "color", "#" + t.lstrip("#").lower()
    if t.isdigit() and int(t) > 0:
        return "season", int(t)
    raise PlayError(f"'{token}' no es un número de temporada ni un color (#rrggbb).")


def resolve_ia(conn, models_dir: Path, token: str) -> dict:
    """Devuelve {label, color, tag, path, sha256} de una IA que todavía tiene su modelo guardado."""
    kind, ref = parse_ia_ref(token)
    if kind == "season":
        o = find_opponent(conn, models_dir, ref)
        return {"label": f"mejor de T{ref}", "color": o["color"], "tag": o["tag"], "path": o["path"], "sha256": o["sha256"]}
    row = conn.execute("SELECT color, model_file, model_sha256, origin_season FROM individuals WHERE color=%s", (ref,)).fetchone()
    if row is None:
        raise PlayError(f"No existe ninguna IA con el color {ref}.")
    if row["model_sha256"] is None:
        raise PlayError(f"La IA {ref} todavía no tiene su modelo escrito en disco.")
    path = models_dir / row["model_file"]
    if not path.exists():
        st = reports.individual_status(conn).get(ref)
        why = f" (expulsada en T{st['last_season']}; el .pth se borró por antigüedad)" if st and st["expelled"] else ""
        raise PlayError(f"La IA {ref} ya no tiene su modelo guardado{why}.")
    badge = titlemod.badge(reports.titles_through(conn).get(ref))
    return {"label": ref, "color": ref, "tag": ref + badge, "path": path, "sha256": row["model_sha256"]}


def resolve_many(conn, models_dir: Path, tokens: list[str]) -> list[dict]:
    """Resuelve varias referencias rechazando IA repetidas (p. ej. una temporada y su color)."""
    out, seen = [], {}
    for tok in tokens:
        ia = resolve_ia(conn, models_dir, tok)
        if ia["color"] in seen:
            raise PlayError(f"'{tok}' y '{seen[ia['color']]}' son la misma IA ({ia['color']}).")
        seen[ia["color"]] = tok
        out.append(ia)
    return out


def _net(ia: dict):
    return modelstore.load_net(ia["path"], ia["sha256"])


# ---------------------------------------------------------------------------
# Pelea: una partida de ida y vuelta
# ---------------------------------------------------------------------------
def fight(a: dict, b: dict) -> dict:
    """Dos partidas: en la 1.ª comienza ``a`` y en la 2.ª comienza ``b``. Gana quien sume más puntos (3/1/0)."""
    na, nb = _net(a), _net(b)
    games = []
    for first, second, nf, ns in ((a, b, na, nb), (b, a, nb, na)):
        out = play_match(nf, ns)
        winner = first["color"] if out.result == "x" else second["color"] if out.result == "o" else None
        games.append({"x": first["color"], "o": second["color"], "result": out.result, "winner": winner,
                      "moves": out.num_moves, "ms": out.duration_ms})
    pts = {a["color"]: 0, b["color"]: 0}
    for g in games:
        pts[g["x"]] += rules.match_points(g["result"], "x")
        pts[g["o"]] += rules.match_points(g["result"], "o")
    winner = a["color"] if pts[a["color"]] > pts[b["color"]] else b["color"] if pts[b["color"]] > pts[a["color"]] else None
    return {"games": games, "points": pts, "winner": winner}


def render_fight(out: Display, a: dict, b: dict, res: dict) -> None:
    out.banner(f"PELEA · {out.ia(a['color'])} ({a['label']})  vs  {out.ia(b['color'])} ({b['label']})")
    for i, g in enumerate(res["games"], start=1):
        who = f"gana {out.ia(g['winner'])}" if g["winner"] else "empate"
        out.emit(f"  Partida {i}: comienza {out.ia(g['x'])} vs {out.ia(g['o'])} -> {who}  [{g['moves']} jug., {g['ms']:.0f}ms]")
    pa, pb = res["points"][a["color"]], res["points"][b["color"]]
    out.emit(f"  Puntos (3 victoria · 1 empate): {out.ia(a['color'])} {pa} · {out.ia(b['color'])} {pb}")
    out.emit(out.bold("  Resultado: " + (f"gana {out.ia(res['winner'])}" if res["winner"] else "empate")))
    out.emit(out.dim("  Las IA juegan de forma determinista: repetir la pelea da el mismo resultado."))


# ---------------------------------------------------------------------------
# Liga de prueba con IA aleatorias
# ---------------------------------------------------------------------------
def run_random_league(tested: list[dict], seed: int, progress=None) -> dict:
    """Liga completa (20 equipos, ida y vuelta) en memoria. ``tested`` + redes aleatorias hasta 20."""
    n = rules.POPULATION
    if not 1 <= len(tested) <= n:
        raise PlayError(f"Indicá entre 1 y {n} IA (recibido {len(tested)}).")
    nets = {t["color"]: _net(t) for t in tested}
    taken = set(nets)
    fill = n - len(tested)
    random_cols = [r.color for r in colormod.spread_colors(fill, derive_rng(seed, "rndcolors"), taken)] if fill else []
    for i, c in enumerate(random_cols):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed(seed, "rnd", i))
            net = SuperTatetiNet().eval()
        for p in net.parameters():
            p.requires_grad_(False)
        nets[c] = net
    members = [t["color"] for t in tested] + random_cols
    slot_order = rules.shuffled_canonical(members, derive_rng(seed, "slots"))
    table = {c: rules.Row(c) for c in members}
    history = {c: [] for c in members}
    games = []
    for md, pairs in enumerate(schedule.league_fixtures(n), start=1):
        today = []
        for xs, os_ in pairs:
            x, o = slot_order[xs - 1], slot_order[os_ - 1]
            out = play_match(nets[x], nets[o])
            table[x].add(out.result, "x")
            table[o].add(out.result, "o")
            g = {"md": md, "x": x, "o": o, "result": out.result, "moves": out.num_moves}
            games.append(g); today.append(g)
        for c in members:
            history[c].append(table[c].points)
        if progress:
            progress(md, today)
    draws: list[rules.DrawRecord] = []
    order = rules.league_final_order(members, {c: table[c].points for c in members}, history,
                                     lambda ctx: derive_rng(seed, "draw", ctx), draws)
    tested_set = set(nets) - set(random_cols)
    summary = {}
    for t in tested:
        c = t["color"]
        vs_rnd, vs_tested = [0, 0, 0], [0, 0, 0]
        for g in games:
            if c not in (g["x"], g["o"]):
                continue
            other = g["o"] if g["x"] == c else g["x"]
            side = "x" if g["x"] == c else "o"
            idx = 0 if g["result"] == side else 1 if g["result"] == "draw" else 2
            (vs_tested if other in tested_set else vs_rnd)[idx] += 1
        summary[c] = {"position": order.index(c) + 1, "row": table[c], "vs_random": vs_rnd, "vs_tested": vs_tested}
    return {"seed": seed, "order": order, "table": table, "tested": [t["color"] for t in tested],
            "random": random_cols, "summary": summary, "games": games, "draws": [d.context for d in draws]}


def render_random_league(out: Display, res: dict, tested: list[dict]) -> None:
    n = len(res["order"])
    out.banner(f"LIGA DE PRUEBA (no se guarda) · {len(res['tested'])} IA probada(s) + {len(res['random'])} aleatoria(s)")
    out.emit(f"  Semilla: {res['seed']}  (repetí con seed={res['seed']} para obtener lo mismo)")
    w = max(out.id_width(), 7) + (2 if out.color else 0)
    out.section("Tabla final")
    out.emit(f"  {'Pos':>3}  {'IA':<{w}} {'PJ':>3} {'G':>3} {'E':>3} {'P':>3} {'Pts':>4}")
    for pos, c in enumerate(res["order"], start=1):
        r = res["table"][c]
        if c in res["tested"]:
            mark = out.bold("  ◄ probada")
        else:
            mark = out.dim("  random")
        out.emit(f"  {pos:>3}  {out.cell(c, w)} {r.played:>3} {r.wins:>3} {r.draws:>3} {r.losses:>3} {r.points:>4}{mark}")
    out.section("Resumen de las IA probadas")
    labels = {t["color"]: t["label"] for t in tested}
    for c in res["tested"]:
        s = res["summary"][c]
        g, e, p = s["vs_random"]
        line = (f"  {out.ia(c)} ({labels[c]}): puesto {s['position']}/{n} · {s['row'].points} pts · "
                f"{s['row'].wins}G {s['row'].draws}E {s['row'].losses}P · vs random: {g}G {e}E {p}P")
        if len(res["tested"]) > 1:
            g2, e2, p2 = s["vs_tested"]
            line += f" · vs otras probadas: {g2}G {e2}E {p2}P"
        out.emit(line)