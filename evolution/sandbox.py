"""Pruebas sueltas con las IA (no se guardan en la base de datos):

* ``fight``: dos IA se enfrentan en una partida de ida y vuelta.
* ``fight_round_robin``: 3 a 8 IA, todas contra todas ida y vuelta.
* ``run_random_league``: una Liga de 20 con las IA indicadas y el resto rellenado con redes aleatorias.

Las IA se indican con un número de temporada (la mejor RED de su tabla anual), con su color (#rrggbb) o, si son
bots estáticos, SIEMPRE por su nombre. Las redes juegan como en los torneos: mejor movimiento legal, sin MCTS.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from model import SuperTatetiNet

from . import colors as colormod
from . import modelstore, reports, rules, schedule, statics
from . import titles as titlemod
from .display import Display
from .engine import play_match
from .play import PlayError, find_opponent
from .rng import derive_rng, derive_seed
from models_statics.base import StaticBot

_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")


# ---------------------------------------------------------------------------
# Referencias a IA
# ---------------------------------------------------------------------------
_NAME = re.compile(r"^(?=.*[A-Za-z])[\w\-.]+$")


def parse_ia_ref(token: str) -> tuple[str, object]:
    """'12' -> ('season', 12) · '#a1b2c3' o 'a1b2c3' -> ('color', '#a1b2c3') · 'BocaJrs' -> ('name', 'BocaJrs').

    Un número de temporada resuelve SIEMPRE a la mejor RED de su tabla anual; un bot estático solo se invoca
    escribiendo su NOMBRE. Un token de exactamente 6 caracteres hexadecimales se toma como color (la shell trata
    '#...' como comentario, por eso se admite sin '#'); si no existe, se prueba como nombre."""
    t = token.strip()
    if t.startswith("#") or (len(t) == 6 and _HEX.match(t)):
        if not _HEX.match(t):
            raise PlayError(f"'{token}' no es un color válido (formato #rrggbb).")
        return "color", "#" + t.lstrip("#").lower()
    if t.isdigit() and int(t) > 0:
        return "season", int(t)
    if _NAME.match(t):
        return "name", t
    raise PlayError(f"'{token}' no es un número de temporada, un color (#rrggbb) ni un nombre de bot estático.")


def _static_ia(conn, statics_dir: Path | None, slug: str, cls) -> dict:
    """IA estática a partir de su código (la lógica vive en models_statics/); la identidad sale de la BD si existe."""
    row = conn.execute("SELECT ia_type FROM individuals WHERE color=%s", (cls.COLOR1,)).fetchone()
    if row is not None and row["ia_type"] != "estatico":
        raise PlayError(f"El COLOR1 {cls.COLOR1} de '{cls.NAME}' ya es la identidad de una red neuronal.")
    badge = titlemod.badge(reports.titles_through(conn).get(cls.COLOR1))
    return {"type": "estatico", "label": f"estático {cls.NAME}", "color": cls.COLOR1, "name": cls.NAME, "color2": cls.COLOR2,
            "slug": slug, "cls": cls, "tag": cls.NAME + badge, "path": None, "sha256": None}


def resolve_ia(conn, models_dir: Path, token: str, statics_dir: Path | None = None) -> dict:
    """Devuelve la IA indicada: red {type, label, color, tag, path, sha256} o estático {type, name, color2, cls, ...}.
    Las redes deben tener su modelo guardado todavía."""
    kind, ref = parse_ia_ref(token)
    if kind == "season":
        o = find_opponent(conn, models_dir, ref)
        return {"type": "neuronal", "label": f"mejor de T{ref}", "color": o["color"], "tag": o["tag"], "path": o["path"],
                "sha256": o["sha256"]}
    if kind == "name" or (kind == "color" and conn.execute("SELECT 1 FROM individuals WHERE color=%s", (ref,)).fetchone() is None):
        if statics_dir is not None:
            try:
                bots = statics.available(statics_dir)
                hits = [(sl, c) for sl, c in bots.items()
                        if (kind == "name" and statics.normalize(ref) in (statics.normalize(c.NAME), statics.normalize(sl)))
                        or (kind == "color" and (c.COLOR1 == ref or statics.normalize(ref) in (statics.normalize(c.NAME), statics.normalize(sl))))]
            except Exception as e:  # noqa: BLE001  (archivo de bot con errores)
                raise PlayError(f"No se pudieron cargar los bots estáticos de {statics_dir}: {e}") from e
            if len(hits) == 1:
                return _static_ia(conn, statics_dir, *hits[0])
            if len(hits) > 1:
                raise PlayError(f"'{token}' coincide con varios bots estáticos: {', '.join(c.NAME for _, c in hits)}.")
        if kind == "name":
            names = ", ".join(c.NAME for c in statics.available(statics_dir).values()) if statics_dir is not None else "ninguno"
            raise PlayError(f"No existe ninguna IA estática llamada '{ref}' (disponibles: {names or 'ninguno'}). "
                            f"Un número de temporada elige la mejor RED; un estático solo se invoca por su nombre.")
        raise PlayError(f"No existe ninguna IA con el color {ref}.")
    row = conn.execute("SELECT color, model_file, model_sha256, origin_season, ia_type, static_slug FROM individuals WHERE color=%s", (ref,)).fetchone()
    if row["ia_type"] == "estatico":      # el color de un estático también lo identifica
        cls = statics.available(statics_dir).get(row["static_slug"]) if statics_dir is not None else None
        if cls is None:
            raise PlayError(f"El bot estático {ref} ya no está en {statics_dir}.")
        return _static_ia(conn, statics_dir, row["static_slug"], cls)
    if row["model_sha256"] is None:
        raise PlayError(f"La IA {ref} todavía no tiene su modelo escrito en disco.")
    path = models_dir / row["model_file"]
    if not path.exists():
        st = reports.individual_status(conn).get(ref)
        why = f" (expulsada en T{st['last_season']}; el .pth se borró por antigüedad)" if st and st["expelled"] else ""
        raise PlayError(f"La IA {ref} ya no tiene su modelo guardado{why}.")
    badge = titlemod.badge(reports.titles_through(conn).get(ref))
    return {"type": "neuronal", "label": ref, "color": ref, "tag": ref + badge, "path": path, "sha256": row["model_sha256"]}


def resolve_many(conn, models_dir: Path, tokens: list[str], statics_dir: Path | None = None) -> list[dict]:
    """Resuelve varias referencias rechazando IA repetidas (p. ej. una temporada y su color)."""
    out, seen = [], {}
    for tok in tokens:
        ia = resolve_ia(conn, models_dir, tok, statics_dir)
        if ia["color"] in seen:
            raise PlayError(f"'{tok}' y '{seen[ia['color']]}' son la misma IA ({ia['name'] if ia.get('name') else ia['color']}).")
        seen[ia["color"]] = tok
        out.append(ia)
    return out


def show(out: Display, ias: list[dict]) -> None:
    """Registra en ``out`` los estáticos indicados (aunque no estén en la BD) para mostrarlos por nombre."""
    for ia in ias:
        if ia.get("type") == "estatico":
            out.statics[ia["color"]] = {"name": ia["name"], "color2": ia["color2"], "slug": ia["slug"]}


def make_player(ia: dict, seed: int = 0):
    """Red cargada de disco o bot estático (con su azar sembrado)."""
    if ia.get("type") == "estatico":
        return statics.make_player(ia["cls"], seed)
    return modelstore.load_net(ia["path"], ia["sha256"])


def _reseed(player, seed: int) -> None:
    if isinstance(player, StaticBot):
        player.reseed(seed)


# ---------------------------------------------------------------------------
# Pelea: una partida de ida y vuelta
# ---------------------------------------------------------------------------
def fight(a: dict, b: dict, seed: int = 0) -> dict:
    """Dos partidas: en la 1.ª comienza ``a`` y en la 2.ª comienza ``b``. Gana quien sume más puntos (3/1/0).
    El azar de un estático se siembra con ``seed`` (misma semilla = misma pelea)."""
    pa, pb = make_player(a, derive_seed(seed, "fight", 0)), make_player(b, derive_seed(seed, "fight", 1))
    games = []
    for k, (first, second, pf, ps) in enumerate(((a, b, pa, pb), (b, a, pb, pa))):
        _reseed(pa, derive_seed(seed, "fight", "a", k)); _reseed(pb, derive_seed(seed, "fight", "b", k))
        out = play_match(pf, ps)
        winner = first["color"] if out.result == "x" else second["color"] if out.result == "o" else None
        games.append({"x": first["color"], "o": second["color"], "result": out.result, "winner": winner,
                      "moves": out.num_moves, "ms": out.duration_ms})
    pts = {a["color"]: 0, b["color"]: 0}
    for g in games:
        pts[g["x"]] += rules.match_points(g["result"], "x")
        pts[g["o"]] += rules.match_points(g["result"], "o")
    winner = a["color"] if pts[a["color"]] > pts[b["color"]] else b["color"] if pts[b["color"]] > pts[a["color"]] else None
    return {"games": games, "points": pts, "winner": winner}


def fight_round_robin(ias: list[dict], seed: int = 0) -> dict:
    """Mini liga de 3 a 8 IA: cada pareja juega ida y vuelta. Devuelve la tabla (3/1/0) y las peleas."""
    if not 3 <= len(ias) <= 8:
        raise PlayError("Una mini liga necesita entre 3 y 8 IA (para dos IA usá una pelea).")
    pts = {ia["color"]: 0 for ia in ias}
    wins = {ia["color"]: [0, 0, 0] for ia in ias}   # G E P
    fights = []
    for i in range(len(ias)):
        for j in range(i + 1, len(ias)):
            res = fight(ias[i], ias[j], derive_seed(seed, "rr", i, j))
            fights.append((ias[i], ias[j], res))
            for g in res["games"]:
                for side, col in (("x", g["x"]), ("o", g["o"])):
                    pts[col] += rules.match_points(g["result"], side)
                    wins[col][0 if g["result"] == side else 1 if g["result"] == "draw" else 2] += 1
    order = sorted(ias, key=lambda ia: (-pts[ia["color"]], ia["color"]))
    return {"points": pts, "record": wins, "order": order, "fights": fights}


def render_fight_table(out: Display, ias: list[dict], res: dict, seed: int) -> None:
    show(out, ias)
    out.banner(f"MINI LIGA · {len(ias)} IA, todas contra todas (ida y vuelta)")
    w = max(out.id_width(), 7) + (2 if out.color else 0)
    out.emit(f"  {'Pos':>3}  {'IA':<{w}} {'G':>3} {'E':>3} {'P':>3} {'Pts':>4}")
    for pos, ia in enumerate(res["order"], start=1):
        g, e, p = res["record"][ia["color"]]
        out.emit(f"  {pos:>3}  {out.cell(ia['color'], w)} {g:>3} {e:>3} {p:>3} {res['points'][ia['color']]:>4}")
    out.section("Enfrentamientos")
    for a, b, r in res["fights"]:
        winner = f"gana {out.ia(r['winner'])}" if r["winner"] else "empate"
        out.emit(f"  {out.ia(a['color'])} vs {out.ia(b['color'])}: "
                 f"{r['points'][a['color']]}-{r['points'][b['color']]} -> {winner}")
    if any(ia.get("type") == "estatico" for ia in ias):
        out.emit(out.dim(f"  Hay bots estáticos: su azar sale de la semilla {seed} (misma semilla = mismos resultados)."))


def render_fight(out: Display, a: dict, b: dict, res: dict) -> None:
    show(out, [a, b])
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
    nets = {t["color"]: make_player(t, derive_seed(seed, "static", t["color"])) for t in tested}   # redes y/o estáticos
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
            _reseed(nets[x], derive_seed(seed, "game", md, xs, "x")); _reseed(nets[o], derive_seed(seed, "game", md, xs, "o"))
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
    show(out, tested)
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