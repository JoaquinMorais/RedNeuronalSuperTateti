"""Consultas de solo lectura y vistas para analizar temporadas desde la terminal.

No toman el bloqueo de entrenamiento: se pueden usar mientras se entrena.
"""
from __future__ import annotations

import re
from collections import Counter

from . import rules
from . import titles as titlemod
from .display import ROUND_NAME, Display

_TAG = re.compile(r"(#[0-9a-f]{6})(?::\d\dL:\d\dC:\d\dRC)?")


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------
def champion_rows(conn, upto: int | None = None) -> list[dict]:
    """Campeones de las temporadas COMPLETADAS (hasta ``upto`` inclusive)."""
    return conn.execute(
        """SELECT s.number, l.color AS league, c.color AS cup, u.color AS runner_up, r.color AS recopa
           FROM seasons s
           LEFT JOIN individuals l ON l.id = s.league_champion_id
           LEFT JOIN individuals c ON c.id = s.cup_champion_id
           LEFT JOIN individuals u ON u.id = s.cup_runner_up_id
           LEFT JOIN individuals r ON r.id = s.recopa_champion_id
           WHERE s.status = 'completed' AND (%s::int IS NULL OR s.number <= %s::int)
           ORDER BY s.number""", (upto, upto)).fetchall()


def individual_status(conn) -> dict[str, dict]:
    """{color: {"last_season": L, "expelled": bool}} para toda IA que haya competido.

    ``expelled`` = su última temporada terminó y no pasó a la siguiente generación
    (``L`` es la temporada en la que fue expulsada). Si no, sigue compitiendo."""
    rows = conn.execute(
        """WITH last AS (SELECT p.individual_id, max(s.number) AS n FROM season_participants p
                         JOIN seasons s ON s.id = p.season_id GROUP BY p.individual_id)
           SELECT i.color, l.n AS last_season,
                  (s.status = 'completed' AND NOT EXISTS (
                       SELECT 1 FROM evolution_plan e WHERE e.season_id = s.id AND e.individual_id = i.id)) AS expelled
           FROM last l JOIN seasons s ON s.number = l.n JOIN individuals i ON i.id = l.individual_id""").fetchall()
    return {r["color"]: {"last_season": r["last_season"], "expelled": r["expelled"]} for r in rows}


def load_statics(conn) -> dict[str, dict]:
    """{color1: {"name", "color2", "slug"}} de todos los bots estáticos registrados (para mostrarlos por nombre)."""
    try:
        rows = conn.execute("SELECT color, static_name, static_color2, static_slug FROM individuals WHERE ia_type='estatico'").fetchall()
    except Exception:  # BD anterior a los estáticos (sin la columna): no hay ninguno
        conn.rollback()
        return {}
    return {r["color"]: {"name": r["static_name"], "color2": r["static_color2"], "slug": r["static_slug"]} for r in rows}


def ensure_statics(conn, out) -> None:
    """Cada vista se basta sola: registra en ``out`` los nombres de los estáticos para mostrarlos por nombre."""
    out.statics = {**load_statics(conn), **out.statics}


def type_label(out, color: str) -> str:
    return "estático" if color in out.statics else "red"


def titles_through(conn, upto: int | None = None) -> dict:
    return titlemod.build_title_index(champion_rows(conn, upto))


# ---------------------------------------------------------------------------
# Estado de las temporadas (+ campeones)
# ---------------------------------------------------------------------------
def _md_total(season_row) -> int:
    """Fechas de Liga de la temporada (38 en las de 20 equipos; BD antiguas sin la columna = 38)."""
    return season_row.get("league_matchdays") or 38


def render_status(conn, out: Display) -> None:
    ensure_statics(conn, out)
    champs = {r["number"]: r for r in champion_rows(conn)}
    seasons = conn.execute("SELECT * FROM seasons ORDER BY number").fetchall()
    if not seasons:
        out.emit("Todavía no hay temporadas.")
    for s in seasons:
        line = f"Temporada {s['number']}: {s['status']} · etapa {s['stage']} · fecha cerrada {s['current_matchday']}/{_md_total(s)}"
        if (s.get("n_participants") or 20) != 20:
            line += f" · {s['n_participants']} equipos"
        c = champs.get(s["number"])
        if c:
            line += (f" · Liga: {out.ia(c['league'], badge=False)} · Copa: {out.ia(c['cup'], badge=False)}"
                     f" · Recopa: {out.ia(c['recopa'], badge=False)}")
        out.emit(line)
    best = conn.execute("""SELECT s.number, i.color, h.total_score, h.model_file FROM historical_best h
                           JOIN seasons s ON s.id=h.season_id JOIN individuals i ON i.id=h.individual_id
                           ORDER BY h.total_score DESC, s.number LIMIT 1""").fetchone()
    if best:
        out.emit(f"Mejor histórico: {out.ia(best['color'], badge=False)} (temporada {best['number']}, "
                 f"{best['total_score']} pts) -> {best['model_file']}")


# ---------------------------------------------------------------------------
# Campeones
# ---------------------------------------------------------------------------
def _seasons_txt(nums: list[int]) -> str:
    return ", ".join(f"T{n}" for n in nums)


def render_champions(conn, out: Display, top: int = 30) -> None:
    """Campeones por temporada (las ``top`` más recientes) y ranking de campeones (los ``top`` mejores)."""
    ensure_statics(conn, out)
    rows = champion_rows(conn)
    out.banner("CAMPEONES")
    if not rows:
        out.emit("  Todavía no terminó ninguna temporada.")
        return
    shown = rows[-top:]
    out.section("Campeones de cada temporada" + (f" (las últimas {top} de {len(rows)})" if len(rows) > top else ""))
    cw = max(10, max((len(out.tag(c, False)) for r in shown for c in (r["league"], r["cup"], r["runner_up"], r["recopa"])), default=10))
    out.emit(f"  {'Temp':>4}  {'Liga':<{cw}} {'Copa':<{cw}} {'Subcampeón':<{cw}} {'Recopa':<{cw}}")
    for r in shown:
        out.emit(f"  {r['number']:>4}  {out.cell(r['league'], cw, False)} {out.cell(r['cup'], cw, False)} "
                 f"{out.cell(r['runner_up'], cw, False)} {out.cell(r['recopa'], cw, False)}")
    status = individual_status(conn)
    ranking = titlemod.rank_champions(titlemod.build_title_index(rows))
    out.section("Ranking de campeones (prioridad: Liga, Copa, Recopa)")
    nw = max(10, out.id_width() - max((len(titlemod.badge(t)) for t in out.titles.values()), default=0))
    out.emit(f"  {'Pos':>3}  {'IA':<{nw}} {'Tipo':<8} {'Liga':>4} {'Copa':>4} {'Recopa':>6}  {'Estado':<15} Títulos")
    for pos, (color, t) in enumerate(ranking[:top], start=1):
        st = status.get(color)
        estado = "compitiendo" if st is None or not st["expelled"] else f"expulsada T{st['last_season']}"
        det = " · ".join(f"{name}: {_seasons_txt(t[k])}" for k, name in (("L", "Ligas"), ("C", "Copas"), ("RC", "Recopas")) if t[k])
        out.emit(f"  {pos:>3}  {out.cell(color, nw, False)} {type_label(out, color):<8} {len(t['L']):>4} {len(t['C']):>4} {len(t['RC']):>6}  {estado:<15} {det}")
    if len(ranking) > top:
        out.emit(out.dim(f"  Mostrando {top} de {len(ranking)} campeones. Para ver más: `champions N` (en el menú: `3 N`)."))


# ---------------------------------------------------------------------------
# Bracket de Copa (texto)
# ---------------------------------------------------------------------------
def render_bracket(ties: dict, out: Display) -> list[str]:
    """``ties[(ronda, nº)] = {a, b, slot_a, slot_b, winner, third}``. Devuelve líneas de texto.

    Cada IA se muestra con su insignia de títulos (``out.titles``). Se compone en texto
    plano (para alinear) con columnas del ancho necesario y recién después se pintan los colores.
    """
    tag = lambda c: out.tag(c) if c else "?"  # noqa: E731
    seen = [c for t in ties.values() for c in (t["a"], t["b"], t["winner"]) if c]
    L = max([len(out.tag(c)) for c in seen] + [7]) + 1          # +1: marca '*' de desempate
    lw = [3 + L, L, L, L]                                        # ancho de etiqueta por columna (la 1.ª lleva '1A ')
    xs = [0]
    for w in lw:
        xs.append(xs[-1] + w + 4)                                # etiqueta + relleno + conector + guion
    rows = 31
    width = xs[4] + L + 10
    grid = [[" "] * width for _ in range(rows)]

    def put(r: int, x: int, text: str) -> None:
        for i, ch in enumerate(text):
            grid[r][x + i] = ch

    def entry(r: int, col: int, text: str) -> None:
        put(r, xs[col], text)
        for i in range(xs[col] + len(text), xs[col] + lw[col] + 1):
            grid[r][i] = "─"

    def connect(col: int, r1: int, r2: int) -> int:
        cx, rw = xs[col] + lw[col] + 1, (r1 + r2) // 2
        grid[r1][cx], grid[r2][cx] = "┐", "┘"
        for r in range(r1 + 1, r2):
            grid[r][cx] = "│"
        grid[rw][cx] = "├"
        put(rw, cx + 1, "──")
        return rw

    mark = lambda t: "*" if t and t.get("third") else ""  # noqa: E731
    winner_row = {}
    for i in range(8):                                           # octavos
        t = ties[("R16", i + 1)]
        r1, r2 = 4 * i, 4 * i + 2
        entry(r1, 0, f"{t['slot_a']} {tag(t['a'])}")
        entry(r2, 0, f"{t['slot_b']} {tag(t['b'])}")
        winner_row[("R16", i + 1)] = connect(0, r1, r2)
    for col, rnd, count in ((1, "QF", 4), (2, "SF", 2), (3, "F", 1)):
        prev = {"QF": "R16", "SF": "QF", "F": "SF"}[rnd]
        for k in range(1, count + 1):
            ra, rb = winner_row[(prev, 2 * k - 1)], winner_row[(prev, 2 * k)]
            for r, n in ((ra, 2 * k - 1), (rb, 2 * k)):
                pt = ties[(prev, n)]
                entry(r, col, tag(pt["winner"]) + mark(pt))
            winner_row[(rnd, k)] = connect(col, ra, rb)
    final = ties[("F", 1)]
    put(winner_row[("F", 1)], xs[4],
        tag(final["winner"]) + mark(final) + (" CAMPEÓN" if final["winner"] else ""))

    head = [" "] * width
    for col, name in enumerate(["Octavos", "Cuartos", "Semifinales", "Final", "Campeón"]):
        for i, ch in enumerate(name):
            head[xs[col] + i] = ch
    lines = ["".join(head).rstrip()] + ["".join(r).rstrip() for r in grid]
    lines = [ln for ln in lines if ln.strip()]
    # Pintado: '#aabbcc' o '#aabbcc:01L:00C:00RC' completo con el color de la IA
    return [out.paint_tags(ln) for ln in lines]


# ---------------------------------------------------------------------------
# Vista de una temporada
# ---------------------------------------------------------------------------
def render_season(conn, out: Display, number: int) -> bool:
    s = conn.execute("SELECT * FROM seasons WHERE number=%s", (number,)).fetchone()
    if s is None:
        out.emit(f"No existe la temporada {number}.")
        return False
    out.titles = titles_through(conn, number)  # insignias "al terminar" esa temporada
    out.statics = load_statics(conn)           # los estáticos se muestran por su nombre (en dos colores)
    color = {r["id"]: r["color"] for r in conn.execute("SELECT id, color FROM individuals")}
    parts = conn.execute("SELECT * FROM season_participants WHERE season_id=%s", (s["id"],)).fetchall()
    sid = s["id"]
    out.banner(f"TEMPORADA {number} · {s['status']} · etapa {s['stage']} · fecha cerrada {s['current_matchday']}/{_md_total(s)}")
    total_md = _md_total(s)
    if len(parts) != 20:
        sizes = (s["draw_meta"] or {}).get("format", {}).get("group_sizes")
        out.emit(f"  {len(parts)} equipos (IA añadidas): Liga de {total_md} fechas y {2 * (len(parts) - 1)} partidas por IA"
                 + (f" · grupos de Copa de {'/'.join(map(str, sizes))}" if sizes else ""))

    # --- Liga ---------------------------------------------------------
    last_md = conn.execute("SELECT max(matchday) m FROM league_history WHERE season_id=%s", (sid,)).fetchone()["m"]
    w = out.id_width() + (2 if out.color else 0)
    if last_md is None:
        out.section("Liga")
        out.emit("  Aún sin partidas jugadas.")
    else:
        hist = {r["individual_id"]: r for r in conn.execute(
            "SELECT * FROM league_history WHERE season_id=%s AND matchday=%s", (sid, last_md))}
        final = s["current_matchday"] >= total_md and all(p["league_position"] for p in parts)
        order = (sorted(parts, key=lambda p: p["league_position"]) if final
                 else sorted(parts, key=lambda p: hist[p["individual_id"]]["position"]))
        out.section("Tabla de Liga" + ("" if final else f" (provisional, tras la fecha {last_md})"))
        out.emit(f"  {'Pos':>3}  {'IA':<{w}} {'PJ':>3} {'G':>3} {'E':>3} {'P':>3} {'Pts':>4}" + ("  Pts evol." if final else ""))
        for pos, p in enumerate(order, start=1):
            h = hist[p["individual_id"]]
            extra = f"  {p['league_score']:>+9d}" if final else ""
            out.emit(f"  {pos:>3}  {out.cell(color[p['individual_id']], w)} {h['played']:>3} {h['wins']:>3} "
                     f"{h['draws']:>3} {h['losses']:>3} {h['points']:>4}{extra}")

    # --- Copa ---------------------------------------------------------
    out.section("Copa")
    groups = conn.execute("SELECT * FROM cup_group_table WHERE season_id=%s ORDER BY group_name, position", (sid,)).fetchall()
    if not groups:
        out.emit("  Fase de grupos en curso.")
    for g in "ABCD":
        rows = [r for r in groups if r["group_name"] == g]
        if rows:
            qual = "  ".join(f"{r['position']}.{out.ia(color[r['individual_id']])}({r['points']})"
                             for r in rows if r["qualified"])
            elim = " ".join(out.ia(color[r["individual_id"]]) for r in rows if not r["qualified"])
            out.emit(f"  Grupo {g}: {qual}   fuera: {elim}")
    ties = {}
    for t in conn.execute("SELECT * FROM cup_ties WHERE season_id=%s", (sid,)).fetchall():
        third = conn.execute("SELECT 1 FROM matches WHERE tie_id=%s AND seq=3", (t["id"],)).fetchone() is not None
        ties[(t["round"], t["tie_no"])] = {
            "slot_a": t["slot_a"], "slot_b": t["slot_b"], "third": third,
            "a": color.get(t["individual_a_id"]), "b": color.get(t["individual_b_id"]),
            "winner": color.get(t["winner_id"])}
    if groups:
        out.emit("")
        for ln in render_bracket(ties, out):
            out.emit("  " + ln)
        out.emit(out.dim("  (* = se resolvió en una 3.ª partida de desempate)"))
    if s["cup_champion_id"]:
        out.emit(f"  Campeón: {out.ia(color[s['cup_champion_id']])}   Subcampeón: {out.ia(color[s['cup_runner_up_id']])}")

    # --- Recopa -------------------------------------------------------
    out.section("Recopa")
    rec = conn.execute("SELECT * FROM recopa_results WHERE season_id=%s", (sid,)).fetchone()
    if rec is None:
        out.emit(f"  Pendiente (se juega tras la fecha {total_md}).")
    elif rec["same_champion"]:
        out.emit(f"  Campeón: {out.ia(color[rec['champion_id']])} (ganó Liga y Copa: campeón sin jugar)")
    elif rec["champion_id"] is None:
        out.emit(f"  En juego: Copa {out.ia(color[rec['cup_champion_id']])} vs Liga {out.ia(color[rec['league_champion_id']])}")
    else:
        how = "sorteo reproducible" if rec["resolution"] == "random_draw" else "más victorias"
        out.emit(f"  Copa {out.ia(color[rec['cup_champion_id']])}: {rec['cup_champ_wins']} V · "
                 f"Liga {out.ia(color[rec['league_champion_id']])}: {rec['league_champ_wins']} V · {rec['draws']} empates")
        out.emit(f"  Campeón: {out.ia(color[rec['champion_id']])} ({how})")

    # --- Tabla anual --------------------------------------------------
    out.section("Tabla anual (Copa + Liga + Recopa)")
    if any(p["final_position"] is None for p in parts):
        out.emit("  Disponible cuando terminen las tres competiciones.")
        return True
    plan = conn.execute("SELECT * FROM evolution_plan WHERE season_id=%s", (sid,)).fetchall()
    kids = Counter(p["parent_id"] for p in plan if p["role"] == "child")
    survivors = {p["individual_id"] for p in plan if p["role"] == "survivor"}
    stay = {p["individual_id"] for p in plan if p["role"] == "static"}
    out.emit(f"  {'Pos':>3}  {'IA':<{w}} {'Tipo':<8} {'Liga':>5} {'Copa':>5} {'Recopa':>7} {'TOTAL':>6}  Destino")
    for p in sorted(parts, key=lambda p: p["final_position"]):
        iid, n = p["individual_id"], kids.get(p["individual_id"], 0)
        if not plan:
            dest = "pendiente"
        elif color[iid] in out.statics:
            dest = "sigue (estático: no se reproduce)" if iid in stay else "sale"
        else:
            bits = ["se conserva"] if iid in survivors else []
            if n:
                bits.append(f"{n} hijo{'s' if n != 1 else ''}")
            dest = " + ".join(bits) if bits else "sale"
        out.emit(f"  {p['final_position']:>3}  {out.cell(color[iid], w)} {type_label(out, color[iid]):<8} {p['league_score']:>5} "
                 f"{p['cup_score']:>5} {p['recopa_score']:>7} {p['total_score']:>6}  {dest}")
    if plan:
        zone = rules.SURVIVORS + sum(1 for p in parts if color[p["individual_id"]] in out.statics)
        extra = f" · estáticos que siguen {len(stay)} (puesto ≤ {zone})" if any(color[p["individual_id"]] in out.statics for p in parts) else ""
        out.emit(out.dim(f"  Se conservan {len(survivors)} redes · descendientes {sum(kids.values())}{extra} · "
                         f"siguiente generación {len(plan)} IA"))
    return True


def render_statics(conn, out, cfg) -> None:
    """Bots de ``models_statics/`` y su estado: sin registrar, pendiente, compitiendo o afuera."""
    from . import additions, statics   # import tardío: evita ciclos
    out.banner(f"BOTS ESTÁTICOS · {cfg.statics_dir}")
    bots, errors = statics._scan(cfg.statics_dir)
    for slug, err in errors.items():
        out.emit(f"  ERROR en {slug}.py (se ignora): {err}")
    if not bots:
        out.emit("  No hay bots: poné un archivo .py (una clase que herede de StaticBot) en esa carpeta.")
        return
    for c in bots.values():
        out.statics.setdefault(c.COLOR1, {"name": c.NAME, "color2": c.COLOR2, "slug": ""})
    rows = {r["static_slug"]: r for r in conn.execute("SELECT * FROM individuals WHERE ia_type='estatico'").fetchall()}
    pend = {r["color"]: r["season_number"] for r in additions.list_pending(conn)}
    status = individual_status(conn)
    titles = titles_through(conn)
    out.emit(f"  {'Nombre':<{max(8, out.id_width())}}  {'Colores':<17} {'Archivo':<14} Estado")
    for slug, c in bots.items():
        r = rows.get(slug)
        if r is None:
            state = "sin registrar (añadir con `7 " + c.NAME + "`)"
        elif r["color"] in pend:
            state = f"pendiente: entra en la temporada {pend[r['color']]}"
        elif r["color"] in status and not status[r["color"]]["expelled"]:
            state = f"compitiendo (última temporada: {status[r['color']]['last_season']})"
        elif r["color"] in status:
            state = f"afuera desde la temporada {status[r['color']]['last_season']} (se puede volver a añadir)"
        else:
            state = "registrado"
        t = titles.get(c.COLOR1)
        extra = ""
        if t:
            extra = f"  · títulos: {len(t['L'])} Liga, {len(t['C'])} Copa, {len(t['RC'])} Recopa"
        out.emit(f"  {out.cell(c.COLOR1, max(8, out.id_width()), False)}  {c.COLOR1}/{c.COLOR2:<8} {slug + '.py':<14} {state}{extra}")
    out.emit(out.dim("  Se invocan SIEMPRE por su nombre (menú 4, 5, 6, 7, 9, 10): un número de temporada elige la mejor red."))