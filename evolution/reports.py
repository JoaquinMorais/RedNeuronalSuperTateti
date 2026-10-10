"""Consultas de solo lectura y vistas para analizar temporadas desde la terminal.

No toman el bloqueo de entrenamiento: se pueden usar mientras se entrena.
"""
from __future__ import annotations

import re
from collections import Counter

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


def titles_through(conn, upto: int | None = None) -> dict:
    return titlemod.build_title_index(champion_rows(conn, upto))


# ---------------------------------------------------------------------------
# Estado de las temporadas (+ campeones)
# ---------------------------------------------------------------------------
def render_status(conn, out: Display) -> None:
    champs = {r["number"]: r for r in champion_rows(conn)}
    seasons = conn.execute("SELECT number, status, stage, current_matchday FROM seasons ORDER BY number").fetchall()
    if not seasons:
        out.emit("Todavía no hay temporadas.")
    for s in seasons:
        line = f"Temporada {s['number']}: {s['status']} · etapa {s['stage']} · fecha cerrada {s['current_matchday']}/38"
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


def render_champions(conn, out: Display) -> None:
    rows = champion_rows(conn)
    out.banner("CAMPEONES")
    if not rows:
        out.emit("  Todavía no terminó ninguna temporada.")
        return
    out.section("Campeones de cada temporada")
    out.emit(f"  {'Temp':>4}  {'Liga':<10} {'Copa':<10} {'Subcampeón':<11} {'Recopa':<10}")
    for r in rows:
        out.emit(f"  {r['number']:>4}  {out.cell(r['league'], 10, False)} {out.cell(r['cup'], 10, False)} "
                 f"{out.cell(r['runner_up'], 11, False)} {out.cell(r['recopa'], 10, False)}")
    out.section("Ranking de campeones (prioridad: Liga, Copa, Recopa)")
    idx = titlemod.build_title_index(rows)
    ranking = titlemod.rank_champions(idx)
    out.emit(f"  {'Pos':>3}  {'IA':<10} {'Liga':>4} {'Copa':>4} {'Recopa':>6}   Títulos")
    for pos, (color, t) in enumerate(ranking, start=1):
        det = " · ".join(f"{name}: {_seasons_txt(t[k])}" for k, name in (("L", "Ligas"), ("C", "Copas"), ("RC", "Recopas")) if t[k])
        out.emit(f"  {pos:>3}  {out.cell(color, 10, False)} {len(t['L']):>4} {len(t['C']):>4} {len(t['RC']):>6}   {det}")


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
    return [_TAG.sub(lambda m: out.paint(m.group(0), m.group(1)), ln) for ln in lines]


# ---------------------------------------------------------------------------
# Vista de una temporada
# ---------------------------------------------------------------------------
def render_season(conn, out: Display, number: int) -> bool:
    s = conn.execute("SELECT * FROM seasons WHERE number=%s", (number,)).fetchone()
    if s is None:
        out.emit(f"No existe la temporada {number}.")
        return False
    out.titles = titles_through(conn, number)  # insignias "al terminar" esa temporada
    color = {r["id"]: r["color"] for r in conn.execute("SELECT id, color FROM individuals")}
    parts = conn.execute("SELECT * FROM season_participants WHERE season_id=%s", (s["id"],)).fetchall()
    sid = s["id"]
    out.banner(f"TEMPORADA {number} · {s['status']} · etapa {s['stage']} · fecha cerrada {s['current_matchday']}/38")

    # --- Liga ---------------------------------------------------------
    last_md = conn.execute("SELECT max(matchday) m FROM league_history WHERE season_id=%s", (sid,)).fetchone()["m"]
    w = out.id_width() + (2 if out.color else 0)
    if last_md is None:
        out.section("Liga")
        out.emit("  Aún sin partidas jugadas.")
    else:
        hist = {r["individual_id"]: r for r in conn.execute(
            "SELECT * FROM league_history WHERE season_id=%s AND matchday=%s", (sid, last_md))}
        final = s["current_matchday"] >= 38 and all(p["league_position"] for p in parts)
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
        out.emit("  Pendiente (se juega tras la fecha 38).")
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
    out.emit(f"  {'Pos':>3}  {'IA':<{w}} {'Liga':>5} {'Copa':>5} {'Recopa':>7} {'TOTAL':>6}  Destino")
    for p in sorted(parts, key=lambda p: p["final_position"]):
        iid, n = p["individual_id"], kids.get(p["individual_id"], 0)
        if not plan:
            dest = "pendiente"
        else:
            bits = ["se conserva"] if iid in survivors else []
            if n:
                bits.append(f"{n} hijo{'s' if n != 1 else ''}")
            dest = " + ".join(bits) if bits else "sale"
        out.emit(f"  {p['final_position']:>3}  {out.cell(color[iid], w)} {p['league_score']:>5} {p['cup_score']:>5} "
                 f"{p['recopa_score']:>7} {p['total_score']:>6}  {dest}")
    if plan:
        out.emit(out.dim(f"  Se conservan {len(survivors)} · descendientes {sum(kids.values())} · "
                         f"siguiente generación {len(plan)} IA"))
    return True