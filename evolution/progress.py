"""Propuesta 1: curva de aprendizaje ABSOLUTA (contra bots fijos) + ritmo + eficacia de las mutaciones.

Todo sale de la base de datos (lo guarda el entrenamiento cada ``BENCH_EVERY`` temporadas).
"""
from __future__ import annotations

import statistics

from . import stats
from .benchmark import OPP_NAME, REFERENCE, read_score

SERIES = [("top1", "random", "1.º de la tabla anual vs aleatorio"),
          ("population", "random", "población entera vs aleatorio"),
          ("top1", "tactical", "1.º de la tabla anual vs táctico"),
          ("population", "tactical", "población entera vs táctico")]


def load_curve(conn) -> dict[int, dict]:
    """{temporada: {(sujeto, rival): stats}} solo de temporadas con medición."""
    rows = conn.execute(
        """SELECT s.number, b.subject, b.opponent, b.wins, b.draws, b.losses FROM season_benchmarks b
           JOIN seasons s ON s.id = b.season_id ORDER BY s.number""").fetchall()
    out: dict[int, dict] = {}
    for r in rows:
        out.setdefault(r["number"], {})[(r["subject"], r["opponent"])] = stats.outcome_stats(r["wins"], r["draws"], r["losses"])
    return out


def pace(conn, last: int = 50) -> dict | None:
    rows = conn.execute(
        """SELECT extract(epoch FROM (completed_at - created_at)) AS secs FROM seasons
           WHERE status='completed' ORDER BY number DESC LIMIT %s""", (last,)).fetchall()
    secs = [float(r["secs"]) for r in rows if r["secs"] is not None]
    if not secs:
        return None
    med = statistics.median(secs)
    return {"n": len(secs), "median": med, "per_hour": 3600 / med if med > 0 else float("nan")}


def mutation_efficacy(conn) -> list[dict]:
    """Por nivel de mutación: ¿los hijos llegan al top 7 de la temporada siguiente? ¿superan a su padre?

    Si los hijos fueran tan buenos como los supervivientes, el 35 % (7 de 20) entraría al top 7 por azar."""
    return conn.execute(
        """WITH kids AS (
               SELECT e.individual_id AS child, e.parent_id, e.mutation_level, s.number AS born_after
               FROM evolution_plan e JOIN seasons s ON s.id = e.season_id WHERE e.role = 'child'),
           res AS (
               SELECT k.mutation_level AS lvl, c.final_position AS cpos, p.final_position AS ppos
               FROM kids k JOIN seasons s2 ON s2.number = k.born_after + 1 AND s2.status = 'completed'
               JOIN season_participants c ON c.season_id = s2.id AND c.individual_id = k.child
               LEFT JOIN season_participants p ON p.season_id = s2.id AND p.individual_id = k.parent_id)
           SELECT lvl, count(*) AS n, avg(cpos)::float AS mean_pos,
                  count(*) FILTER (WHERE cpos <= 7) AS top7,
                  count(ppos) AS with_parent, count(*) FILTER (WHERE cpos < ppos) AS beat_parent
           FROM res GROUP BY lvl ORDER BY CASE lvl WHEN 'small' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END""").fetchall()


def _fmt(st: dict | None) -> str:
    return "      —      " if not st or st["n"] == 0 else f"{st['score']:5.0%} ±{st['ci']:3.0%}"


def render_progress(conn, out, last: int = 25) -> None:
    curve = load_curve(conn)
    out.banner("PROGRESO · curva de aprendizaje contra rivales fijos")
    n_seasons = conn.execute("SELECT count(*) AS n FROM seasons WHERE status='completed'").fetchone()["n"]
    out.emit(f"  Temporadas completadas: {n_seasons} · con medición: {len(curve)}")
    pc = pace(conn)
    if pc:
        out.emit(f"  Ritmo: {pc['median']:.1f}s por temporada (mediana de las últimas {pc['n']}) ≈ {pc['per_hour']:.0f} temporadas/hora")
    if not curve:
        out.emit("  Todavía no hay mediciones: se guardan cada BENCH_EVERY temporadas (por defecto 5; 0 las apaga).")
        out.emit("  Para medir una IA ahora mismo: `9 <temporada|#color>` en el menú.")
    else:
        seasons = sorted(curve)
        shown = seasons[-last:]
        out.section(f"Puntaje contra los bots fijos (última{'s' if len(shown) > 1 else ''} {len(shown)} medición(es); victoria 1, empate 0,5)")
        out.emit(f"  {'Temp':>5}  {'1.º vs aleat.':<13} {'pob. vs aleat.':<14} {'1.º vs táctico':<14} {'pob. vs táctico':<14}")
        for n in shown:
            c = curve[n]
            out.emit(f"  {n:>5}  {_fmt(c.get(('top1', 'random'))):<13} {_fmt(c.get(('population', 'random'))):<14} "
                     f"{_fmt(c.get(('top1', 'tactical'))):<14} {_fmt(c.get(('population', 'tactical'))):<14}")
        out.section("Tendencia (¿sube con las temporadas o es ruido?)")
        for subject, opp, label in SERIES:
            xs = [n for n in seasons if (subject, opp) in curve[n]]
            ys = [curve[n][(subject, opp)]["score"] for n in xs]
            if not ys:
                continue
            t = stats.trend(xs, ys)
            lo, hi = (0.0, 1.0)
            out.emit(f"  {label:<38} {stats.sparkline(ys, 40, lo, hi)}")
            if t["verdict"] == "pocos datos":
                out.emit(f"      {len(ys)} medición(es): hacen falta al menos 6 para opinar. Último: {ys[-1]:.0%}")
            else:
                ref = REFERENCE[opp]
                out.emit(f"      primer tercio {t['first']:.0%} -> último tercio {t['last']:.0%} · pendiente {t['slope_per_100']:+.1f} pts/100 temp. · "
                         f"ρ={t['rho']:+.2f} (umbral ±{t['threshold']:.2f}) -> {out.bold(t['verdict'])}"
                         f" · nivel actual: {read_score(opp, ys[-1])} (red sin entrenar ≈ {ref:.0%})")
    eff = mutation_efficacy(conn)
    out.section("Eficacia de las mutaciones (¿los hijos mejoran algo?)")
    if not eff:
        out.emit("  Aún no hay hijos que hayan jugado una temporada.")
    else:
        out.emit(f"  {'Nivel':<7} {'hijos':>6} {'puesto medio':>13} {'entran al top 7':>16} {'superan a su padre':>20}")
        for r in eff:
            top = r["top7"] / r["n"]
            par = f"{r['beat_parent'] / r['with_parent']:.0%} ({r['with_parent']})" if r["with_parent"] else "—"
            out.emit(f"  {r['lvl']:<7} {r['n']:>6} {r['mean_pos']:>13.1f} {top:>15.0%} {par:>20}")
        out.emit(out.dim("  Referencia: si un hijo fuera tan bueno como los demás, entraría al top 7 el 35 % de las veces "
                         "y superaría a su padre ≈ 50 %. Muy por debajo = las mutaciones empeoran (o el ruido las mata)."))