"""Lógica de competiciones sobre la base de datos.

Cada función recibe ``conn`` (dentro de una transacción) y es IDEMPOTENTE: puede
repetirse tras una caída sin duplicar ni contar dos veces nada. Devuelven "eventos"
(dicts) que el llamador imprime DESPUÉS del commit.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from . import rules
from .rng import derive_rng, derive_seed



@dataclass
class SeasonInfo:
    id: int
    number: int
    seed: int
    league_matchdays: int = rules.LEAGUE_MATCHDAYS           # 38 con 20 equipos; más con IA añadidas
    color_of: dict[int, str] = field(default_factory=dict)  # id individuo -> color
    id_of: dict[str, int] = field(default_factory=dict)     # color -> id individuo

    def rng_factory(self):
        return lambda context: derive_rng(self.seed, "draw", context)


def load_season_info(conn, season_row) -> SeasonInfo:
    rows = conn.execute(
        """SELECT i.id, i.color FROM season_participants p JOIN individuals i ON i.id = p.individual_id
           WHERE p.season_id=%s ORDER BY p.slot""", (season_row["id"],)).fetchall()
    return SeasonInfo(season_row["id"], season_row["number"], season_row["seed"],
                      season_row.get("league_matchdays") or rules.LEAGUE_MATCHDAYS,
                      {r["id"]: r["color"] for r in rows}, {r["color"]: r["id"] for r in rows})



def persist_draws(conn, s: SeasonInfo, draws: list[rules.DrawRecord]) -> None:
    for d in draws:
        conn.execute(
            """INSERT INTO tiebreak_draws (season_id, context, seed, members, result_order)
               VALUES (%s,%s,%s,%s::jsonb,%s::jsonb) ON CONFLICT (season_id, context) DO NOTHING""",
            (s.id, d.context, derive_seed(s.seed, "draw", d.context), json.dumps(d.members), json.dumps(d.result_order)))


# ---------------------------------------------------------------------------
# Partidas: reclamar y registrar de forma atómica
# ---------------------------------------------------------------------------
_ORDER = "CASE competition WHEN 'league' THEN 0 WHEN 'cup_group' THEN 1 WHEN 'cup_ko' THEN 2 ELSE 3 END"


def reset_running(conn, season_id: int) -> int:
    """Partidas que quedaron 'running' por una interrupción: se reinician.

    Decisión: no se guarda el estado a mitad de partida. La selección de jugada es
    determinista y una partida dura ~10 ms, así que repetirla desde cero produce
    EXACTAMENTE el mismo resultado; solo se contabiliza al registrarse como
    'finished' (UPDATE condicionado a status<>'finished'), por lo que nunca cuenta dos veces.
    """
    return conn.execute(
        "UPDATE matches SET status='pending', started_at=NULL WHERE season_id=%s AND status='running'",
        (season_id,)).rowcount


def fetch_pending(conn, season_id: int, upto_matchday: int) -> list[dict]:
    return conn.execute(
        f"""SELECT * FROM matches WHERE season_id=%s AND status<>'finished' AND matchday<=%s
            ORDER BY matchday, {_ORDER}, id""", (season_id, upto_matchday)).fetchall()


def claim_match(conn, match_id: int) -> bool:
    """Marca 'running'. False si ya estaba finalizada (no se vuelve a jugar)."""
    row = conn.execute("SELECT status FROM matches WHERE id=%s", (match_id,)).fetchone()
    if row["status"] == "finished":
        return False
    conn.execute("UPDATE matches SET status='running', started_at=now() WHERE id=%s AND status='pending'", (match_id,))
    return True


def record_result(conn, match: dict, outcome) -> bool:
    """Registra el resultado de forma atómica. False si ya estaba finalizada."""
    winner = match["x_individual_id"] if outcome.result == "x" else match["o_individual_id"] if outcome.result == "o" else None
    n = conn.execute(
        """UPDATE matches SET status='finished', result=%s, winner_id=%s, moves=%s, num_moves=%s,
                              duration_ms=%s, finished_at=now()
           WHERE id=%s AND status<>'finished'""",
        (outcome.result, winner, outcome.moves, outcome.num_moves, outcome.duration_ms, match["id"])).rowcount
    return n == 1


def progress(conn, season_id: int) -> dict:
    r = conn.execute(
        """SELECT count(*) FILTER (WHERE status='finished') AS done, count(*) AS total,
                  COALESCE(sum(duration_ms) FILTER (WHERE status='finished'),0) AS sim_ms
           FROM matches WHERE season_id=%s""", (season_id,)).fetchone()
    return {"done": r["done"], "total": r["total"], "pending": r["total"] - r["done"], "sim_ms": r["sim_ms"]}


# ---------------------------------------------------------------------------
# Liga
# ---------------------------------------------------------------------------
def league_games(conn, season_id: int, upto: int, s: SeasonInfo):
    rows = conn.execute(
        """SELECT matchday, x_individual_id x, o_individual_id o, result FROM matches
           WHERE season_id=%s AND competition='league' AND status='finished' AND matchday<=%s""",
        (season_id, upto)).fetchall()
    return [(s.color_of[r["x"]], s.color_of[r["o"]], r["result"]) for r in rows]


def close_matchday(conn, s: SeasonInfo, md: int) -> dict:
    """Cierra la fecha: guarda la tabla de Liga tras la fecha ``md`` (evolución de puntos)."""
    cur_md = conn.execute("SELECT current_matchday FROM seasons WHERE id=%s", (s.id,)).fetchone()["current_matchday"]
    if cur_md >= md:
        return {"already_closed": True}
    unfinished = conn.execute(
        "SELECT count(*) AS n FROM matches WHERE season_id=%s AND competition='league' AND matchday<=%s AND status<>'finished'",
        (s.id, md)).fetchone()["n"]
    if unfinished:
        raise RuntimeError(f"Fecha {md}: quedan {unfinished} partidas de Liga sin terminar")
    table = rules.build_table(list(s.id_of), league_games(conn, s.id, md, s))
    # Posición provisional (solo informativa): puntos y, a igualdad, color. El desempate real se hace al final.
    ordered = sorted(table.values(), key=lambda r: (-r.points, r.pid))
    for pos, row in enumerate(ordered, start=1):
        conn.execute(
            """INSERT INTO league_history (season_id, matchday, individual_id, played, wins, draws, losses, points, position)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            (s.id, md, s.id_of[row.pid], row.played, row.wins, row.draws, row.losses, row.points, pos))
    conn.execute("UPDATE seasons SET current_matchday=%s WHERE id=%s AND current_matchday<%s", (md, s.id, md))
    return {"already_closed": False}


def league_table_snapshot(conn, s: SeasonInfo, md: int) -> list[dict]:
    rows = conn.execute(
        """SELECT individual_id, played, wins, draws, losses, points, position FROM league_history
           WHERE season_id=%s AND matchday=%s ORDER BY position""", (s.id, md)).fetchall()
    return [{**r, "color": s.color_of[r["individual_id"]]} for r in rows]


def finish_league(conn, s: SeasonInfo) -> list[dict]:
    """Clasificación definitiva tras la última fecha de Liga (con desempates) y recompensa evolutiva."""
    season = conn.execute("SELECT * FROM seasons WHERE id=%s", (s.id,)).fetchone()
    if season["stage"] != "league_cup":
        return []
    if season["current_matchday"] != s.league_matchdays:
        raise RuntimeError("La Liga no ha terminado")
    if season["cup_champion_id"] is None:
        raise RuntimeError("La Copa no ha terminado (la final es en la fecha 37)")
    hist = defaultdict(dict)
    for r in conn.execute("SELECT matchday, individual_id, points FROM league_history WHERE season_id=%s", (s.id,)):
        hist[s.color_of[r["individual_id"]]][r["matchday"]] = r["points"]
    history = {c: [hist[c][d] for d in range(1, s.league_matchdays + 1)] for c in s.id_of}
    final_points = {c: history[c][-1] for c in s.id_of}
    draws: list[rules.DrawRecord] = []
    order = rules.league_final_order(list(s.id_of), final_points, history, s.rng_factory(), draws)
    persist_draws(conn, s, draws)
    for pos, c in enumerate(order, start=1):
        conn.execute(
            """UPDATE season_participants SET league_points=%s, league_played=%s, league_position=%s, league_score=%s
               WHERE season_id=%s AND individual_id=%s""",
            (final_points[c], 2 * (len(s.id_of) - 1), pos, rules.league_reward(pos), s.id, s.id_of[c]))
    conn.execute("UPDATE seasons SET league_champion_id=%s, stage='recopa' WHERE id=%s", (s.id_of[order[0]], s.id))
    return [{"type": "league_final", "order": order, "points": final_points, "draws": [d.context for d in draws]}]


# ---------------------------------------------------------------------------
# Copa
# ---------------------------------------------------------------------------
def _create_tie_matches(conn, s: SeasonInfo, tie: dict, a_id: int, b_id: int) -> None:
    conn.execute(
        "UPDATE cup_ties SET individual_a_id=%s, individual_b_id=%s, status='pending' WHERE id=%s",
        (a_id, b_id, tie["id"]))
    for seq, (x, o) in enumerate([(a_id, b_id), (b_id, a_id)], start=1):  # ida y vuelta: alternan quién comienza
        conn.execute(
            """INSERT INTO matches (season_id, competition, matchday, round_no, tie_id, seq, x_individual_id, o_individual_id)
               VALUES (%s,'cup_ko',%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            (s.id, tie["matchday"], rules.KNOCKOUT_ROUNDS.index(tie["round"]) + 1, tie["id"], seq, x, o))


def _resolve_groups(conn, s: SeasonInfo) -> list[dict]:
    games = defaultdict(list)
    for r in conn.execute(
            "SELECT group_name g, x_individual_id x, o_individual_id o, result FROM matches "
            "WHERE season_id=%s AND competition='cup_group'", (s.id,)):
        games[r["g"]].append((s.color_of[r["x"]], s.color_of[r["o"]], r["result"]))
    members = defaultdict(list)
    for r in conn.execute("SELECT cup_group g, individual_id FROM season_participants WHERE season_id=%s", (s.id,)):
        members[r["g"]].append(s.color_of[r["individual_id"]])
    draws: list[rules.DrawRecord] = []
    pos_to_color: dict[str, str] = {}
    tables = {}
    for g in "ABCD":
        table = rules.build_table(members[g], games[g])
        order = rules.rank_with_draws(members[g], key=lambda p: -table[p].points,
                                      context=f"cup_group:{g}", rng_factory=s.rng_factory(), draws=draws)
        tables[g] = [(p, table[p]) for p in order]
        for pos, p in enumerate(order, start=1):
            t = table[p]
            conn.execute(
                """INSERT INTO cup_group_table (season_id, group_name, individual_id, position, played, wins, draws,
                                                losses, points, qualified) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING""",
                (s.id, g, s.id_of[p], pos, t.played, t.wins, t.draws, t.losses, t.points, pos <= 4))
            pos_to_color[f"{pos}{g}"] = p
            if pos > 4:   # grupos de 5, 6, 7 u 8: clasifican 4, el resto queda eliminado
                conn.execute("UPDATE season_participants SET cup_stage='group', cup_score=%s WHERE season_id=%s AND individual_id=%s",
                             (rules.CUP_REWARD["group"], s.id, s.id_of[p]))
    persist_draws(conn, s, draws)
    r16 = conn.execute("SELECT * FROM cup_ties WHERE season_id=%s AND round='R16' ORDER BY tie_no", (s.id,)).fetchall()
    for tie in r16:
        _create_tie_matches(conn, s, tie, s.id_of[pos_to_color[tie["slot_a"]]], s.id_of[pos_to_color[tie["slot_b"]]])
    conn.execute("UPDATE seasons SET cup_groups_resolved=TRUE WHERE id=%s", (s.id,))
    return [{"type": "groups_resolved", "tables": tables, "draws": [d.context for d in draws],
             "r16": [(t["slot_a"], pos_to_color[t["slot_a"]], t["slot_b"], pos_to_color[t["slot_b"]]) for t in r16]}]


def _finish_tie(conn, s: SeasonInfo, tie: dict, winner_color: str, resolution: str) -> list[dict]:
    a, b = tie["individual_a_id"], tie["individual_b_id"]
    w = s.id_of[winner_color]
    loser = b if w == a else a
    conn.execute("UPDATE cup_ties SET status='finished', winner_id=%s, resolution=%s WHERE id=%s", (w, resolution, tie["id"]))
    rnd = tie["round"]
    ev = [{"type": "tie_finished", "round": rnd, "tie_no": tie["tie_no"], "winner": winner_color,
           "loser": s.color_of[loser], "resolution": resolution}]
    if rnd == "F":
        for pid, stage in ((w, "champion"), (loser, "runner_up")):
            conn.execute("UPDATE season_participants SET cup_stage=%s, cup_score=%s WHERE season_id=%s AND individual_id=%s",
                         (stage, rules.CUP_REWARD[stage], s.id, pid))
        conn.execute("UPDATE seasons SET cup_champion_id=%s, cup_runner_up_id=%s WHERE id=%s", (w, loser, s.id))
        ev.append({"type": "cup_finished", "champion": winner_color, "runner_up": s.color_of[loser]})
        return ev
    conn.execute("UPDATE season_participants SET cup_stage=%s, cup_score=%s WHERE season_id=%s AND individual_id=%s",
                 (rnd, rules.CUP_REWARD[rnd], s.id, loser))
    nxt = rules.knockout_next_round(rnd)
    nxt_no = (tie["tie_no"] + 1) // 2
    sib_no = tie["tie_no"] + 1 if tie["tie_no"] % 2 else tie["tie_no"] - 1
    sibling = conn.execute("SELECT * FROM cup_ties WHERE season_id=%s AND round=%s AND tie_no=%s",
                           (s.id, rnd, sib_no)).fetchone()
    if sibling["status"] == "finished":
        first, second = (tie, sibling) if tie["tie_no"] < sibling["tie_no"] else (sibling, tie)
        w1 = w if first["id"] == tie["id"] else sibling["winner_id"]
        w2 = w if second["id"] == tie["id"] else sibling["winner_id"]
        nt = conn.execute("SELECT * FROM cup_ties WHERE season_id=%s AND round=%s AND tie_no=%s", (s.id, nxt, nxt_no)).fetchone()
        if nt["individual_a_id"] is None:
            _create_tie_matches(conn, s, nt, w1, w2)
            ev.append({"type": "tie_scheduled", "round": nxt, "tie_no": nxt_no, "a": s.color_of[w1], "b": s.color_of[w2],
                       "matchday": nt["matchday"]})
    return ev


def advance_cup(conn, s: SeasonInfo) -> list[dict]:
    """Hace avanzar la Copa con lo que ya esté jugado: cierra grupos, resuelve
    eliminatorias, crea desempates y programa la ronda siguiente."""
    events: list[dict] = []
    season = conn.execute("SELECT cup_groups_resolved FROM seasons WHERE id=%s", (s.id,)).fetchone()
    if not season["cup_groups_resolved"]:
        pend = conn.execute("SELECT count(*) AS n FROM matches WHERE season_id=%s AND competition='cup_group' AND status<>'finished'",
                            (s.id,)).fetchone()["n"]
        if pend == 0:
            events += _resolve_groups(conn, s)
    ties = conn.execute("SELECT * FROM cup_ties WHERE season_id=%s AND status='pending' ORDER BY matchday, tie_no", (s.id,)).fetchall()
    for tie in ties:
        legs = conn.execute("SELECT * FROM matches WHERE tie_id=%s ORDER BY seq", (tie["id"],)).fetchall()
        if len(legs) < 2 or any(m["status"] != "finished" for m in legs):
            continue
        ca, cb = s.color_of[tie["individual_a_id"]], s.color_of[tie["individual_b_id"]]
        if len(legs) == 2:
            winner, res = rules.evaluate_two_legs(
                ca, cb, [(s.color_of[m["x_individual_id"]], s.color_of[m["o_individual_id"]], m["result"]) for m in legs])
            if winner is None:  # eliminatoria empatada: tercera partida con inicio sorteado (reproducible)
                ctx = f"cup_third:{tie['round']}:{tie['tie_no']}"
                draws: list[rules.DrawRecord] = []
                order = rules.rank_with_draws([ca, cb], key=lambda p: 0, context=ctx, rng_factory=s.rng_factory(), draws=draws)
                persist_draws(conn, s, draws)
                x, o = s.id_of[order[0]], s.id_of[order[1]]
                conn.execute(
                    """INSERT INTO matches (season_id, competition, matchday, round_no, tie_id, seq, x_individual_id, o_individual_id)
                       VALUES (%s,'cup_ko',%s,%s,%s,3,%s,%s) ON CONFLICT DO NOTHING""",
                    (s.id, tie["matchday"], rules.KNOCKOUT_ROUNDS.index(tie["round"]) + 1, tie["id"], x, o))
                events.append({"type": "tie_third_game", "round": tie["round"], "tie_no": tie["tie_no"],
                               "x": order[0], "o": order[1]})
                continue
            events += _finish_tie(conn, s, tie, winner, res)
        else:
            third = legs[2]
            winner, res = rules.evaluate_third_game(s.color_of[third["x_individual_id"]], s.color_of[third["o_individual_id"]], third["result"])
            events += _finish_tie(conn, s, tie, winner, res)
    return events


# ---------------------------------------------------------------------------
# Recopa
# ---------------------------------------------------------------------------
def setup_recopa(conn, s: SeasonInfo) -> list[dict]:
    season = conn.execute("SELECT * FROM seasons WHERE id=%s", (s.id,)).fetchone()
    if season["stage"] != "recopa" or conn.execute("SELECT 1 FROM recopa_results WHERE season_id=%s", (s.id,)).fetchone():
        return []
    cup, league = season["cup_champion_id"], season["league_champion_id"]
    if cup == league:
        conn.execute(
            """INSERT INTO recopa_results (season_id, cup_champion_id, league_champion_id, same_champion, champion_id, resolution)
               VALUES (%s,%s,%s,TRUE,%s,'same_champion')""", (s.id, cup, league, cup))
        conn.execute("UPDATE season_participants SET recopa_score=%s WHERE season_id=%s AND individual_id=%s",
                     (rules.RECOPA_BONUS, s.id, cup))
        conn.execute("UPDATE seasons SET recopa_champion_id=%s, stage='scoring' WHERE id=%s", (cup, s.id))
        return [{"type": "recopa_same", "champion": s.color_of[cup]}]
    conn.execute("INSERT INTO recopa_results (season_id, cup_champion_id, league_champion_id, same_champion) VALUES (%s,%s,%s,FALSE)",
                 (s.id, cup, league))
    for g in range(1, rules.RECOPA_GAMES + 1):
        x, o = (cup, league) if g % 2 == 1 else (league, cup)  # 5 y 5
        conn.execute(
            """INSERT INTO matches (season_id, competition, matchday, round_no, seq, x_individual_id, o_individual_id)
               VALUES (%s,'recopa',%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            (s.id, s.league_matchdays + 1, 1, g, x, o))
    return [{"type": "recopa_scheduled", "cup": s.color_of[cup], "league": s.color_of[league]}]


def finish_recopa(conn, s: SeasonInfo) -> list[dict]:
    season = conn.execute("SELECT * FROM seasons WHERE id=%s", (s.id,)).fetchone()
    if season["stage"] != "recopa":
        return []
    res = conn.execute("SELECT * FROM recopa_results WHERE season_id=%s", (s.id,)).fetchone()
    games = [(s.color_of[m["x_individual_id"]], s.color_of[m["o_individual_id"]], m["result"]) for m in conn.execute(
        "SELECT * FROM matches WHERE season_id=%s AND competition='recopa' AND status='finished' ORDER BY seq", (s.id,))]
    if len(games) < rules.RECOPA_GAMES:
        return []
    cup_c, lg_c = s.color_of[res["cup_champion_id"]], s.color_of[res["league_champion_id"]]
    wc, wl, d, winner = rules.recopa_outcome(cup_c, lg_c, games)
    resolution = "wins"
    if winner is None:  # empate global: sorteo reproducible, sin undécima partida
        draws: list[rules.DrawRecord] = []
        order = rules.rank_with_draws([cup_c, lg_c], key=lambda p: 0, context="recopa", rng_factory=s.rng_factory(), draws=draws)
        persist_draws(conn, s, draws)
        winner, resolution = order[0], "random_draw"
    wid = s.id_of[winner]
    conn.execute("""UPDATE recopa_results SET cup_champ_wins=%s, league_champ_wins=%s, draws=%s, champion_id=%s, resolution=%s
                    WHERE season_id=%s""", (wc, wl, d, wid, resolution, s.id))
    conn.execute("UPDATE season_participants SET recopa_score=%s WHERE season_id=%s AND individual_id=%s",
                 (rules.RECOPA_BONUS, s.id, wid))
    conn.execute("UPDATE seasons SET recopa_champion_id=%s, stage='scoring' WHERE id=%s", (wid, s.id))
    return [{"type": "recopa_finished", "champion": winner, "cup": cup_c, "league": lg_c,
             "wins_cup": wc, "wins_league": wl, "draws": d, "resolution": resolution}]


# ---------------------------------------------------------------------------
# Puntuación global
# ---------------------------------------------------------------------------
def compute_global(conn, s: SeasonInfo) -> list[dict]:
    season = conn.execute("SELECT stage FROM seasons WHERE id=%s", (s.id,)).fetchone()
    if season["stage"] != "scoring":
        return []
    rows = conn.execute("SELECT * FROM season_participants WHERE season_id=%s", (s.id,)).fetchall()
    by = {s.color_of[r["individual_id"]]: r for r in rows}
    draws: list[rules.DrawRecord] = []
    order = rules.global_order(
        list(by), {c: r["league_score"] for c, r in by.items()}, {c: r["cup_score"] for c, r in by.items()},
        {c: r["recopa_score"] for c, r in by.items()}, {c: r["league_position"] for c, r in by.items()},
        s.rng_factory(), draws)
    persist_draws(conn, s, draws)
    for pos, c in enumerate(order, start=1):
        r = by[c]
        conn.execute("UPDATE season_participants SET total_score=%s, final_position=%s WHERE season_id=%s AND individual_id=%s",
                     (r["league_score"] + r["cup_score"] + r["recopa_score"], pos, s.id, s.id_of[c]))
    conn.execute("UPDATE seasons SET stage='evolution' WHERE id=%s", (s.id,))
    return [{"type": "global_ranking", "order": order}]