"""Medida ABSOLUTA de nivel: cada red juega contra rivales fijos que no evolucionan (ver bots.py).

La puntuación de las temporadas es relativa (la población solo se compara consigo misma): puede
mejorar sin que los puntos lo muestren. Esta medida usa siempre la misma vara, así que un progreso
real se ve como una curva ascendente.
"""
from __future__ import annotations

import random
from typing import Callable

from game import SuperTateti

from . import stats
from .bots import BOTS
from .engine import select_move
from .rng import derive_seed

OPPONENTS = ("random", "tactical")
OPP_NAME = {"random": "bot aleatorio", "tactical": "bot táctico"}

# Referencias medidas con redes SIN entrenar (pesos aleatorios) y con el bot táctico contra el aleatorio.
REFERENCE = {"random": 0.52, "tactical": 0.05}
TACTICAL_VS_RANDOM = 0.96


def evaluate_net(net, opponent: str, games: int, seed: int) -> dict:
    """``games`` partidas de la red contra un bot, alternando lado (pares: la red comienza como X)."""
    rng = random.Random(seed)
    bot = BOTS[opponent](rng)
    tot = [0, 0, 0]
    side_x, side_o = [0, 0, 0], [0, 0, 0]
    for g in range(games):
        side = 1 if g % 2 == 0 else -1
        game = SuperTateti()
        while game.winner is None:
            if game.current_player == side:
                mv = select_move(net, game)
                game.make_move(mv // 9, mv % 9)
            else:
                b, c = bot.choose(game)
                game.make_move(b, c)
        idx = 0 if game.winner == side else 1 if game.winner == 0 else 2
        tot[idx] += 1
        (side_x if side == 1 else side_o)[idx] += 1
    return {"w": tot[0], "d": tot[1], "l": tot[2], "x": side_x, "o": side_o, "games": games}


def _add(a: dict, b: dict) -> dict:
    return {"w": a["w"] + b["w"], "d": a["d"] + b["d"], "l": a["l"] + b["l"], "games": a["games"] + b["games"],
            "x": [p + q for p, q in zip(a["x"], b["x"])], "o": [p + q for p, q in zip(a["o"], b["o"])]}


def side_score(c) -> float:
    n = sum(c)
    return (c[0] + 0.5 * c[1]) / n if n else float("nan")


def should_run(cfg, season_number: int) -> bool:
    return cfg.bench_every > 0 and cfg.bench_games > 0 and (season_number == 1 or season_number % cfg.bench_every == 0)


def run_season_benchmark(db, cfg, info, nets: dict, top1_id: int, check_stop: Callable[[], None] = lambda: None) -> dict | None:
    """Mide al 1.º de la tabla anual (``bench_games`` partidas por rival) y a TODA la población
    (``bench_pop_games`` por IA y rival). Idempotente: si ya está guardado, no se repite."""
    if not should_run(cfg, info.number):
        return None
    have = db.run(lambda c: c.execute("SELECT count(*) AS n FROM season_benchmarks WHERE season_id=%s", (info.id,)).fetchone()["n"])
    rows = []
    if have < 2 * len(OPPONENTS):
        for opp in OPPONENTS:
            check_stop()
            top = evaluate_net(nets[top1_id], opp, cfg.bench_games, derive_seed(info.seed, "bench", "top1", opp))
            rows.append(("top1", opp, top1_id, top, derive_seed(info.seed, "bench", "top1", opp)))
            pop = None
            if cfg.bench_pop_games > 0:
                for iid in sorted(info.color_of):
                    check_stop()
                    r = evaluate_net(nets[iid], opp, cfg.bench_pop_games, derive_seed(info.seed, "bench", "pop", opp, iid))
                    pop = r if pop is None else _add(pop, r)
                rows.append(("population", opp, None, pop, derive_seed(info.seed, "bench", "pop", opp)))

        def save(conn):
            for subject, opp, iid, r, seed in rows:
                conn.execute(
                    """INSERT INTO season_benchmarks (season_id, subject, opponent, individual_id, games, wins, draws, losses,
                                                      x_score, o_score, seed) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT DO NOTHING""",
                    (info.id, subject, opp, iid, r["games"], r["w"], r["d"], r["l"], side_score(r["x"]), side_score(r["o"]), seed))
        db.run(save)
    return summary_for_season(db, info.id)


def summary_for_season(db, season_id: int) -> dict:
    rows = db.run(lambda c: c.execute("SELECT * FROM season_benchmarks WHERE season_id=%s", (season_id,)).fetchall())
    return {(r["subject"], r["opponent"]): stats.outcome_stats(r["wins"], r["draws"], r["losses"]) for r in rows}


# ---------------------------------------------------------------------------
# Medida puntual de una IA (menú)
# ---------------------------------------------------------------------------
def read_score(opponent: str, score: float) -> str:
    """Traduce un puntaje a una lectura en palabras (ver REFERENCE)."""
    if opponent == "random":
        return ("nivel del azar" if score < 0.57 else "algo aprendido" if score < 0.75
                else "buen nivel" if score < 0.90 else "domina al azar")
    return ("muy por debajo del bot táctico" if score < 0.10 else "va mejorando" if score < 0.35
            else "competitiva" if score < 0.60 else "supera al bot táctico")


def bench_ia(ia: dict, games: int, seed: int, load_net) -> dict:
    net = load_net(ia)
    return {opp: evaluate_net(net, opp, games, derive_seed(seed, "bench_ia", opp)) for opp in OPPONENTS}


def render_bench(out, ia: dict, res: dict, games: int, seed: int) -> None:
    out.banner(f"BENCHMARK · {out.ia(ia['color'])} ({ia['label']})")
    out.emit(f"  {games} partidas por rival (mitad como X, mitad como O) · semilla {seed}")
    for opp in OPPONENTS:
        r = res[opp]
        st = stats.outcome_stats(r["w"], r["d"], r["l"])
        out.emit(f"  vs {OPP_NAME[opp]:<13} {r['w']}G {r['d']}E {r['l']}P -> {st['score']:.0%} ±{st['ci']:.0%}  "
                 f"(como X {side_score(r['x']):.0%} · como O {side_score(r['o']):.0%})  -> {read_score(opp, st['score'])}")
    out.emit(out.dim(f"  Referencias: una red SIN entrenar ≈ {REFERENCE['random']:.0%} vs aleatorio y ≈ {REFERENCE['tactical']:.0%} vs táctico; "
                     f"el bot táctico le gana al aleatorio ≈ {TACTICAL_VS_RANDOM:.0%}. "
                     "Un puntaje cuenta victoria 1, empate 0,5, derrota 0."))


# ---------------------------------------------------------------------------
# Medición retroactiva de temporadas ya jugadas
# ---------------------------------------------------------------------------
def backfill(db, cfg, step: int, games: int, emit=print) -> int:
    """Mide al 1.º de la tabla anual de las temporadas completadas SIN medición (una de cada ``step``).

    Solo el 1.º (los .pth del resto de la población pueden haberse borrado). Usa las mismas semillas que la
    medición automática. Se puede ejecutar mientras se entrena. Devuelve cuántas temporadas midió."""
    from . import modelstore

    todo = db.run(lambda c: c.execute(
        """SELECT s.id, s.number, s.seed, p.individual_id, i.model_file, i.model_sha256 FROM seasons s
           JOIN season_participants p ON p.season_id = s.id AND p.final_position = 1
           JOIN individuals i ON i.id = p.individual_id
           WHERE s.status = 'completed'
             AND NOT EXISTS (SELECT 1 FROM season_benchmarks b WHERE b.season_id = s.id)
           ORDER BY s.number""").fetchall())
    todo = [r for r in todo if r["number"] == 1 or r["number"] % step == 0]
    done = 0
    for r in todo:
        path = cfg.models_dir / r["model_file"]
        if not path.exists():
            emit(f"  T{r['number']}: sin .pth, se omite")
            continue
        net = modelstore.load_net(path, r["model_sha256"])
        rows = []
        for opp in OPPONENTS:
            seed = derive_seed(r["seed"], "bench", "top1", opp)
            rows.append((opp, seed, evaluate_net(net, opp, games, seed)))

        def save(conn, r=r, rows=rows):
            for opp, seed, res in rows:
                conn.execute(
                    """INSERT INTO season_benchmarks (season_id, subject, opponent, individual_id, games, wins, draws, losses,
                                                      x_score, o_score, seed) VALUES (%s,'top1',%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT DO NOTHING""",
                    (r["id"], opp, r["individual_id"], res["games"], res["w"], res["d"], res["l"],
                     side_score(res["x"]), side_score(res["o"]), seed))
        db.run(save)
        done += 1
        a = stats.outcome_stats(*[rows[0][2][k] for k in ("w", "d", "l")])
        b = stats.outcome_stats(*[rows[1][2][k] for k in ("w", "d", "l")])
        emit(f"  T{r['number']}: vs aleatorio {a['score']:.0%} · vs táctico {b['score']:.0%}")
    return done