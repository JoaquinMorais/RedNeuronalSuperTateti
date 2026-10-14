"""Medición de progreso (bots fijos, curva, torneo maestro), espacio en disco y niveles de salida."""
import io
import random
import re
from pathlib import Path

import pytest
import torch

from evolution import benchmark, bots, disk, ladder, modelstore, progress, prune, reports, stats
from evolution.__main__ import main
from evolution.config import load_config
from evolution.display import Display
from game import SuperTateti
from model import SuperTatetiNet
from tests.helpers import TEST_URL, Env


def _net(seed):
    torch.manual_seed(seed)
    n = SuperTatetiNet().eval()
    for p in n.parameters():
        p.requires_grad_(False)
    return n


# ====================== bots ======================
def test_bots_juegan_legal_y_son_reproducibles():
    for cls in (bots.RandomBot, bots.TacticalBot):
        g1, g2 = SuperTateti(), SuperTateti()
        b1, b2 = cls(random.Random(5)), cls(random.Random(5))
        for g, b in ((g1, b1), (g2, b2)):
            seq = []
            while g.winner is None:
                mv = b.choose(g)
                assert g.get_valid_moves()[mv[0] * 9 + mv[1]] == 1
                g.make_move(*mv); seq.append(mv)
            g.seq = seq
        assert g1.seq == g2.seq


def test_bot_tactico_gana_cuando_puede():
    g = SuperTateti()                       # X puede cerrar el tablero chico 0 en la casilla 2
    g.board[0][0] = g.board[0][1] = 1
    g.board[1][3] = g.board[1][4] = -1
    g.active_board, g.current_player = 0, 1
    assert bots.TacticalBot(random.Random(1)).choose(g) == (0, 2)
    g2 = SuperTateti()                      # X puede ganar la partida: ya tiene los tableros 0 y 1
    g2.small_boards[0] = g2.small_boards[1] = 1
    g2.board[2][0] = g2.board[2][1] = 1
    g2.active_board, g2.current_player = 2, 1
    assert bots.TacticalBot(random.Random(1)).choose(g2) == (2, 2)


def test_bot_tactico_bloquea_un_tablero_del_rival():
    g = SuperTateti()                       # O amenaza cerrar el tablero 0 en la casilla 2
    g.board[0][0] = g.board[0][1] = -1
    g.active_board, g.current_player = 0, 1
    assert bots.TacticalBot(random.Random(1)).choose(g) == (0, 2)


def test_bot_tactico_domina_al_aleatorio():
    rng = random.Random(3)
    wins = 0
    for i in range(40):
        side = 1 if i % 2 == 0 else -1
        t, r, g = bots.TacticalBot(rng), bots.RandomBot(rng), SuperTateti()
        while g.winner is None:
            b, c = (t if g.current_player == side else r).choose(g)
            g.make_move(b, c)
        wins += g.winner == side
    assert wins >= 30                       # la escala de medición tiene resolución


# ====================== estadística ======================
def test_outcome_stats():
    s = stats.outcome_stats(10, 0, 10)
    assert s["n"] == 20 and s["score"] == 0.5 and abs(s["ci"] - 1.96 * (0.25 / 20) ** 0.5) < 1e-9
    assert stats.outcome_stats(0, 0, 0)["n"] == 0


def test_spearman_y_tendencia():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert stats.spearman([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0
    xs = list(range(1, 21))
    up = [0.4 + 0.01 * x + (0.01 if x % 2 else -0.01) for x in xs]
    flat = [0.5 + (0.03 if x % 2 else -0.03) for x in xs]
    down = [0.9 - 0.01 * x for x in xs]
    assert stats.trend(xs, up)["verdict"] == "MEJORA" and stats.trend(xs, up)["slope_per_100"] > 0
    assert stats.trend(xs, flat)["verdict"] == "SIN TENDENCIA CLARA"
    assert stats.trend(xs, down)["verdict"] == "EMPEORA"
    assert stats.trend([1, 2, 3], [0.1, 0.2, 0.3])["verdict"] == "pocos datos"


def test_sparkline():
    s = stats.sparkline([0, 0.5, 1], 10)
    assert len(s) == 3 and s[0] == "▁" and s[-1] == "█"
    assert len(stats.sparkline(list(range(100)), 20, 0, 99)) == 20 and stats.sparkline([]) == ""


def test_bradley_terry_ordena_por_fuerza():
    score = {(0, 1): 2.0, (1, 0): 0.0, (0, 2): 2.0, (2, 0): 0.0, (1, 2): 1.5, (2, 1): 0.5}
    games = {(0, 1): 2, (1, 0): 2, (0, 2): 2, (2, 0): 2, (1, 2): 2, (2, 1): 2}
    elo = stats.bradley_terry(3, score, games)
    assert elo[0] > elo[1] > elo[2] and sum(elo) / 3 == pytest.approx(1000, abs=1.0)


# ====================== medición ======================
def test_evaluate_net_cuenta_y_es_reproducible():
    net = _net(1)
    r = benchmark.evaluate_net(net, "random", 11, seed=9)
    assert r["w"] + r["d"] + r["l"] == 11 == r["games"] and sum(r["x"]) == 6 and sum(r["o"]) == 5   # lados alternados
    assert benchmark.evaluate_net(net, "random", 11, seed=9) == r
    assert benchmark.evaluate_net(net, "tactical", 4, seed=9)["games"] == 4


def test_should_run():
    cfg = load_config({"BENCH_EVERY": "5"})
    assert [n for n in range(1, 16) if benchmark.should_run(cfg, n)] == [1, 5, 10, 15]
    assert not benchmark.should_run(load_config({"BENCH_EVERY": "0"}), 1)
    assert not benchmark.should_run(load_config({"BENCH_GAMES": "0"}), 1)


def test_lectura_de_puntajes():
    assert benchmark.read_score("random", 0.45) == "nivel del azar" and benchmark.read_score("random", 0.95) == "domina al azar"
    assert benchmark.read_score("tactical", 0.05).startswith("muy por debajo") and benchmark.read_score("tactical", 0.7).startswith("supera")


def test_configuracion_nueva():
    c = load_config({})
    assert (c.bench_every, c.protect_every, c.compact_after, c.output) == (5, 1, 0, "full")
    c = load_config({"OUTPUT": "TEMPORADA", "MODEL_PROTECT_EVERY": "10", "DB_COMPACT_AFTER": "50"})
    assert (c.output, c.protect_every, c.compact_after) == ("temporada", 10, 50)
    for bad in ({"OUTPUT": "mucho"}, {"MODEL_PROTECT_EVERY": "0"}, {"BENCH_EVERY": "-1"}, {"DB_COMPACT_AFTER": "-2"}):
        with pytest.raises(ValueError):
            load_config(bad)


# ====================== torneo maestro ======================
def _champs(tmp_path, n=4):
    out = []
    for i in range(n):
        p = tmp_path / f"{i:02x}{i:02x}{i:02x}.pth"
        digest = modelstore.save_model(p, _net(300 + i).state_dict(), {"color": f"#{i:02x}{i:02x}{i:02x}"})
        out.append({"season": 1 + 10 * i, "color": f"#{i:02x}{i:02x}{i:02x}", "path": p, "sha256": digest})
    return out


def test_torneo_maestro(tmp_path):
    champs = _champs(tmp_path, 4)
    res = ladder.run_ladder(champs)
    assert len(res["elo"]) == 4 and 0 <= res["newer_share"] <= 1 and res["triples"] == 4 and 0 <= res["cycles"] <= 4
    # 2 partidas por pareja: los puntos de i contra j y de j contra i suman 2
    assert all(res["score"][(i, j)] + res["score"][(j, i)] == 2.0 for i in range(4) for j in range(4) if i != j)
    out = Display(False, io.StringIO()); ladder.render_ladder(out, res)
    t = out.stream.getvalue()
    assert "TORNEO MAESTRO" in t and "Veredicto" in t and "Ciclos" in t
    small = Display(False, io.StringIO()); ladder.render_ladder(small, ladder.run_ladder(champs[:2]))
    assert "al menos 3" in small.stream.getvalue()


# ====================== niveles de salida ======================
def test_niveles_de_salida(env):
    env.trainer(level="fecha").run()
    txt = env.out.getvalue()
    assert "(comienza)" not in txt and "Tabla de Liga tras la fecha" not in txt         # sin partidas ni tablas por fecha
    assert "TEMPORADA 1" in txt and "FIN DE LA TEMPORADA 1" in txt
    assert len(re.findall(r"^\s+Fecha \d+:", txt, re.M)) == 38                             # una línea por fecha
    assert len(txt.splitlines()) < 300


# ====================== cuatro temporadas: medición, limpieza, compactación, espacio ======================
def test_cuatro_temporadas_medicion_limpieza_compactacion_y_disco(tmp_path):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    e = Env(tmp_path)
    cfg_kw = dict(retention_seasons=1, protect_every=2, compact_after=1, bench_every=2, bench_games=6, bench_pop_games=2)
    try:
        e.trainer(max_seasons=4, level="temporada", **cfg_kw).run()
        txt = e.out.getvalue()
        # --- salida compacta: una línea por temporada (+ limpieza), nada de fechas ---
        lines = txt.splitlines()
        assert "FECHA" not in txt and len([l for l in lines if l.startswith("Temporada ")]) == 4 and len(lines) < 15
        assert [("MEDIDA" in l) for l in lines if l.startswith("Temporada ")] == [True, True, False, True]
        # --- mediciones guardadas (temporadas 1, 2 y 4) ---
        rows = e.q("SELECT s.number, b.subject, b.opponent, b.games FROM season_benchmarks b JOIN seasons s ON s.id=b.season_id ORDER BY 1,2,3")
        assert sorted({r["number"] for r in rows}) == [1, 2, 4] and len(rows) == 12
        assert {(r["subject"], r["games"]) for r in rows} == {("top1", 6), ("population", 40)}
        out = Display(False, io.StringIO()); progress.render_progress(e.conn(), out)
        t = out.stream.getvalue()
        assert "con medición: 3" in t and "temporadas/hora" in t and "hacen falta al menos 6" in t
        eff = {r["lvl"]: r["n"] for r in progress.mutation_efficacy(e.conn())}
        assert sum(eff.values()) == 13 * 3                                              # hijos de las temporadas 1-3 que ya jugaron
        assert progress.pace(e.conn())["median"] > 0
        # --- limpieza de .pth con protección cada 2 temporadas (cálculo independiente) ---
        info = e.q("""WITH last AS (SELECT p.individual_id, max(s.number) n FROM season_participants p JOIN seasons s ON s.id=p.season_id GROUP BY 1)
                      SELECT i.id, i.color, i.model_file, i.model_pruned_at, l.n AS last_season,
                             EXISTS (SELECT 1 FROM evolution_plan ep JOIN seasons s2 ON s2.id=ep.season_id WHERE s2.number=l.n AND ep.individual_id=i.id) AS continues
                      FROM individuals i LEFT JOIN last l ON l.individual_id=i.id""")
        first_seasons = {}
        for r in e.q("SELECT s.number, p.individual_id FROM season_participants p JOIN seasons s ON s.id=p.season_id WHERE p.final_position=1"):
            first_seasons.setdefault(r["individual_id"], set()).add(r["number"])
        for r in e.q("SELECT number, league_champion_id l, cup_champion_id c, recopa_champion_id rc FROM seasons"):
            for k in ("l", "c", "rc"):
                first_seasons.setdefault(r[k], set()).add(r["number"])
        pruned = kept_protected = 0
        for r in info:
            if r["last_season"] is None:
                continue
            protected = any(s % 2 == 0 for s in first_seasons.get(r["id"], ()))
            should = (not r["continues"]) and r["last_season"] <= 3 and not protected
            exists = (e.models / r["model_file"]).exists()
            assert exists != should, (r["color"], r["last_season"], protected)
            assert (r["model_pruned_at"] is not None) == should
            pruned += should
            kept_protected += (not r["continues"]) and protected
        assert pruned > 0
        # --- compactación: temporadas 1-3 sin jugadas ni historial intermedio; la 4 intacta ---
        for num, compacted in ((1, True), (2, True), (3, True), (4, False)):
            sid = e.q("SELECT id FROM seasons WHERE number=%s", (num,))[0]["id"]
            nmoves = e.q("SELECT count(*) c FROM matches WHERE season_id=%s AND moves IS NOT NULL", (sid,))[0]["c"]
            mds = e.q("SELECT count(DISTINCT matchday) c FROM league_history WHERE season_id=%s", (sid,))[0]["c"]
            assert (nmoves == 0) == compacted and (mds == 1) == compacted and (compacted or mds == 38)
        shown = Display(False, io.StringIO()); reports.render_season(e.conn(), shown, 1)             # las vistas siguen funcionando
        league = [l for l in shown.stream.getvalue().split("Tabla de Liga")[1].split("--- Copa")[0].splitlines() if re.match(r"\s+\d+\s+#", l)]
        assert len(league) == 20
        # --- medición retroactiva de la temporada 3 (sin medición) ---
        db = e.db(); db.connect()
        try:
            assert benchmark.backfill(db, e.cfg(**cfg_kw), step=1, games=4, emit=lambda *_: None) == 1
        finally:
            db.close()
        assert e.q("SELECT count(*) c FROM season_benchmarks b JOIN seasons s ON s.id=b.season_id WHERE s.number=3")[0]["c"] == 2
        assert sorted(progress.load_curve(e.conn())) == [1, 2, 3, 4]
        # --- elegir campeones para el torneo maestro ---
        champs = ladder.pick_champions(e.conn(), e.models, step=None, max_n=20)
        assert 1 <= len(champs) <= 4 and len({c["color"] for c in champs}) == len(champs) and all(c["path"].exists() for c in champs)
        # --- espacio en disco ---
        cfg = e.cfg(**cfg_kw)
        d = disk.collect(e.conn(), cfg)
        assert sum(n for n, _ in d["cats"].values()) == len(list(e.models.glob("*.pth")))
        assert d["cats"]["borrables ya"][0] == 0 and d["db_bytes"] > 0 and d["orphans"] == []
        (e.models / "abcdef.pth").write_bytes(b"x" * 10); (e.models / "abcdef.json").write_text("{}")
        (e.models / "historical").mkdir(exist_ok=True); (e.models / "historical" / "huerfano.pth").write_bytes(b"y")
        orphans = disk.find_orphans(e.conn(), e.models)
        assert {p.name for p in orphans} == {"abcdef.pth", "abcdef.json", "huerfano.pth"}
        # los archivos de IA reales y los históricos de la base NO se consideran huérfanos
        assert not any(p.name == r["model_file"] for p in orphans for r in info)
        shown = Display(False, io.StringIO()); disk.render(shown, disk.collect(e.conn(), cfg), cfg)
        assert "HUÉRFANOS: 3" in shown.stream.getvalue() and "Base de datos" in shown.stream.getvalue()
        assert disk.delete_orphans(orphans, e.models) == 10 + 2 + 1
        assert disk.find_orphans(e.conn(), e.models) == []
    finally:
        e.close()


def test_prune_huerfanos_por_cli(env, monkeypatch, capsys, tmp_path):
    import os
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={env.schema}")
    monkeypatch.setenv("MODELS_DIR", str(env.models))
    env.trainer().run()
    (env.models / "112233.pth").write_bytes(b"z" * 5)
    assert main(["prune", "--orphans"]) == 0 and "Para borrarlos" in capsys.readouterr().out
    assert (env.models / "112233.pth").exists()                                           # sin --yes no borra
    assert main(["prune", "--orphans", "--yes"]) == 0 and "borrados" in capsys.readouterr().out
    assert not (env.models / "112233.pth").exists()
    assert main(["disk"]) == 0 and "ESPACIO EN DISCO" in capsys.readouterr().out


def test_cli_bench_progress_y_menu(reference_season, monkeypatch, capsys):
    e = reference_season
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={e.schema}")
    monkeypatch.setenv("MODELS_DIR", str(e.models))
    assert main(["bench", "1", "6", "seed=3"]) == 0
    out = capsys.readouterr().out
    assert "BENCHMARK" in out and "vs bot aleatorio" in out and "vs bot táctico" in out and "Referencias" in out
    assert main(["bench"]) == 2 and main(["bench", "1", "1"]) == 2 and main(["bench", "#010101"]) == 1
    assert main(["progress"]) == 0 and "PROGRESO" in capsys.readouterr().out
    assert main(["progress", "x"]) == 2 and main(["progress", "--backfill", "0"]) == 2
    assert main(["ladder", "1", "2", "3"]) == 2 and main(["ladder", "1", "99"]) == 2
    assert main(["ladder", "x"]) == 1 and "No existe ninguna IA estática" in capsys.readouterr().err        # "x" ahora es un nombre
    assert main(["ladder"]) == 1 and "al menos 3" in capsys.readouterr().out
    answers = iter(["8", "9 1 4", "10", "11", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["menu"]) == 0
    out = capsys.readouterr().out
    assert "PROGRESO" in out and "BENCHMARK" in out and "ESPACIO EN DISCO" in out