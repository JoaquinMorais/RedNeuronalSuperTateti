"""Temporadas con IA añadidas (21+ equipos), pelear entre IA y liga de prueba con aleatorias."""
import io
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import torch

from evolution import additions, play, rules, sandbox, schedule
from evolution import colors as C
from evolution.__main__ import main
from evolution.display import Display
from model import SuperTatetiNet
from tests.helpers import TEST_URL, Env


def _save_net(path: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    torch.save(SuperTatetiNet().state_dict(), path)
    return path


# ====================== formato de temporada (puro) ======================
@pytest.mark.parametrize("extra,sizes,mds,games", [
    (0, [5, 5, 5, 5], 38, 38), (1, [6, 5, 5, 5], 42, 40), (2, [6, 6, 5, 5], 42, 42),
    (3, [6, 6, 6, 5], 46, 44), (4, [6, 6, 6, 6], 46, 46), (6, [7, 7, 6, 6], 50, 50), (12, [8, 8, 8, 8], 62, 62)])
def test_formato_de_temporada(extra, sizes, mds, games):
    f = rules.season_format(20 + extra)
    assert f["group_sizes"] == sizes and f["league_matchdays"] == mds and f["games_per_team"] == games
    assert sum(sizes) == 20 + extra and f["group_eliminated"] == [g - 4 for g in sizes]


def test_formato_fuera_de_rango():
    for bad in (19, 33):
        with pytest.raises(ValueError):
            rules.season_format(bad)


@pytest.mark.parametrize("n", [20, 21, 22, 23, 24, 26])
def test_liga_con_n_equipos(n):
    fx = schedule.league_fixtures(n)
    f = rules.season_format(n)
    assert len(fx) == f["league_matchdays"] and sum(len(d) for d in fx) == n * (n - 1)
    per, pairs = Counter(), Counter()
    for day in fx:
        slots = [s for p in day for s in p]
        assert len(slots) == len(set(slots))                       # nadie juega dos veces en la misma fecha
        assert len(slots) == (n if n % 2 == 0 else n - 1)           # con n impar descansa exactamente una IA
        for x, o in day:
            per[x] += 1; per[o] += 1; pairs[(x, o)] += 1
    assert set(per.values()) == {f["games_per_team"]} and len(per) == n
    assert all(pairs[(o, x)] == 1 for x, o in pairs) and len(pairs) == n * (n - 1)   # ida y vuelta, inicio alternado


@pytest.mark.parametrize("size", [5, 6, 7, 8])
def test_grupos_de_cualquier_tamano(size):
    fx = schedule.group_fixtures(size)
    per = Counter()
    for _, rnd in fx:
        for x, o in rnd:
            per[x] += 1; per[o] += 1
    assert sum(len(r) for _, r in fx) == size * (size - 1) and set(per.values()) == {2 * (size - 1)}
    mds = [md for md, _ in fx]
    assert len(set(mds)) == len(mds) and max(mds) < 18             # siempre antes de los octavos
    if size <= 6:
        assert max(mds) <= 12                                       # como en la especificación original
    assert schedule.group_fixtures(5)[0][0] == 1 and [md for md, _ in schedule.group_fixtures(5)] == rules.GROUP_ROUND_MATCHDAYS


def test_sorteo_de_grupos_con_extras():
    pids = [f"#{i:06x}" for i in range(26)]
    g = schedule.draw_groups(pids, __import__("random").Random(1))
    assert [len(g[k]) for k in "ABCD"] == [7, 7, 6, 6] and sorted(sum(g.values(), [])) == sorted(pids)


def test_recompensa_de_liga_mas_alla_del_20():
    assert [rules.league_reward(p) for p in (1, 2, 20, 21, 26)] == [30, 24, -3, -3, -3]


def test_referencias_a_ia():
    assert sandbox.parse_ia_ref("12") == ("season", 12)
    assert sandbox.parse_ia_ref("#FFFFFF") == ("color", "#ffffff")
    assert sandbox.parse_ia_ref("a1b2c3") == ("color", "#a1b2c3")      # sin '#': la shell lo toma como comentario
    assert sandbox.parse_ia_ref("BocaJrs") == ("name", "BocaJrs") and sandbox.parse_ia_ref("Wall_E") == ("name", "Wall_E")
    for bad in ("#fff", "0", "#gggggg", "-3", "12ab%"):
        with pytest.raises(play.PlayError):
            sandbox.parse_ia_ref(bad)


# ====================== migración de una base creada con la versión anterior ======================
def test_migracion_de_base_antigua(env):
    from importlib import resources
    sql = resources.files("evolution").joinpath("schema.sql").read_text(encoding="utf-8")
    old = sql.split("-- Migración idempotente: fecha")[0]            # esquema original (slot 1..20, sin 'imported')
    env.admin.execute(f'SET search_path TO "{env.schema}"'); env.admin.execute(old)
    with pytest.raises(Exception):
        env.admin.execute("INSERT INTO individuals (color, origin_season, kind, model_file, architecture, architecture_sig) "
                          "VALUES ('#010203', 1, 'imported', 'x', 'a', 'b')")
    db = env.db(); db.connect(); db.init_schema(); db.init_schema()      # migrar (dos veces: idempotente)
    db.close()
    env.admin.execute("INSERT INTO individuals (color, origin_season, kind, model_file, architecture, architecture_sig) "
                      "VALUES ('#010203', 1, 'imported', 'x', 'a', 'b')")
    sid = env.admin.execute("INSERT INTO seasons (number, seed, n_participants, league_matchdays) VALUES (1, 1, 26, 50) RETURNING id").fetchone()["id"]
    iid = env.admin.execute("SELECT id FROM individuals").fetchone()["id"]
    env.admin.execute("INSERT INTO season_participants (season_id, individual_id, slot, cup_group) VALUES (%s,%s,26,'A')", (sid, iid))


# ====================== registrar IA para la próxima temporada ======================
def test_registrar_validar_y_cancelar(env, tmp_path):
    files = [_save_net(tmp_path / f"ia{i}.pth", i) for i in range(3)]
    db = env.db(); db.connect_with_retries(); db.init_schema()
    cfg = env.cfg()
    try:
        res = additions.register_additions(db, cfg, [str(files[0]), str(files[1]), str(files[2])])
        assert res["target"] == 1 and res["total_extra"] == 3 and len(res["added"]) == 3
        assert res["format"]["teams"] == 23 and res["format"]["group_sizes"] == [6, 6, 6, 5] and res["format"]["league_matchdays"] == 46
        rows = env.q("SELECT * FROM individuals ORDER BY id")
        assert len(rows) == 3 and all(r["kind"] == "imported" and r["parent_id"] is None and C.is_valid_hex(r["color"]) for r in rows)
        assert len({r["color"] for r in rows}) == 3
        for r in rows:                                                   # modelo guardado con el formato de siempre
            assert (env.models / r["model_file"]).exists() and r["model_sha256"]
            info = json.loads((env.models / r["model_file"]).with_suffix(".json").read_text())
            assert info["weights_sha256"] == r["model_sha256"] and info["kind"] == "imported"
        assert [p["color"] for p in additions.list_pending(env.conn())] == [r["color"] for r in rows]
        # validaciones
        with pytest.raises(additions.AdditionError, match="No se encuentra"):
            additions.register_additions(db, cfg, [str(tmp_path / "nada.pth")])
        (tmp_path / "malo.pth").write_bytes(b"no es un modelo")
        with pytest.raises(additions.AdditionError, match="no es un modelo válido"):
            additions.register_additions(db, cfg, [str(tmp_path / "malo.pth")])
        torch.save({"w": torch.zeros(2)}, tmp_path / "otra_red.pth")
        with pytest.raises(additions.AdditionError, match="no es un modelo válido"):
            additions.register_additions(db, cfg, [str(tmp_path / "otra_red.pth")])
        with pytest.raises(additions.AdditionError, match="ya está añadida"):
            additions.register_additions(db, cfg, [str(files[0])])
        with pytest.raises(additions.AdditionError, match="repetido"):
            additions.register_additions(db, cfg, [str(files[0]), str(files[0])])
        many = [_save_net(tmp_path / f"m{i}.pth", 100 + i) for i in range(10)]
        with pytest.raises(additions.AdditionError, match="como máximo 12"):
            additions.register_additions(db, cfg, [str(m) for m in many])
        # cancelar una
        victim = rows[1]
        out = additions.cancel_addition(db, cfg, victim["color"])
        assert out["season"] == 1 and not (env.models / victim["model_file"]).exists()
        assert len(env.q("SELECT * FROM individuals")) == 2 and len(additions.list_pending(env.conn())) == 2
        with pytest.raises(additions.AdditionError, match="no está entre"):
            additions.cancel_addition(db, cfg, victim["color"])
    finally:
        db.close()


def test_ruta_relativa_a_models(env, tmp_path):
    env.models.mkdir(parents=True)
    _save_net(env.models / "mi_ia.pth", 5)
    db = env.db(); db.connect_with_retries(); db.init_schema()
    try:
        res = additions.register_additions(db, env.cfg(), ["mi_ia.pth"])      # solo el nombre: se busca en models/
        assert res["total_extra"] == 1 and res["format"]["group_sizes"] == [6, 5, 5, 5] and res["format"]["league_matchdays"] == 42
    finally:
        db.close()


# ====================== temporadas jugadas con IA añadidas ======================
def _check_extra_season(e, sid_num, n):
    f = rules.season_format(n)
    s = e.q("SELECT * FROM seasons WHERE number=%s", (sid_num,))[0]
    sid = s["id"]
    assert s["n_participants"] == n and s["league_matchdays"] == f["league_matchdays"] and s["status"] == "completed"
    assert e.q("SELECT count(*) c FROM season_participants WHERE season_id=%s", (sid,))[0]["c"] == n
    # Liga: n(n-1) partidas, cada IA juega 2(n-1), una por fecha como máximo
    assert e.q("SELECT count(*) c FROM matches WHERE season_id=%s AND competition='league' AND status='finished'", (sid,))[0]["c"] == n * (n - 1)
    per = e.q("""SELECT count(*) c FROM (SELECT p FROM (SELECT x_individual_id p FROM matches WHERE season_id=%s AND competition='league'
                 UNION ALL SELECT o_individual_id FROM matches WHERE season_id=%s AND competition='league') t GROUP BY p
                 HAVING count(*) = %s) z""", (sid, sid, f["games_per_team"]))[0]["c"]
    assert per == n
    assert e.q("SELECT max(matchday) m FROM matches WHERE season_id=%s AND competition='league'", (sid,))[0]["m"] == f["league_matchdays"]
    assert e.q("SELECT count(*) c FROM (SELECT matchday, p FROM (SELECT matchday, x_individual_id p FROM matches WHERE season_id=%s AND competition='league' "
               "UNION ALL SELECT matchday, o_individual_id FROM matches WHERE season_id=%s AND competition='league') t GROUP BY 1,2 HAVING count(*)>1) z",
               (sid, sid))[0]["c"] == 0
    # Copa: grupos de tamaño variable, clasifican 4 por grupo, 16 en el cuadro
    for g, size in zip("ABCD", f["group_sizes"]):
        assert e.q("SELECT count(*) c FROM season_participants WHERE season_id=%s AND cup_group=%s", (sid, g))[0]["c"] == size
        assert e.q("SELECT count(*) c FROM matches WHERE season_id=%s AND competition='cup_group' AND group_name=%s", (sid, g))[0]["c"] == size * (size - 1)
        q = e.q("SELECT qualified FROM cup_group_table WHERE season_id=%s AND group_name=%s ORDER BY position", (sid, g))
        assert [r["qualified"] for r in q] == [True] * 4 + [False] * (size - 4)
    assert e.q("SELECT count(*) c FROM season_participants WHERE season_id=%s AND cup_stage='group'", (sid,))[0]["c"] == n - 16
    assert e.q("SELECT count(*) c FROM cup_ties WHERE season_id=%s AND status='finished'", (sid,))[0]["c"] == 15
    # Recopa en la fecha siguiente a la última de Liga; puntuaciones y evolución
    assert {r["m"] for r in e.q("SELECT matchday m FROM matches WHERE season_id=%s AND competition='recopa'", (sid,))} <= {f["league_matchdays"] + 1}
    assert sorted(r["p"] for r in e.q("SELECT final_position p FROM season_participants WHERE season_id=%s", (sid,))) == list(range(1, n + 1))
    for r in e.q("SELECT league_position p, league_score s FROM season_participants WHERE season_id=%s", (sid,)):
        assert r["s"] == rules.league_reward(r["p"])
    for r in e.q("SELECT * FROM season_participants WHERE season_id=%s", (sid,)):
        assert r["total_score"] == r["league_score"] + r["cup_score"] + r["recopa_score"]
    plan = e.q("SELECT role FROM evolution_plan WHERE season_id=%s ORDER BY position", (sid,))
    assert [p["role"] for p in plan] == ["survivor"] * 7 + ["child"] * 13          # la siguiente generación vuelve a ser de 20


def test_temporada_con_3_ia_anadidas_y_la_siguiente_vuelve_a_20(tmp_path):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    e = Env(tmp_path)
    try:
        files = [_save_net(tmp_path / f"x{i}.pth", 40 + i) for i in range(3)]
        db = e.db(); db.connect_with_retries(); db.init_schema()
        additions.register_additions(db, e.cfg(), [str(f) for f in files]); db.close()
        e.trainer(max_seasons=2).run()
        assert e.q("SELECT count(*) c FROM seasons WHERE status='completed'")[0]["c"] == 2
        _check_extra_season(e, 1, 23)
        _check_extra_season(e, 2, 20)                                               # la temporada siguiente: 20 equipos
        txt = e.out.getvalue()
        assert "3 redes añadidas" in txt and "grupos de 6/6/6/5" in txt and "FECHA 46 / 46" in txt
        assert "FECHA 38 / 38" in txt                                               # y la 2 es la normal
        # las importadas aparecen en la temporada y conservan identidad/archivo
        imp = e.q("SELECT color FROM individuals WHERE kind='imported'")
        assert len(imp) == 3
    finally:
        e.close()


def test_anadir_una_ia_a_la_proxima_temporada_mientras_hay_otra_creada(tmp_path):
    """Con la temporada 2 ya creada se añade una IA: va a la 3 (21 equipos); la 4 vuelve a 20."""
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    e = Env(tmp_path)
    try:
        e.trainer(max_seasons=1).run()                                               # temporada 1 normal
        e.trainer(max_seasons=0, stop=_StopAfterCreate()).run()                       # crea la 2 y se detiene antes de jugar
        assert e.q("SELECT number, status FROM seasons ORDER BY number")[-1] == {"number": 2, "status": "in_progress"}
        db = e.db(); db.connect(); db.init_schema()
        res = additions.register_additions(db, e.cfg(), [str(_save_net(tmp_path / "nueva.pth", 77))]); db.close()
        assert res["target"] == 3 and res["format"]["teams"] == 21
        e.trainer(max_seasons=2).run()                                               # termina la 2 y juega la 3
        assert e.q("SELECT n_participants n FROM seasons ORDER BY number") == [{"n": 20}, {"n": 20}, {"n": 21}]
        _check_extra_season(e, 3, 21)
        s3 = e.q("SELECT draw_meta FROM seasons WHERE number=3")[0]["draw_meta"]["format"]
        assert s3["group_sizes"] == [6, 5, 5, 5] and s3["league_matchdays"] == 42
        e.trainer(max_seasons=1).run()
        assert e.q("SELECT n_participants n FROM seasons ORDER BY number")[-1]["n"] == 20      # temporada 4: otra vez 20
        assert e.q("SELECT count(*) c FROM season_additions")[0]["c"] == 1
    finally:
        e.close()


class _StopAfterCreate:
    """Se activa al segundo chequeo: deja crear la temporada y corta antes de jugar."""
    def __init__(self):
        self.calls = 0

    def set(self, *a):
        pass

    def is_set(self):
        self.calls += 1
        return self.calls > 1


# ====================== pelear y liga de prueba ======================
def test_resolver_ia_por_temporada_y_por_color(reference_season):
    e = reference_season
    conn, models = e.conn(), e.models
    best = e.q("SELECT i.color FROM season_participants p JOIN individuals i ON i.id=p.individual_id WHERE p.final_position=1")[0]["color"]
    other = e.q("SELECT i.color FROM season_participants p JOIN individuals i ON i.id=p.individual_id WHERE p.final_position=2")[0]["color"]
    assert sandbox.resolve_ia(conn, models, "1")["color"] == best
    assert sandbox.resolve_ia(conn, models, other)["color"] == other
    assert sandbox.resolve_ia(conn, models, other.lstrip("#"))["color"] == other
    assert sandbox.resolve_ia(conn, models, best)["tag"].startswith(best)
    with pytest.raises(play.PlayError, match="No existe ninguna IA"):
        sandbox.resolve_ia(conn, models, "#010101")
    with pytest.raises(play.PlayError, match="No existe la temporada"):
        sandbox.resolve_ia(conn, models, "9")
    with pytest.raises(play.PlayError, match="la misma IA"):
        sandbox.resolve_many(conn, models, ["1", best])
    # una IA cuyo .pth ya se borró: error explicativo
    victim = e.q("SELECT color, model_file FROM individuals WHERE color=%s", (other,))[0]
    (models / victim["model_file"]).rename(models / "tmp_hidden")
    try:
        with pytest.raises(play.PlayError, match="ya no tiene su modelo guardado"):
            sandbox.resolve_ia(conn, models, other)
    finally:
        (models / "tmp_hidden").rename(models / victim["model_file"])


def test_pelea_ida_y_vuelta(reference_season):
    e = reference_season
    ias = sandbox.resolve_many(e.conn(), e.models, ["1", e.q(
        "SELECT i.color FROM season_participants p JOIN individuals i ON i.id=p.individual_id WHERE p.final_position=2")[0]["color"]])
    a, b = ias
    res = sandbox.fight(a, b)
    g1, g2 = res["games"]
    assert (g1["x"], g1["o"]) == (a["color"], b["color"]) and (g2["x"], g2["o"]) == (b["color"], a["color"])   # inicio alternado
    expect = {a["color"]: 0, b["color"]: 0}
    for g in res["games"]:
        expect[g["x"]] += rules.match_points(g["result"], "x"); expect[g["o"]] += rules.match_points(g["result"], "o")
    assert res["points"] == expect
    top = max(expect, key=expect.get)
    assert res["winner"] == (None if expect[a["color"]] == expect[b["color"]] else top)
    again = sandbox.fight(a, b)                                                         # determinista: misma pelea, mismo resultado
    assert [(g["result"], g["moves"]) for g in again["games"]] == [(g["result"], g["moves"]) for g in res["games"]]
    out = Display(False, io.StringIO()); sandbox.render_fight(out, a, b, res)
    assert "Partida 1" in out.stream.getvalue() and "Resultado" in out.stream.getvalue()


def test_liga_de_prueba_con_random(reference_season):
    e = reference_season
    before = e.q("SELECT count(*) c FROM matches")[0]["c"]
    tested = sandbox.resolve_many(e.conn(), e.models, ["1"])
    r1 = sandbox.run_random_league(tested, seed=11)
    assert len(r1["order"]) == 20 == len(set(r1["order"])) and len(r1["random"]) == 19 and r1["tested"] == [tested[0]["color"]]
    assert all(r1["table"][c].played == 38 for c in r1["order"]) and len(r1["games"]) == 380
    s = r1["summary"][tested[0]["color"]]
    assert sum(s["vs_random"]) == 38 and sum(s["vs_tested"]) == 0 and 1 <= s["position"] <= 20
    assert e.q("SELECT count(*) c FROM matches")[0]["c"] == before                       # no se guarda nada en la BD
    r2 = sandbox.run_random_league(tested, seed=11)
    assert (r2["order"], r2["random"]) == (r1["order"], r1["random"])                    # misma semilla, mismo resultado
    r3 = sandbox.run_random_league(tested, seed=12)
    assert r3["random"] != r1["random"]
    out = Display(False, io.StringIO()); sandbox.render_random_league(out, r1, tested)
    txt = out.stream.getvalue()
    assert "◄ probada" in txt and txt.count("random") >= 19 and "Resumen de las IA probadas" in txt


def test_liga_de_prueba_con_varias_ia_y_limites(tmp_path):
    nets = []
    for i in range(3):
        p = _save_net(tmp_path / f"t{i}.pth", 200 + i)
        from evolution import modelstore
        sd = modelstore.load_state_dict(p)
        nets.append({"label": f"t{i}", "color": f"#0{i}0{i}0{i}", "tag": f"#0{i}0{i}0{i}", "path": p, "sha256": modelstore.weights_hash(sd)})
    r = sandbox.run_random_league(nets, seed=3)
    assert len(r["random"]) == 17
    assert all(sum(r["summary"][c]["vs_tested"]) == 4 for c in r["tested"])             # 2 partidas contra cada una de las otras 2
    assert all(sum(r["summary"][c]["vs_random"]) == 34 for c in r["tested"])             # 17 random x 2
    with pytest.raises(play.PlayError):
        sandbox.run_random_league([], seed=1)
    with pytest.raises(play.PlayError):
        sandbox.run_random_league(nets * 8, seed=1)                                      # más de 20


# ====================== CLI y menú ======================
def test_cli_y_menu_nuevas_opciones(reference_season, monkeypatch, capsys):
    e = reference_season
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={e.schema}")
    monkeypatch.setenv("MODELS_DIR", str(e.models))
    other = e.q("SELECT i.color FROM season_participants p JOIN individuals i ON i.id=p.individual_id WHERE p.final_position=2")[0]["color"]
    assert main(["fight", "1", other.lstrip("#")]) == 0 and "PELEA" in capsys.readouterr().out
    assert main(["fight", "1"]) == 2
    assert main(["fight", "1", "1"]) == 1 and "la misma IA" in capsys.readouterr().err
    assert main(["fight", "1", "#010101"]) == 1 and "No existe ninguna IA" in capsys.readouterr().err
    assert main(["testleague"]) == 2 and main(["testleague", "1", "seed=x"]) == 2
    assert main(["addai"]) == 2
    answers = iter([f"5 1 {other}", "5 1", "6 1 seed=4", "7 --list", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    # "5 1" sin segunda IA muestra el uso; el resto ejecuta
    assert main(["menu"]) == 0
    out = capsys.readouterr()
    assert "PELEA" in out.out and "LIGA DE PRUEBA" in out.out and "Semilla: 4" in out.out and "No hay IA añadidas pendientes" in out.out
    assert "Uso: fight A B" in out.err


def test_cli_addai_y_menu(env, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={env.schema}")
    monkeypatch.setenv("MODELS_DIR", str(env.models))
    f1, f2 = _save_net(tmp_path / "a.pth", 1), _save_net(tmp_path / "b dir.pth", 2)
    assert main(["addai", str(f1)]) == 0                                    # base vacía: se migra sola
    out = capsys.readouterr().out
    assert "La temporada 1 tendrá 21 equipos" in out and "42 fechas" in out and "40 partidas por IA" in out and "grupos de Copa de 6/5/5/5" in out
    assert main(["addai", "--list"]) == 0 and "temporada 1" in capsys.readouterr().out
    answers = iter([f'7 "{f2}"', "0"])                                      # ruta con espacios entre comillas
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["menu"]) == 0
    assert "22 equipos" in capsys.readouterr().out
    pend = additions.list_pending(env.conn())
    assert len(pend) == 2
    assert main(["addai", "--cancel", pend[0]["color"]]) == 0 and "Quitada" in capsys.readouterr().out
    assert main(["addai", "--cancel", "#010101"]) == 1
    assert main(["addai", str(tmp_path / "no_existe.pth")]) == 1 and "No se encuentra" in capsys.readouterr().err