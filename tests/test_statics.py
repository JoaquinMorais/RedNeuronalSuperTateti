"""Bots estáticos: registro, torneo, evolución, nombres, menú y herramientas de prueba."""
import io
import json
import re
from pathlib import Path

import pytest
import torch

from evolution import additions, benchmark, disk, engine, ladder, play, prune, reports, rules, sandbox, statics
from evolution.__main__ import main
from evolution.display import Display
from models_statics.base import StaticBot, split_name
from model import SuperTatetiNet
from tests.helpers import ROOT, TEST_URL, Env

REPO_STATICS = ROOT / "models_statics"

FUERTE = '''
from models_statics.base import StaticBot
from evolution.bots import TacticalBot


class Fuerte(StaticBot):
    NAME = "BocaJrs"
    COLOR1 = "#0033aa"
    COLOR2 = "#ffcc00"

    def choose(self, game):
        return TacticalBot(self.rng).choose(game)
'''
FLOJO = '''
from models_statics.base import StaticBot


class Flojo(StaticBot):
    NAME = "Flojo"
    COLOR1 = "#aa0033"
    COLOR2 = "#cccccc"

    def choose(self, game):
        mask = game.get_valid_moves()
        i = next(k for k in range(81) if mask[k] == 1)
        return i // 9, i % 9
'''


def make_dir(path: Path, **bots) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for slug, src in bots.items():
        (path / f"{slug}.py").write_text(src)
    return path


def bot_src(name, c1, c2, cls="Bot"):
    return (f'from models_statics.base import StaticBot\n\n\nclass {cls}(StaticBot):\n    NAME = "{name}"\n'
            f'    COLOR1 = "{c1}"\n    COLOR2 = "{c2}"\n\n    def choose(self, game):\n'
            f'        i = next(k for k in range(81) if game.get_valid_moves()[k] == 1)\n        return i // 9, i % 9\n')


def _net(seed):
    torch.manual_seed(seed)
    n = SuperTatetiNet().eval()
    for p in n.parameters():
        p.requires_grad_(False)
    return n


# ====================== nombres y colores (sin base de datos) ======================
def test_el_nombre_se_parte_de_a_dos_letras_alternando_colores():
    assert split_name("WALL-E", "#e8b923", "#3b6fb6") == [("WA", "#e8b923"), ("LL", "#3b6fb6"), ("-E", "#e8b923")]
    assert [t for t, _ in split_name("BocaJrs", "#1", "#2")] == ["Bo", "ca", "Jr", "s"]


def test_se_encuentran_por_nombre_sin_importar_mayusculas_ni_guiones():
    for text in ("WALL-E", "wall-e", "Wall_E", "wall e", "WALLE"):
        slug, cls = statics.find(REPO_STATICS, text)
        assert slug == "wall-e" and cls.NAME == "WALL-E"
    with pytest.raises(statics.StaticError, match="No hay ningún bot estático llamado 'nadie'.*WALL-E"):
        statics.find(REPO_STATICS, "nadie")


def test_dos_bots_no_pueden_repetir_nombre_ni_color(tmp_path):
    d = make_dir(tmp_path / "a", uno=bot_src("Uno", "#111111", "#222222"), dos=bot_src("UNO", "#333333", "#444444"))
    with pytest.raises(statics.StaticError, match="nombre"):
        statics.check_unique(d)
    d = make_dir(tmp_path / "b", uno=bot_src("Uno", "#111111", "#222222"), dos=bot_src("Dos", "#111111", "#444444"))
    with pytest.raises(statics.StaticError, match="COLOR1"):
        statics.check_unique(d)
    statics.check_unique(REPO_STATICS)


def test_un_bot_con_colores_invalidos_o_iguales_se_rechaza(tmp_path):
    d = make_dir(tmp_path / "c", malo=bot_src("Malo", "#fff", "#222222"), igual=bot_src("Igual", "#111111", "#111111"),
                 bueno=bot_src("Bueno", "#111111", "#222222"))
    assert set(statics.available(d)) == {"bueno"}                          # los inválidos se ignoran con su motivo
    errs = statics.broken(d)
    assert set(errs) == {"malo", "igual"} and all("ValueError" in e for e in errs.values())
    with pytest.raises(statics.StaticError, match="tiene errores"):
        statics.load_one(d, "malo")


def test_presentacion_por_nombre_en_dos_colores():
    d = Display(True)
    d.statics = {"#e8b923": {"name": "WALL-E", "color2": "#3b6fb6", "slug": "wall-e"}}
    plain = lambda t: re.sub(r"\x1b\[[0-9;]*m", "", t)
    txt = d.ia("#e8b923")
    assert plain(txt) == "■ WALL-E"
    segs = re.findall(r"\x1b\[38;2;(\d+;\d+;\d+)m([^\x1b]*)", txt)
    assert [t for _, t in segs] == ["■ ", "WA", "LL", "-E"]
    assert [c for c, t in segs if t in ("WA", "LL", "-E")] == ["232;185;35", "59;111;182", "232;185;35"]
    d.titles = {"#e8b923": {"L": [1, 2], "C": [], "RC": [3]}}
    assert plain(d.ia("#e8b923")) == "■ WALL-E:02L:00C:01RC" and d.tag("#e8b923", badge=False) == "WALL-E"
    assert d.id_width() == len("WALL-E") + len(":02L:00C:01RC") if len("WALL-E") > 7 else d.id_width() == 7 + len(":02L:00C:01RC")
    assert len(plain(d.cell("#e8b923", 30))) == 30                       # el relleno cuenta solo lo visible
    assert plain(d.paint_tags("1A WALL-E:02L:00C:01RC ─ #112233")) == "1A WALL-E:02L:00C:01RC ─ #112233"
    off = Display(False); off.statics = d.statics
    assert off.ia("#e8b923") == "WALL-E" and off.ia("#112233") == "#112233"       # sin colores: texto plano legible


# ====================== quién sigue (regla pura) ======================
def _rows(positions_types):
    return [{"id": i, "type": t, "position": p} for i, (p, t) in enumerate(positions_types, start=1)]


def test_regla_de_continuidad_de_los_estaticos():
    nets = [(p, "neuronal") for p in range(1, 22)]
    # un estático: zona = 7 + 1 = 8
    for pos, stays in ((1, True), (8, True), (9, False), (12, False)):
        pl = [(p if p < pos else p + 1, t) for p, t in nets] + [(pos, "estatico")]
        r = rules.select_next_generation(_rows(pl))
        assert (len(r["statics_stay"]) == 1) == stays and r["zone"] == 8 and len(r["survivors"]) == 7
    # dos estáticos: zona = 9, cada uno se evalúa por separado
    pl = [(p, "neuronal") for p in range(1, 24) if p not in (3, 12)] + [(3, "estatico"), (12, "estatico")]
    r = rules.select_next_generation(_rows(pl))
    assert r["zone"] == 9 and len(r["statics_stay"]) == 1 and len(r["statics_leave"]) == 1
    # los estáticos no ocupan cupos de redes: las 7 mejores REDES se conservan aunque un estático sea 1.º
    pl = [(1, "estatico")] + [(p, "neuronal") for p in range(2, 23)]
    r = rules.select_next_generation(_rows(pl))
    assert r["survivors"] == [2, 3, 4, 5, 6, 7, 8] and r["trained_ranked"][:2] == [2, 3]
    # sin estáticos: como siempre
    r = rules.select_next_generation(_rows([(p, "neuronal") for p in range(1, 21)]))
    assert r["zone"] == 7 and r["statics_stay"] == [] and r["survivors"] == list(range(1, 8))


# ====================== motor ======================
def test_un_bot_y_una_red_juegan_una_partida_legal_y_reproducible():
    cls = statics.available(REPO_STATICS)["wall-e"]
    net = _net(3)
    outs = []
    for _ in range(2):
        bot = cls(seed=11)
        outs.append(engine.play_match(bot, net).moves)
    assert outs[0] == outs[1]                                              # misma semilla, misma partida
    firsts = {cls(seed=s).choose(__import__("game").SuperTateti()) for s in range(12)}
    assert len(firsts) >= 2                                                # el azar controlado varía la apertura
    o = engine.play_match(net, cls(seed=5))
    assert o.result in ("x", "o", "draw") and o.num_moves == len(o.moves.split(","))


def test_wall_e_es_fuerte_contra_el_azar():
    cls = statics.available(REPO_STATICS)["wall-e"]
    res = benchmark.evaluate_net(cls(seed=1), "random", 10, seed=4)
    assert res["w"] >= 8 and res["x"][0] + res["o"][0] == res["w"]


# ====================== registro en la base de datos ======================
def _cfg_env(tmp_path, **bots):
    d = make_dir(tmp_path / "bots", **bots)
    return Env(tmp_path, statics_dir=d), d


def test_registrar_un_estatico_guarda_su_identidad_y_tipo(tmp_path):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    e, d = _cfg_env(tmp_path, fuerte=FUERTE, flojo=FLOJO)
    try:
        db = e.db(); db.connect_with_retries(); db.init_schema()
        res = additions.register_additions(db, e.cfg(), ["bocajrs"])          # por nombre, sin importar mayúsculas
        assert res["target"] == 1 and res["total_extra"] == 1 and res["format"]["teams"] == 21
        a = res["added"][0]
        assert a["type"] == "estatico" and a["name"] == "BocaJrs" and a["color"] == "#0033aa" and a["color2"] == "#ffcc00"
        row = e.q("SELECT * FROM individuals")[0]
        assert row["ia_type"] == "estatico" and row["kind"] == "static" and row["color"] == "#0033aa"
        assert (row["static_name"], row["static_color2"], row["static_slug"]) == ("BocaJrs", "#ffcc00", "fuerte")
        assert row["model_file"] == "static:fuerte" and row["model_sha256"] == statics.code_hash(d, "fuerte")
        assert not list(e.models.glob("*.pth"))                                # sin archivo de pesos
        pend = additions.list_pending(e.conn())
        assert [(p["static_name"], p["ia_type"], p["season_number"]) for p in pend] == [("BocaJrs", "estatico", 1)]
        # errores claros
        with pytest.raises(additions.AdditionError, match="No hay ningún bot estático llamado 'nadie'.*BocaJrs"):
            additions.register_additions(db, e.cfg(), ["nadie"])
        with pytest.raises(additions.AdditionError):
            additions.register_additions(db, e.cfg(), ["BocaJrs", "fuerte"])   # repetido en el pedido
        # se puede mezclar con redes y con otros estáticos en el mismo pedido
        torch.save(_net(5).state_dict(), tmp_path / "red.pth")
        res = additions.register_additions(db, e.cfg(), [str(tmp_path / "red.pth"), "Flojo"])
        assert res["total_extra"] == 3 and res["format"]["teams"] == 23
        assert {r["ia_type"] for r in e.q("SELECT ia_type FROM individuals")} == {"estatico", "neuronal"}
        # quitar por nombre: el estático sin historia desaparece del todo
        out = additions.cancel_addition(db, e.cfg(), "flojo")
        assert out["type"] == "estatico" and out["name"] == "Flojo"
        assert [r["static_name"] for r in e.q("SELECT static_name FROM individuals WHERE ia_type='estatico'")] == ["BocaJrs"]
        with pytest.raises(additions.AdditionError, match="no está entre"):
            additions.cancel_addition(db, e.cfg(), "Flojo")
        db.close()
    finally:
        e.close()


def test_conflictos_de_color_y_limite(tmp_path):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    e, d = _cfg_env(tmp_path)
    try:
        db = e.db(); db.connect_with_retries(); db.init_schema()
        torch.save(_net(1).state_dict(), tmp_path / "red.pth")
        net_color = additions.register_additions(db, e.cfg(), [str(tmp_path / "red.pth")])["added"][0]["color"]
        make_dir(d, choque=bot_src("Choque", net_color, "#123456"))
        with pytest.raises(additions.AdditionError, match="identidad de una red neuronal"):
            additions.register_additions(db, e.cfg(), ["Choque"])
        make_dir(d, **{f"b{i}": bot_src(f"Bot{i}", f"#{i + 16:02x}{i + 16:02x}00", "#ffffff", f"B{i}") for i in range(12)})
        with pytest.raises(additions.AdditionError, match="como máximo 12"):
            additions.register_additions(db, e.cfg(), [f"Bot{i}" for i in range(12)])   # 12 + 1 red añadida > 12
        db.close()
    finally:
        e.close()


def test_migracion_de_base_anterior_a_los_estaticos(env):
    from importlib import resources
    sql = resources.files("evolution").joinpath("schema.sql").read_text(encoding="utf-8")
    old = sql.split("-- ===== Tipos de IA")[0]
    env.admin.execute(f'SET search_path TO "{env.schema}"'); env.admin.execute(old)
    env.admin.execute("ALTER TABLE individuals ADD CONSTRAINT individuals_kind_values "      # como la dejaba la versión anterior
                      "CHECK (kind IN ('founder_seed','founder_random','mutant','imported'))")
    with pytest.raises(Exception):
        env.admin.execute("INSERT INTO individuals (color, origin_season, kind, model_file, architecture, architecture_sig) "
                          "VALUES ('#010203', 1, 'static', 'x', 'a', 'b')")
    db = env.db(); db.connect(); db.init_schema(); db.init_schema(); db.close()       # migrar (idempotente)
    env.admin.execute("INSERT INTO individuals (color, origin_season, kind, model_file, architecture, architecture_sig, ia_type, static_name) "
                      "VALUES ('#010203', 1, 'static', 'static:x', 'static', 'static', 'estatico', 'X')")
    assert env.q("SELECT ia_type FROM individuals")[0]["ia_type"] == "estatico"
    sid = env.admin.execute("INSERT INTO seasons (number, seed) VALUES (1, 1) RETURNING id").fetchone()["id"]
    iid = env.admin.execute("SELECT id FROM individuals").fetchone()["id"]
    env.admin.execute("INSERT INTO evolution_plan (season_id, position, role, individual_id) VALUES (%s,25,'static',%s)", (sid, iid))
    for n in (2, 3):                                                                   # el mismo estático en temporadas distintas
        env.admin.execute("INSERT INTO season_additions (season_number, individual_id, source) VALUES (%s,%s,'static:x')", (n, iid))
    with pytest.raises(Exception):
        env.admin.execute("INSERT INTO individuals (color, origin_season, kind, model_file, architecture, architecture_sig, ia_type, static_name) "
                          "VALUES ('#010204', 1, 'static', 'static:y', 'static', 'static', 'estatico', 'x')")   # nombre repetido


# ====================== dos temporadas con dos estáticos (BocaJrs fuerte, Flojo débil) ======================
@pytest.fixture(scope="module")
def static_season(tmp_path_factory):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    tmp = tmp_path_factory.mktemp("st")
    d = make_dir(tmp / "bots", fuerte=FUERTE, flojo=FLOJO)
    e = Env(tmp, statics_dir=d)
    db = e.db(); db.connect_with_retries(); db.init_schema()
    additions.register_additions(db, e.cfg(), ["BocaJrs", "Flojo"]); db.close()
    e.trainer(max_seasons=2, bench_every=1, bench_games=4, bench_pop_games=1).run()
    yield e
    e.close()


def _participants(e, number):
    return e.q("""SELECT i.id, i.color, i.ia_type, i.static_name, p.final_position FROM season_participants p
                  JOIN individuals i ON i.id=p.individual_id JOIN seasons s ON s.id=p.season_id
                  WHERE s.number=%s ORDER BY p.final_position""", (number,))


def test_evolucion_con_estaticos(static_season):
    e = static_season
    s1 = e.q("SELECT * FROM seasons WHERE number=1")[0]
    parts = _participants(e, 1)
    assert s1["n_participants"] == 22 and len(parts) == 22 and s1["league_matchdays"] == 42
    assert sorted(p["static_name"] for p in parts if p["ia_type"] == "estatico") == ["BocaJrs", "Flojo"]
    gen = rules.select_next_generation([{"id": p["id"], "type": p["ia_type"], "position": p["final_position"]} for p in parts])
    plan = e.q("SELECT * FROM evolution_plan WHERE season_id=%s ORDER BY position", (s1["id"],))
    roles = [p["role"] for p in plan]
    assert roles[:7] == ["survivor"] * 7 and roles[7:20] == ["child"] * 13 and set(roles[20:]) <= {"static"}
    assert [p["individual_id"] for p in plan if p["role"] == "survivor"] == gen["survivors"]           # las 7 mejores REDES
    assert sorted(p["individual_id"] for p in plan if p["role"] == "static") == sorted(gen["statics_stay"])
    assert gen["zone"] == 9
    # hijos: solo de redes y según el puesto ENTRE REDES (5, 3, 2, 2, 1)
    kids = {}
    for p in plan:
        if p["role"] == "child":
            kids[p["parent_id"]] = kids.get(p["parent_id"], 0) + 1
    assert [kids.get(i, 0) for i in gen["trained_ranked"]] == [5, 3, 2, 2, 1] + [0] * (len(gen["trained_ranked"]) - 5)
    assert e.q("SELECT count(*) c FROM individuals i JOIN individuals p ON p.id=i.parent_id WHERE p.ia_type='estatico'")[0]["c"] == 0
    # la temporada 2 = plan (20 redes + los estáticos que siguen)
    s2 = e.q("SELECT * FROM seasons WHERE number=2")[0]
    assert {p["id"] for p in _participants(e, 2)} == {p["individual_id"] for p in plan}
    assert s2["n_participants"] == 20 + len(gen["statics_stay"]) and s2["status"] == "completed"
    # los estáticos no se mutan ni dejan archivos de pesos
    assert not [f for f in e.models.glob("*.pth") if f.stem in ("0033aa", "aa0033")]


def test_el_mejor_historico_y_la_medicion_son_de_redes(static_season):
    e = static_season
    assert {r["ia_type"] for r in e.q("SELECT i.ia_type FROM historical_best h JOIN individuals i ON i.id=h.individual_id")} == {"neuronal"}
    top = e.q("SELECT i.ia_type, b.games FROM season_benchmarks b JOIN individuals i ON i.id=b.individual_id WHERE b.subject='top1'")
    assert top and {r["ia_type"] for r in top} == {"neuronal"}
    pop = e.q("SELECT b.games FROM season_benchmarks b JOIN seasons s ON s.id=b.season_id WHERE s.number=1 AND b.subject='population'")
    assert {r["games"] for r in pop} == {20}                                     # 22 equipos - 2 estáticos = 20 redes x 1 partida


def test_salida_del_entrenamiento_con_estaticos(static_season):
    txt = static_season.out.getvalue()
    assert "2 estáticos: BocaJrs, Flojo" in txt and "22" in txt
    assert "Estáticos que siguen" in txt or "Estáticos que salen" in txt
    assert "#0033aa" not in txt and "#aa0033" not in txt                          # nunca se muestran por su color


def test_vista_de_temporada_con_estaticos(static_season):
    e = static_season
    out = Display(False, io.StringIO())
    reports.render_season(e.conn(), out, 1)
    txt = out.stream.getvalue()
    assert "BocaJrs" in txt and "Flojo" in txt and "estático" in txt and "#0033aa" not in txt
    annual = txt.split("Tabla anual")[1]
    rows = {m.group(1): l for l in annual.splitlines() for m in [re.match(r"\s+\d+\s+(BocaJrs|Flojo)\S*", l)] if m}
    assert set(rows) == {"BocaJrs", "Flojo"} and all(("sigue (estático" in l) or l.rstrip().endswith("sale") for l in rows.values())
    assert "estáticos que siguen" in annual
    league = [l for l in txt.split("Tabla de Liga")[1].split("--- Copa")[0].splitlines() if re.match(r"\s+\d+\s+\S", l)]
    assert len(league) == 22


def test_los_estaticos_ganan_titulos_que_se_ven_en_el_menu_3(static_season):
    e = static_season
    conn = e.conn()
    flojo = e.q("SELECT id FROM individuals WHERE static_name='Flojo'")[0]["id"]
    before = e.q("SELECT league_champion_id l FROM seasons WHERE number=1")[0]["l"]
    try:
        e.admin.execute("UPDATE seasons SET league_champion_id=%s WHERE number=1", (flojo,))
        out = Display(False, io.StringIO())
        reports.render_champions(conn, out, 30)
        txt = out.stream.getvalue()
        assert re.search(r"\s+\d+\s+Flojo\s+estático\s+1\s+\d+\s+\d+", txt) and "#aa0033" not in txt and "Ligas: T1" in txt
        stat = Display(False, io.StringIO()); reports.render_status(conn, stat)
        assert "Flojo" in stat.stream.getvalue() and "#aa0033" not in stat.stream.getvalue()
    finally:
        e.admin.execute("UPDATE seasons SET league_champion_id=%s WHERE number=1", (before,))


def test_la_limpieza_y_el_disco_ignoran_a_los_estaticos(static_season):
    e = static_season
    db = e.db(); db.connect()
    try:
        rows = prune.prune_models(db, e.cfg(retention_seasons=1), completed_season=60, dry_run=True)
        assert rows and all(r["color"] not in ("#0033aa", "#aa0033") for r in rows)
    finally:
        db.close()
    d = disk.collect(e.conn(), e.cfg())
    assert sum(n for n, _ in d["cats"].values()) == len(list(e.models.glob("*.pth")))


def test_un_estatico_que_salio_puede_volver_y_uno_que_compite_no(static_season):
    e = static_season
    plan = e.q("SELECT individual_id FROM evolution_plan WHERE season_id=(SELECT id FROM seasons WHERE number=2)")
    competing = {p["individual_id"] for p in plan}
    db = e.db(); db.connect()
    try:
        for name in ("BocaJrs", "Flojo"):
            iid = e.q("SELECT id FROM individuals WHERE static_name=%s", (name,))[0]["id"]
            if iid in competing:
                with pytest.raises(additions.AdditionError, match="ya compite"):
                    additions.register_additions(db, e.cfg(), [name])
            else:
                res = additions.register_additions(db, e.cfg(), [name])      # vuelve con la MISMA identidad
                assert res["target"] == 3
                assert e.q("SELECT count(*) c FROM individuals WHERE static_name=%s", (name,))[0]["c"] == 1
                additions.cancel_addition(db, e.cfg(), name)
                assert e.q("SELECT count(*) c FROM individuals WHERE static_name=%s", (name,))[0]["c"] == 1   # conserva su historia
    finally:
        db.close()


# ====================== referencias, peleas y pruebas ======================
def _two_nets(e):
    return [r["color"] for r in _participants(e, 1) if r["ia_type"] == "neuronal"][:3]


def test_un_numero_de_temporada_siempre_es_la_mejor_red_y_el_estatico_solo_por_nombre(static_season):
    e = static_season
    conn, d = e.conn(), e.statics_dir
    best_net = next(r for r in _participants(e, 1) if r["ia_type"] == "neuronal")["color"]
    ia = sandbox.resolve_ia(conn, e.models, "1", d)
    assert ia["type"] == "neuronal" and ia["color"] == best_net
    st = sandbox.resolve_ia(conn, e.models, "bocajrs", d)
    assert st["type"] == "estatico" and st["name"] == "BocaJrs" and st["color"] == "#0033aa" and st["tag"].startswith("BocaJrs")
    assert sandbox.resolve_ia(conn, e.models, "#0033aa", d)["name"] == "BocaJrs"                 # su color también lo identifica
    with pytest.raises(play.PlayError, match="No existe ninguna IA estática llamada 'nadie'.*BocaJrs"):
        sandbox.resolve_ia(conn, e.models, "nadie", d)
    with pytest.raises(play.PlayError, match="la misma IA"):
        sandbox.resolve_many(conn, e.models, ["BocaJrs", "#0033aa"], d)
    make_dir(d, nuevo=bot_src("Nuevo", "#445566", "#665544", "Nuevo"))                           # existe en el código, aún sin registrar
    nuevo = sandbox.resolve_ia(conn, e.models, "Nuevo", d)
    assert nuevo["type"] == "estatico" and nuevo["color"] == "#445566"
    (d / "nuevo.py").unlink()


def test_pelea_entre_una_red_y_un_estatico(static_season):
    e = static_season
    a, b = sandbox.resolve_many(e.conn(), e.models, ["1", "BocaJrs"], e.statics_dir)
    r1, r2 = sandbox.fight(a, b, seed=3), sandbox.fight(a, b, seed=3)
    assert [(g["result"], g["moves"]) for g in r1["games"]] == [(g["result"], g["moves"]) for g in r2["games"]]   # misma semilla
    g1, g2 = r1["games"]
    assert (g1["x"], g2["x"]) == (a["color"], b["color"])
    out = Display(False, io.StringIO()); sandbox.render_fight(out, a, b, r1)
    assert "BocaJrs" in out.stream.getvalue() and "#0033aa" not in out.stream.getvalue()


def test_mini_liga_de_tres(static_season):
    e = static_season
    n1, n2 = _two_nets(e)[:2]
    ias = sandbox.resolve_many(e.conn(), e.models, ["1", n2, "BocaJrs"], e.statics_dir)
    res = sandbox.fight_round_robin(ias, seed=5)
    assert sum(res["points"].values()) >= 12 and len(res["order"]) == 3 and len(res["fights"]) == 3
    assert all(sum(r) == 4 for r in res["record"].values())                        # 2 rivales x 2 partidas
    out = Display(False, io.StringIO()); sandbox.render_fight_table(out, ias, res, 5)
    assert "MINI LIGA" in out.stream.getvalue() and "BocaJrs" in out.stream.getvalue()
    with pytest.raises(play.PlayError):
        sandbox.fight_round_robin(ias[:2])


def test_liga_de_prueba_con_un_estatico(static_season):
    e = static_season
    tested = sandbox.resolve_many(e.conn(), e.models, ["BocaJrs", "1"], e.statics_dir)
    before = e.q("SELECT count(*) c FROM matches")[0]["c"]
    r = sandbox.run_random_league(tested, seed=7)
    assert len(r["order"]) == 20 and len(r["random"]) == 18 and all(r["table"][c].played == 38 for c in r["order"])
    assert r["summary"]["#0033aa"]["position"] <= 5                                # el estático fuerte queda arriba de los aleatorios
    assert e.q("SELECT count(*) c FROM matches")[0]["c"] == before                 # nada se guarda
    r2 = sandbox.run_random_league(tested, seed=7)
    assert (r2["order"], r2["random"]) == (r["order"], r["random"])
    out = Display(False, io.StringIO()); sandbox.render_random_league(out, r, tested)
    assert "BocaJrs" in out.stream.getvalue() and "◄ probada" in out.stream.getvalue()


def test_medir_un_estatico_contra_los_bots_fijos(static_season):
    e = static_season
    ia = sandbox.resolve_ia(e.conn(), e.models, "BocaJrs", e.statics_dir)
    res = benchmark.bench_ia(ia, 8, 3, lambda i: sandbox.make_player(i, 3))
    assert res["random"]["w"] >= 6 and sum(res["random"][k] for k in ("w", "d", "l")) == 8
    out = Display(False, io.StringIO()); benchmark.render_bench(out, ia, res, 8, 3)
    assert "BocaJrs" in out.stream.getvalue()


def test_jugar_contra_un_estatico(static_season):
    e = static_season
    ia = sandbox.resolve_ia(e.conn(), e.models, "Flojo", e.statics_dir)
    opp = {"season": None, "color": ia["color"], "tag": ia["tag"], "static": True, "name": ia["name"], "color2": ia["color2"],
           "total": None, "league": None, "cup": None, "recopa": None}
    s = play.PlaySession(sandbox.make_player(ia, 0), opp)
    st = s.new_game(-1)                                                            # empieza el estático
    assert st["opponent"]["static"] is True and st["opponent"]["name"] == "Flojo" and st["turn"] == 1
    while st["winner"] is None:
        i = st["valid"].index(1)
        st = s.human_move(i // 9, i % 9)
        assert st["last_ai"] is not None or st["winner"] is not None


def test_torneo_maestro_con_un_estatico(static_season):
    e = static_season
    nets = [r for r in _participants(e, 1) if r["ia_type"] == "neuronal"][:3]
    champs = []
    for k, r in enumerate(nets, start=1):
        ia = sandbox.resolve_ia(e.conn(), e.models, r["color"], e.statics_dir)
        champs.append({"season": k, "color": ia["color"], "path": ia["path"], "sha256": ia["sha256"]})
    st = sandbox.resolve_ia(e.conn(), e.models, "BocaJrs", e.statics_dir)
    champs.append({"season": None, "color": st["color"], "cls": st["cls"], "slug": st["slug"], "path": None, "sha256": None})
    res = ladder.run_ladder(champs)
    assert len(res["elo"]) == 4 and 0 <= res["newer_share"] <= 1
    assert res["vs_older"][3] == res["vs_older"][3] and res["vs_newer"][3] != res["vs_newer"][3]       # el estático: solo contra las redes
    assert res["elo"][3] > min(res["elo"][:3])
    out = Display(False, io.StringIO()); ladder.render_ladder(out, res)
    assert "BocaJrs" in out.stream.getvalue() and "—" in out.stream.getvalue()


def test_cli_y_menu_con_estaticos(static_season, monkeypatch, capsys):
    e = static_season
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={e.schema}")
    monkeypatch.setenv("MODELS_DIR", str(e.models))
    monkeypatch.setenv("STATICS_DIR", str(e.statics_dir))
    n2 = _two_nets(e)[1]
    assert main(["statics"]) == 0
    out = capsys.readouterr().out
    assert "BocaJrs" in out and "Flojo" in out and re.search(r"compitiendo|afuera desde", out)
    assert main(["fight", "1", "BocaJrs", "seed=3"]) == 0 and "PELEA" in capsys.readouterr().out
    assert main(["fight", "1", n2, "BocaJrs", "seed=3"]) == 0 and "MINI LIGA" in capsys.readouterr().out
    assert main(["fight", "1", "nadie"]) == 1 and "No existe ninguna IA estática" in capsys.readouterr().err
    assert main(["fight", "BocaJrs"]) == 2
    assert main(["bench", "BocaJrs", "6", "seed=2"]) == 0 and "BocaJrs" in capsys.readouterr().out
    assert main(["champions"]) == 0 and "Tipo" in capsys.readouterr().out
    assert main(["season", "1"]) == 0 and "BocaJrs" in capsys.readouterr().out
    served = {}
    monkeypatch.setattr(play, "serve", lambda session, host, port, emit=print: served.update(opp=session.opponent))
    assert main(["play", "BocaJrs"]) == 0 and served["opp"]["static"] and served["opp"]["name"] == "BocaJrs"
    assert main(["play", "1"]) == 0 and not served["opp"].get("static")
    assert main(["play", "nadie"]) == 1 and "No existe ninguna IA estática" in capsys.readouterr().err
    answers = iter(["4 Flojo", "5 1 BocaJrs", "12", "7 --list", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["menu"]) == 0
    out = capsys.readouterr().out
    assert served["opp"]["name"] == "Flojo" and "PELEA" in out and "BOTS ESTÁTICOS" in out


def test_un_bot_roto_no_frena_a_los_demas(tmp_path):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    e, d = _cfg_env(tmp_path, fuerte=FUERTE, roto="class Roto(:\n   esto no es python\n", sinclase="x = 1\n")
    try:
        assert set(statics.available(d)) == {"fuerte"} and set(statics.broken(d)) == {"roto", "sinclase"}
        with pytest.raises(statics.StaticError, match="nadie.*Archivos con errores: roto.py"):
            statics.find(d, "nadie")
        db = e.db(); db.connect_with_retries(); db.init_schema()
        additions.register_additions(db, e.cfg(), ["BocaJrs"]); db.close()          # el bueno se registra igual
        e.trainer(max_seasons=1).run()                                               # y compite sin que el roto moleste
        assert e.q("SELECT n_participants n FROM seasons")[0]["n"] == 21
        out = Display(False, io.StringIO()); reports.render_statics(e.conn(), out, e.cfg())
        txt = out.stream.getvalue()
        assert "ERROR en roto.py" in txt and "BocaJrs" in txt
        with pytest.raises(statics.StaticError, match="tiene errores"):
            statics.load_one(d, "roto")
    finally:
        e.close()


def test_si_falta_el_archivo_de_un_estatico_que_compite_el_error_es_claro(static_season, tmp_path):
    from evolution.runner import Trainer
    e = static_season
    t = e.trainer(statics_dir=tmp_path / "vacio")
    with pytest.raises(statics.StaticError, match="compite en esta temporada pero no se encuentra"):
        info = e.q("SELECT id FROM seasons WHERE number=2")[0]
        t.db.connect_with_retries()
        row = t.db.run(lambda c: c.execute("SELECT * FROM seasons WHERE number=2").fetchone())
        from evolution import competitions as comp
        t._load_nets(t.db.run(lambda c: comp.load_season_info(c, row)))
    t.db.close()