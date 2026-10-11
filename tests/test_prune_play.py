"""Limpieza de modelos, ranking con estado, menú con argumentos y juego contra la IA."""
import io
import json
import re
import threading
import urllib.error
import urllib.request

import pytest
import torch

from evolution import play, prune, reports
from evolution.config import load_config
from evolution.display import Display
from evolution.__main__ import main, parse_menu_line
from evolution.modelstore import sidecar_path, weights_hash
from game import SuperTateti
from model import SuperTatetiNet
from tests.helpers import TEST_URL


# ====================== menú / configuración ======================
def test_parse_menu_line():
    assert parse_menu_line("3 100") == ("3", ["100"])
    assert parse_menu_line("  2   21 ") == ("2", ["21"])
    assert parse_menu_line("4") == ("4", [])
    assert parse_menu_line("") == ("", [])


def test_config_nuevas_variables():
    c = load_config({})
    assert (c.retention_seasons, c.champions_top, c.play_port) == (10, 30, 8765)
    c = load_config({"MODEL_RETENTION_SEASONS": "0", "CHAMPIONS_TOP": "50", "PLAY_PORT": "9000"})
    assert (c.retention_seasons, c.champions_top, c.play_port) == (0, 50, 9000)
    with pytest.raises(ValueError):
        load_config({"CHAMPIONS_TOP": "0"})


# ====================== jugar contra la IA (sin base de datos) ======================
def _opponent():
    return {"season": 1, "color": "#112233", "tag": "#112233:01L:00C:00RC", "total": 40, "league": 30, "cup": 10, "recopa": 0}


def _session(seed=1):
    torch.manual_seed(seed)
    return play.PlaySession(SuperTatetiNet().eval(), _opponent())


def _first_valid(state):
    i = state["valid"].index(1)
    return i // 9, i % 9


def test_la_ia_responde_con_jugadas_legales_hasta_terminar():
    for human in (1, -1):
        s = _session(human + 3)
        st = s.new_game(human)
        assert st["started"] and st["human"] == human and st["current"] == human
        assert st["turn"] == (0 if human == 1 else 1)           # si la persona es O, la IA ya movió
        shadow = SuperTateti()
        guard = 0
        while st["winner"] is None:
            b, c = _first_valid(st)
            st = s.human_move(b, c)
            guard += 1
            assert guard < 82
        assert st["winner"] in ("human", "ai", "draw") and sum(st["valid"]) == 0
        assert sum(s.wins.values()) == 1


def test_jugada_ilegal_o_fuera_de_turno_se_rechaza_sin_cambiar_el_estado():
    s = _session()
    with pytest.raises(play.PlayError):
        s.human_move(0, 0)                                      # sin partida
    st = s.new_game(1)
    s.human_move(4, 4)                                          # válida: la IA responde
    st = s.state()
    illegal = [i for i in range(81) if st["valid"][i] == 0][0]
    before = json.dumps(st)
    with pytest.raises(play.PlayError):
        s.human_move(illegal // 9, illegal % 9)
    with pytest.raises(play.PlayError):
        s.human_move(9, 0)
    with pytest.raises(play.PlayError):
        s.human_move("1", 0)
    assert json.dumps(s.state()) == before
    with pytest.raises(play.PlayError):
        s.new_game(7)


def test_servidor_http_partida_completa():
    s = _session(5)
    srv = play.make_server(s, "127.0.0.1", 0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    try:
        code, html = call("/")
        assert code == 200 and b"Super Tateti" in html and b"/api/move" in html
        assert json.loads(call("/api/state")[1])["started"] is False
        assert call("/api/move", {"board": 0, "cell": 0})[0] == 400
        st = json.loads(call("/api/new", {"human": "x"})[1])
        assert st["started"] and st["human"] == 1
        st = json.loads(call("/api/move", {"board": 4, "cell": 4})[1])
        assert st["turn"] == 2 and st["last_human"] == [4, 4] and st["last_ai"] is not None
        code, body = call("/api/move", {"board": "x"})
        assert code == 400 and "error" in json.loads(body)
        assert call("/api/new", {"human": "z"})[0] == 400
        st = json.loads(call("/api/new", {"human": "o"})[1])
        assert st["human"] == -1 and st["turn"] == 1
        assert call("/nada")[0] == 404
    finally:
        srv.shutdown(); srv.server_close()


def test_la_ia_juega_igual_que_en_los_torneos():
    """La respuesta del servidor es select_move de la red (sin MCTS)."""
    from evolution.engine import select_move
    s = _session(9)
    s.new_game(1)
    s.human_move(4, 4)
    shadow = SuperTateti(); shadow.make_move(4, 4)
    mv = select_move(s.net, shadow)
    assert s.state()["last_ai"] == [mv // 9, mv % 9]


# ====================== con la temporada de referencia ======================
def test_oponente_es_el_primero_de_la_tabla_anual(reference_season):
    e = reference_season
    conn = e.conn()
    opp = play.find_opponent(conn, e.models, None)
    best = e.q("""SELECT i.color FROM season_participants p JOIN individuals i ON i.id=p.individual_id
                  WHERE p.final_position=1""")[0]["color"]
    assert opp["color"] == best and opp["season"] == 1 and opp["tag"].startswith(best)
    net = play.load_opponent_net(opp)
    assert weights_hash(net.state_dict()) == opp["sha256"]
    for bad, msg in ((9, "No existe"),):
        with pytest.raises(play.PlayError, match=msg):
            play.find_opponent(conn, e.models, bad)


def test_ranking_limita_filas_y_muestra_estado(reference_season):
    e = reference_season
    conn = e.conn()
    out = Display(False, io.StringIO())
    reports.render_champions(conn, out, top=1)
    txt = out.stream.getvalue()
    assert "Estado" in txt and "Mostrando 1 de" in txt
    rank_rows = [l for l in txt.split("Ranking de campeones")[1].splitlines() if re.match(r"\s+\d+\s+#", l)]
    assert len(rank_rows) == 1 and re.search(r"compitiendo|expulsada T1", rank_rows[0])
    # el estado coincide con la base: expulsada = no pasó a la siguiente generación
    plan = {r["color"] for r in e.q("SELECT i.color FROM evolution_plan p JOIN individuals i ON i.id=p.individual_id")}
    for color, st in reports.individual_status(conn).items():
        assert st["last_season"] == 1 and st["expelled"] == (color not in plan)


def test_prune_en_seco_no_toca_nada_y_respeta_protegidas(reference_season):
    e = reference_season
    db = e.db()
    db.connect()
    try:
        before = sorted(p.name for p in e.models.glob("*.pth"))
        cfg = e.cfg(retention_seasons=10)
        rows = prune.prune_models(db, cfg, completed_season=12, dry_run=True)    # como si hubieran pasado 11 temporadas
        assert sorted(p.name for p in e.models.glob("*.pth")) == before         # dry-run: no borra nada
        protected = {r["color"] for r in e.q(
            """SELECT i.color FROM individuals i WHERE EXISTS (SELECT 1 FROM season_participants q WHERE q.individual_id=i.id AND q.final_position=1)
               OR EXISTS (SELECT 1 FROM seasons z WHERE i.id IN (z.league_champion_id, z.cup_champion_id, z.recopa_champion_id))""")}
        plan = {r["color"] for r in e.q("SELECT i.color FROM evolution_plan p JOIN individuals i ON i.id=p.individual_id")}
        got = {r["color"] for r in rows}
        assert got and not (got & protected) and not (got & plan)
        assert got == {r["color"] for r in e.q("SELECT color FROM individuals WHERE origin_season=1")} - protected - plan
        assert prune.prune_models(db, e.cfg(retention_seasons=0), 12) == []        # 0 = desactivado
        assert prune.prune_models(db, cfg, completed_season=10, dry_run=True) == []  # aún no pasaron 10 temporadas
    finally:
        db.close()


def test_cli_menu_con_argumentos(reference_season, monkeypatch, capsys):
    e = reference_season
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={e.schema}")
    monkeypatch.setenv("MODELS_DIR", str(e.models))
    assert main(["champions", "1"]) == 0 and "Mostrando 1 de" in capsys.readouterr().out
    assert main(["champions", "x"]) == 2
    served = {}
    monkeypatch.setattr(play, "serve", lambda session, host, port, emit=print: served.update(opp=session.opponent, port=port))
    answers = iter(["3 1", "2 1", "4 1", "4", "3 zzz", "99", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["menu"]) == 0
    out = capsys.readouterr().out
    assert "Mostrando 1 de" in out                       # "3 1"
    assert "Tabla anual" in out                          # "2 1"
    assert served["opp"]["season"] == 1 and served["port"] == 8765   # "4 1" / "4"
    assert "debe ser un número entero positivo" in out and "Opción no válida" in out
    assert main(["play", "7"]) == 1                      # temporada inexistente: error claro, sin servidor
    assert "No existe la temporada 7" in capsys.readouterr().err


# ====================== limpieza real a lo largo de varias temporadas ======================
def test_prune_borra_pth_antiguos_y_conserva_json(tmp_path):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    from tests.helpers import Env
    e = Env(tmp_path)
    try:
        # retención 1: tras completar la temporada 3 se borran las expulsadas en la 1 (y antes) sin títulos
        e.trainer(max_seasons=3, retention_seasons=1).run()
        assert e.q("SELECT count(*) n FROM seasons WHERE status='completed'")[0]["n"] == 3
        rows = e.q("""
            WITH last AS (SELECT p.individual_id, max(s.number) n FROM season_participants p JOIN seasons s ON s.id=p.season_id GROUP BY 1)
            SELECT i.id, i.color, i.model_file, i.model_sha256, i.model_pruned_at, l.n AS last_season,
                   EXISTS (SELECT 1 FROM evolution_plan ep JOIN seasons s2 ON s2.id=ep.season_id
                           WHERE s2.number=l.n AND ep.individual_id=i.id) AS continues,
                   (EXISTS (SELECT 1 FROM season_participants q WHERE q.individual_id=i.id AND q.final_position=1)
                    OR EXISTS (SELECT 1 FROM seasons z WHERE i.id IN (z.league_champion_id, z.cup_champion_id, z.recopa_champion_id))) AS protected
            FROM individuals i LEFT JOIN last l ON l.individual_id=i.id""")
        pruned = kept = 0
        for r in rows:
            pth = e.models / r["model_file"]
            should = r["last_season"] is not None and not r["continues"] and not r["protected"] and r["last_season"] <= 2
            if should:
                pruned += 1
                assert r["model_pruned_at"] is not None and not pth.exists(), r["color"]
                info = json.loads(sidecar_path(pth).read_text())           # el .json se conserva con la información
                assert info["weights_deleted"] is True and info["weights_sha256"] == r["model_sha256"]
            else:
                kept += 1
                assert r["model_pruned_at"] is None and pth.exists(), r["color"]
        assert pruned > 0 and kept > 0
        # protegidas: 1.º de cada tabla anual y campeones conservan su .pth aunque estén expulsadas
        for r in e.q("""SELECT i.model_file FROM season_participants p JOIN individuals i ON i.id=p.individual_id
                        WHERE p.final_position=1"""):
            assert (e.models / r["model_file"]).exists()
        assert list((e.models / "historical").glob("*.pth"))              # históricos intactos
        assert "Limpieza de modelos" in e.out.getvalue()
    finally:
        e.close()