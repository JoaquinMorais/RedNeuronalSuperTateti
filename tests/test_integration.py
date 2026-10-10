"""Integración contra PostgreSQL real (TEST_DATABASE_URL)."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psycopg
import pytest

from evolution import modelstore, rules
from evolution.db import AlreadyRunningError, Database
from evolution.runner import StopFlag
from tests.helpers import ROOT, TEST_URL, Env


# ---------------- temporada completa ----------------
def test_temporada_completa_invariantes(reference_season):
    e = reference_season
    assert e.q("SELECT status, stage FROM seasons")[0] == {"status": "completed", "stage": "completed"}
    assert e.q("SELECT count(*) n FROM season_participants")[0]["n"] == 20
    cols = [r["color"] for r in e.q("SELECT color FROM individuals")]
    assert len(cols) == len(set(cols)) == 33 and all(c.startswith("#") and len(c) == 7 for c in cols)
    # Liga
    assert e.q("SELECT count(*) n FROM matches WHERE competition='league' AND status='finished'")[0]["n"] == 380
    assert e.q("SELECT count(DISTINCT matchday) n FROM matches WHERE competition='league'")[0]["n"] == 38
    per = e.q("""SELECT count(*) n FROM (SELECT p FROM (SELECT x_individual_id p FROM matches WHERE competition='league'
                 UNION ALL SELECT o_individual_id FROM matches WHERE competition='league') t GROUP BY p HAVING count(*)=38) z""")
    assert per[0]["n"] == 20
    # Grupos y clasificación
    assert e.q("SELECT count(*) n FROM matches WHERE competition='cup_group'")[0]["n"] == 80
    for g in "ABCD":
        assert e.q("SELECT count(*) n FROM season_participants WHERE cup_group=%s", (g,))[0]["n"] == 5
        assert e.q("SELECT count(*) n FROM matches WHERE competition='cup_group' AND group_name=%s", (g,))[0]["n"] == 20
        q = e.q("SELECT position, qualified FROM cup_group_table WHERE group_name=%s ORDER BY position", (g,))
        assert [r["qualified"] for r in q] == [True] * 4 + [False]
    # Octavos exactos a partir de las posiciones finales de grupo
    pos = {f"{r['position']}{r['group_name']}": r["individual_id"] for r in e.q("SELECT * FROM cup_group_table")}
    for t in e.q("SELECT * FROM cup_ties WHERE round='R16'"):
        assert (t["individual_a_id"], t["individual_b_id"]) == (pos[t["slot_a"]], pos[t["slot_b"]])
    # Fechas de las rondas
    assert {r["round"]: r["matchday"] for r in e.q("SELECT DISTINCT round, matchday FROM cup_ties")} == rules.KNOCKOUT_MATCHDAYS
    assert {r["matchday"] for r in e.q("SELECT matchday FROM matches WHERE competition='cup_group'")} <= set(range(1, 13))
    # Cada eliminatoria: ida y vuelta con inicio alternado (+ desempate solo si hubo empate)
    for t in e.q("SELECT * FROM cup_ties"):
        legs = e.q("SELECT * FROM matches WHERE tie_id=%s ORDER BY seq", (t["id"],))
        assert legs[0]["x_individual_id"] == legs[1]["o_individual_id"] and legs[0]["o_individual_id"] == legs[1]["x_individual_id"]
        assert len(legs) in (2, 3) and t["status"] == "finished"
    # Puntos de Copa
    stages = {r["cup_stage"]: r["cup_score"] for r in e.q("SELECT DISTINCT cup_stage, cup_score FROM season_participants")}
    assert stages == {"champion": 24, "runner_up": 17, "SF": 12, "QF": 7, "R16": 3, "group": 0}
    assert e.q("SELECT count(*) n FROM season_participants WHERE cup_stage='champion'")[0]["n"] == 1
    # Recopa
    rec = e.q("SELECT * FROM recopa_results")[0]
    n_rec = e.q("SELECT count(*) n FROM matches WHERE competition='recopa'")[0]["n"]
    assert n_rec == (0 if rec["same_champion"] else 10)
    if not rec["same_champion"]:
        starts = e.q("SELECT x_individual_id x, count(*) n FROM matches WHERE competition='recopa' GROUP BY 1")
        assert sorted(r["n"] for r in starts) == [5, 5]
    assert e.q("SELECT sum(recopa_score) s FROM season_participants")[0]["s"] == 5
    # Puntuación global
    for r in e.q("SELECT * FROM season_participants"):
        assert r["total_score"] == r["league_score"] + r["cup_score"] + r["recopa_score"]
    assert sorted(r["final_position"] for r in e.q("SELECT final_position FROM season_participants")) == list(range(1, 21))
    totals = [r["total_score"] for r in e.q("SELECT total_score FROM season_participants ORDER BY final_position")]
    assert totals == sorted(totals, reverse=True)
    for r in e.q("SELECT league_position p, league_score s FROM season_participants"):
        assert r["s"] == rules.LEAGUE_REWARD[r["p"] - 1]
    assert e.q("SELECT count(*) n FROM league_history")[0]["n"] == 38 * 20


def test_siguiente_generacion_7_mas_13(reference_season):
    e = reference_season
    plan = e.q("SELECT * FROM evolution_plan ORDER BY position")
    assert len(plan) == 20 and [p["role"] for p in plan] == ["survivor"] * 7 + ["child"] * 13
    top7 = [r["individual_id"] for r in e.q("SELECT individual_id FROM season_participants WHERE final_position<=7 ORDER BY final_position")]
    assert [p["individual_id"] for p in plan[:7]] == top7
    # Supervivientes: mismos archivos y pesos (no se tocaron durante ni después de la temporada)
    for iid in top7:
        r = e.q("SELECT model_file, model_sha256 FROM individuals WHERE id=%s", (iid,))[0]
        assert modelstore.weights_hash(modelstore.load_state_dict(e.models / r["model_file"])) == r["model_sha256"]
    # Padres intactos y linaje explícito; hijos con semilla y nivel
    for p in plan[7:]:
        ch = e.q("SELECT * FROM individuals WHERE id=%s", (p["individual_id"],))[0]
        par = e.q("SELECT * FROM individuals WHERE id=%s", (ch["parent_id"],))[0]
        assert ch["parent_id"] == p["parent_id"] and ch["mutation_seed"] == p["mutation_seed"] and ch["mutation_level"] in ("small", "medium", "large")
        assert ch["origin_season"] == 2 and ch["model_sha256"] != par["model_sha256"]
        assert modelstore.weights_hash(modelstore.load_state_dict(e.models / par["model_file"])) == par["model_sha256"]
        assert (e.models / ch["model_file"]).exists()
    assert len({p["parent_id"] for p in plan[7:]}) >= 2
    hb = e.q("SELECT * FROM historical_best")[0]
    assert (e.models / hb["model_file"]).exists()


def test_mcts_no_se_ejecuta_en_el_entrenamiento(reference_season):
    assert "mcts" not in sys.modules


def test_reproducible_misma_semilla(reference_season, tmp_path):
    e2 = Env(tmp_path, seed=42)
    try:
        e2.trainer().run()
        assert e2.signature() == reference_season.sig
    finally:
        e2.close()


def test_semilla_distinta_cambia_resultados(reference_season, tmp_path):
    e2 = Env(tmp_path, seed=43)
    try:
        e2.trainer().run()
        assert e2.signature()["hash"] != reference_season.sig["hash"]
    finally:
        e2.close()


# ---------------- reanudación ----------------
def test_interrupcion_entre_fechas_no_repite_partidas(reference_season, tmp_path):
    e = Env(tmp_path)
    try:
        stop = StopFlag()
        e.trainer(stop=stop, on_matchday_closed=lambda md: stop.set() if md == 5 else None).run()
        assert e.q("SELECT current_matchday FROM seasons")[0]["current_matchday"] == 5
        before = {r["id"]: r["finished_at"] for r in e.q("SELECT id, finished_at FROM matches WHERE status='finished'")}
        assert before and e.q("SELECT status FROM seasons")[0]["status"] == "in_progress"
        e.trainer().run()  # "reinicio": proceso nuevo, estado solo desde la BD
        after = {r["id"]: r["finished_at"] for r in e.q("SELECT id, finished_at FROM matches WHERE status='finished'")}
        assert all(after[i] == t for i, t in before.items())  # ninguna finalizada se rejugó
        assert e.q("SELECT count(*) n FROM seasons")[0]["n"] == 1
        assert e.signature() == reference_season.sig  # y el resultado es idéntico al sin interrupción
        assert "REANUDADA" in e.out.getvalue()
    finally:
        e.close()


def test_interrupcion_durante_una_partida(reference_season, tmp_path):
    e = Env(tmp_path)
    try:
        calls = {"n": 0}

        def crash(match, outcome):
            calls["n"] += 1
            if calls["n"] == 25:
                raise RuntimeError("corte de luz")  # tras jugar, antes de registrar

        with pytest.raises(RuntimeError):
            e.trainer(on_match_played=crash).run()
        st = {r["status"]: r["n"] for r in e.q("SELECT status, count(*) n FROM matches GROUP BY status")}
        assert st["running"] == 1 and st["finished"] == 24
        e.trainer().run()
        assert "Recuperadas 1 partida" in e.out.getvalue()
        assert e.q("SELECT count(*) n FROM matches WHERE status<>'finished'")[0]["n"] == 0
        assert e.signature() == reference_season.sig
    finally:
        e.close()


def test_registro_de_resultado_es_idempotente(env):
    from evolution import competitions as comp
    from evolution.engine import MatchOutcome
    t = env.trainer()
    t.db.connect_with_retries(); t.db.init_schema()
    t._create_next_season(42)
    m = t.db.run(lambda c: c.execute("SELECT * FROM matches ORDER BY id LIMIT 1").fetchone())
    out = MatchOutcome("x", "1,2", 2, 1.0)
    assert t.db.run(lambda c: comp.record_result(c, m, out)) is True
    assert t.db.run(lambda c: comp.record_result(c, m, MatchOutcome("o", "9", 1, 1.0))) is False  # no se pisa ni cuenta doble
    assert t.db.run(lambda c: comp.claim_match(c, m["id"])) is False
    assert env.q("SELECT result FROM matches WHERE id=%s", (m["id"],))[0]["result"] == "x"
    t.db.close()


def test_recuperacion_de_conexion_a_la_base(reference_season, tmp_path):
    e = Env(tmp_path)
    try:
        db = e.db()
        state = {"n": 0}

        def kill_backend(match, outcome):
            state["n"] += 1
            if state["n"] == 30:  # corta la conexión del entrenador desde otra sesión
                with psycopg.connect(TEST_URL, autocommit=True) as c:
                    c.execute("SELECT pg_terminate_backend(%s)", (db.conn.info.backend_pid,))

        e.trainer(db=db, on_match_played=kill_backend).run()
        assert db.reconnections >= 1
        assert e.signature() == reference_season.sig
    finally:
        e.close()


def test_bloqueo_impide_dos_entrenamientos(env):
    a, b = env.db(), env.db()
    a.connect()
    with pytest.raises(AlreadyRunningError):
        b.connect()
    a.close()
    b.connect()  # al liberarse, otro puede tomarlo
    b.close()


def test_nueva_temporada_se_crea_desde_la_bd_tras_reinicio(tmp_path):
    """Un 'reinicio de Docker' = proceso nuevo con la misma BD y el mismo directorio de modelos."""
    e = Env(tmp_path)
    try:
        e.trainer().run()  # temporada 1 completa
        e.trainer(max_seasons=0, stop=_StopAfterLoad()).run()  # proceso nuevo: crea la temporada 2 y se detiene
        assert e.q("SELECT number, status FROM seasons ORDER BY number")[-1] == {"number": 2, "status": "in_progress"}
        sid2 = "(SELECT id FROM seasons WHERE number=2)"
        ids = {r["individual_id"] for r in e.q(f"SELECT individual_id FROM season_participants WHERE season_id={sid2}")}
        plan = {r["individual_id"] for r in e.q("SELECT individual_id FROM evolution_plan")}
        assert ids == plan and len(ids) == 20
        # calendarios completos guardados antes de jugar nada
        assert e.q(f"SELECT count(*) n FROM matches WHERE season_id={sid2} AND competition='league'")[0]["n"] == 380
        assert e.q(f"SELECT count(*) n FROM matches WHERE season_id={sid2} AND competition='cup_group'")[0]["n"] == 80
        assert e.q(f"SELECT count(*) n FROM matches WHERE season_id={sid2} AND status='finished'")[0]["n"] == 0
        assert e.q(f"SELECT count(*) n FROM cup_ties WHERE season_id={sid2}")[0]["n"] == 15
        for r in e.q("SELECT model_file, model_sha256 FROM individuals WHERE id = ANY(%s)", (list(ids),)):
            assert modelstore.weights_hash(modelstore.load_state_dict(e.models / r["model_file"])) == r["model_sha256"]
    finally:
        e.close()


class _StopAfterLoad(StopFlag):
    """Se activa en cuanto se consulta por segunda vez: permite crear la temporada y cortar antes de jugar."""
    def __init__(self):
        super().__init__(); self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > 1


# ---------------- Docker/señales: proceso real ----------------
def test_sigterm_cierre_seguro_y_reanudacion(reference_season, tmp_path):
    e = Env(tmp_path)
    try:
        env_vars = {**os.environ, "DATABASE_URL": TEST_URL, "MODELS_DIR": str(e.models), "SEED": "42", "MAX_SEASONS": "1",
                    "SEED_MODEL_PATH": str(ROOT / "model.pth"), "PYTHONUNBUFFERED": "1",
                    "PGOPTIONS": f"-c search_path={e.schema}"}
        proc = subprocess.Popen([sys.executable, "-m", "evolution", "run"], cwd=ROOT, env=env_vars,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        seen = []
        for line in proc.stdout:
            seen.append(line)
            if "FECHA 4 /" in line:
                proc.send_signal(signal.SIGTERM)
                break
        rest, _ = proc.communicate(timeout=60)
        assert proc.returncode == 0 and "Detención solicitada" in rest
        done = e.q("SELECT current_matchday FROM seasons")[0]["current_matchday"]
        assert 3 <= done < 38
        r = subprocess.run([sys.executable, "-m", "evolution", "run"], cwd=ROOT, env=env_vars, capture_output=True, text=True, timeout=240)
        assert r.returncode == 0 and "REANUDADA" in r.stdout
        assert e.signature() == reference_season.sig
    finally:
        e.close()
