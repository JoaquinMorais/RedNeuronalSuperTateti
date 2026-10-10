"""Insignias de títulos, ranking de campeones y vistas de análisis."""
import io
import re

import pytest

from evolution import reports, titles
from evolution.display import Display
from tests.helpers import TEST_URL

ROWS = [
    {"number": 1, "league": "#aaaaaa", "cup": "#bbbbbb", "recopa": "#aaaaaa"},
    {"number": 2, "league": "#aaaaaa", "cup": "#cccccc", "recopa": "#cccccc"},
    {"number": 3, "league": "#dddddd", "cup": "#bbbbbb", "recopa": "#bbbbbb"},
    {"number": 4, "league": "#dddddd", "cup": "#cccccc", "recopa": "#dddddd"},
]


# ---------------- lógica pura ----------------
def test_insignia_formato_y_ocultamiento():
    idx = titles.build_title_index(ROWS[:1])
    assert titles.badge(idx["#aaaaaa"]) == ":01L:00C:01RC"
    assert titles.badge(idx["#bbbbbb"]) == ":00L:01C:00RC"
    assert titles.badge(None) == "" and titles.badge({"L": [], "C": [], "RC": []}) == ""
    d = Display(False)
    d.titles = idx
    assert d.tag("#aaaaaa") == "#aaaaaa:01L:00C:01RC"
    assert d.tag("#eeeeee") == "#eeeeee"            # nunca ganó nada: sin insignia
    assert d.tag("#aaaaaa", badge=False) == "#aaaaaa"


def test_ranking_prioridad_liga_copa_recopa():
    idx = titles.build_title_index(ROWS)
    rank = [c for c, _ in titles.rank_champions(idx)]
    assert idx["#aaaaaa"] == {"L": [1, 2], "C": [], "RC": [1]}
    assert idx["#dddddd"] == {"L": [3, 4], "C": [], "RC": [4]}
    assert rank[:2] == ["#aaaaaa", "#dddddd"]        # empate total: gana quien ganó antes
    assert rank[2:] == ["#bbbbbb", "#cccccc"]        # 1L menos: ambos con 0 Ligas, 1 y 2 Copas...


def test_ranking_unica_con_mas_ligas_va_primera():
    rows = ROWS + [{"number": 5, "league": "#ffffff", "cup": None, "recopa": None},
                   {"number": 6, "league": "#ffffff", "cup": None, "recopa": None},
                   {"number": 7, "league": "#ffffff", "cup": None, "recopa": None}]
    rank = titles.rank_champions(titles.build_title_index(rows))
    assert rank[0][0] == "#ffffff" and rank[0][1]["L"] == [5, 6, 7]


def test_ranking_copas_antes_que_recopas():
    rows = [{"number": 1, "league": "#111111", "cup": "#222222", "recopa": "#111111"},
            {"number": 2, "league": "#333333", "cup": "#222222", "recopa": "#444444"}]
    rank = [c for c, _ in titles.rank_champions(titles.build_title_index(rows))]
    # Liga manda: #111111 (1L+1RC) > #333333 (1L) > #222222 (0L, 2C) > #444444 (0L, 0C, 1RC)
    assert rank == ["#111111", "#333333", "#222222", "#444444"]


def test_bracket_con_cuadro_vacio():
    ties = {}
    for i in range(1, 9):
        ties[("R16", i)] = {"slot_a": "1A", "slot_b": "4B", "a": None, "b": None, "winner": None, "third": False}
    for rnd, n in (("QF", 4), ("SF", 2), ("F", 1)):
        for k in range(1, n + 1):
            ties[(rnd, k)] = {"slot_a": "", "slot_b": "", "a": None, "b": None, "winner": None, "third": False}
    lines = reports.render_bracket(ties, Display(False))
    assert "Octavos" in lines[0] and "Final" in lines[0] and "?" in "".join(lines)


# ---------------- vistas sobre una temporada real ----------------
@pytest.fixture
def conn(reference_season):
    return reference_season.conn()


def _render(fn, *args):
    out = Display(False, io.StringIO())
    fn(*args[:1], out, *args[1:])
    return out.stream.getvalue()


def test_vista_de_temporada(reference_season, conn):
    txt = _render(reports.render_season, conn, 1)
    e = reference_season
    season = e.q("SELECT * FROM seasons")[0]
    color = {r["id"]: r["color"] for r in e.q("SELECT id, color FROM individuals")}
    assert "Tabla de Liga" in txt and "Copa" in txt and "Recopa" in txt and "Tabla anual" in txt
    league_rows = [l for l in txt.split("Tabla de Liga")[1].split("--- Copa")[0].splitlines() if re.match(r"\s+\d+\s+#", l)]
    assert len(league_rows) == 20
    assert league_rows[0].split()[1].startswith(color[season["league_champion_id"]])
    for h in ("Octavos", "Cuartos", "Semifinales", "Final", "CAMPEÓN"):
        assert h in txt
    # el campeón de Copa aparece en el cuadro y en la línea de campeón
    assert re.search(re.escape(color[season["cup_champion_id"]]) + r"(:\d\dL:\d\dC:\d\dRC)? CAMPEÓN", txt)
    # la Copa (grupos y cuadro) muestra las insignias de las IA que ya son campeonas
    copa = txt.split("--- Copa ---")[1].split("--- Recopa ---")[0]
    for key in ("league_champion_id", "cup_champion_id"):
        assert re.search(re.escape(color[season[key]]) + r":\d\dL:\d\dC:\d\dRC", copa), key
    assert f"Campeón: {color[season['recopa_champion_id']]}" in txt
    # tabla anual: 7 conservadas y 13 hijos en total, y los hijos por padre coinciden con la BD
    annual = txt.split("Tabla anual")[1]
    assert annual.count("se conserva") >= 7
    assert "Se conservan 7 · descendientes 13 · siguiente generación 20 IA" in annual
    rows = [l for l in annual.splitlines() if re.match(r"\s+\d+\s+#", l)]
    assert len(rows) == 20
    hijos = sum(int(m.group(1)) for l in rows for m in [re.search(r"(\d+) hijos?", l)] if m)
    assert hijos == 13
    kids = {r["parent_id"]: r["n"] for r in e.q("SELECT parent_id, count(*) n FROM evolution_plan WHERE role='child' GROUP BY 1")}
    for l in rows:
        col = l.split()[1].split(":")[0]
        iid = next(i for i, c in color.items() if c == col)
        m = re.search(r"(\d+) hijos?", l)
        assert (int(m.group(1)) if m else 0) == kids.get(iid, 0)
    # las campeonas llevan insignia
    assert re.search(r"#[0-9a-f]{6}:01L:", txt) and re.search(r"#[0-9a-f]{6}:\d\dL:\d\dC:\d\dRC", txt)


def test_vista_temporada_inexistente(conn):
    out = Display(False, io.StringIO())
    assert reports.render_season(conn, out, 9) is False


def test_status_y_campeones(reference_season, conn):
    e = reference_season
    s = e.q("SELECT * FROM seasons")[0]
    color = {r["id"]: r["color"] for r in e.q("SELECT id, color FROM individuals")}
    st = _render(reports.render_status, conn)
    assert (f"Temporada 1: completed · etapa completed · fecha cerrada 38/38 · Liga: {color[s['league_champion_id']]}"
            f" · Copa: {color[s['cup_champion_id']]} · Recopa: {color[s['recopa_champion_id']]}") in st
    ch = _render(reports.render_champions, conn)
    assert "Campeones de cada temporada" in ch and "Ranking de campeones" in ch
    assert color[s["cup_runner_up_id"]] in ch
    assert "Ligas: T1" in ch and "Copas: T1" in ch


def test_cli_status_season_champions_menu(reference_season, monkeypatch, capsys):
    from evolution.__main__ import main
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={reference_season.schema}")
    assert main(["status"]) == 0 and "Temporada 1: completed" in capsys.readouterr().out
    assert main(["season", "1"]) == 0 and "Tabla anual" in capsys.readouterr().out
    assert main(["champions"]) == 0 and "Ranking de campeones" in capsys.readouterr().out
    assert main(["season", "x"]) == 2
    answers = iter(["2", "1", "3", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["menu"]) == 0
    out = capsys.readouterr().out
    assert "Tabla anual" in out and "Ranking de campeones" in out