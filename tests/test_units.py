"""Pruebas unitarias (sin base de datos)."""
import ast
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from evolution import colors as C
from evolution import engine, modelstore, mutation, rules, schedule
from evolution.config import DEFAULT_MUTATION_LEVELS, load_config
from evolution.rng import derive_rng
from game import SuperTateti
from model import SuperTatetiNet

ROOT = Path(__file__).resolve().parent.parent


def rf(seed=1):
    return lambda ctx: derive_rng(seed, "draw", ctx)


# ---------------- colores ----------------
def test_colores_formato_y_unicidad():
    rng, taken = random.Random(0), set()
    for _ in range(300):
        c = C.random_color(rng, taken).color
        assert C.is_valid_hex(c) and c not in taken
        taken.add(c)
    assert len(taken) == 300


@pytest.mark.parametrize("level,lo,hi", [("small", 1, 3), ("medium", 4, 24), ("large", 25, 80)])
def test_mutacion_de_color_respeta_magnitud(level, lo, hi):
    rng, taken = random.Random(3), {"#808080"}
    for _ in range(200):
        r = C.mutate_color("#808080", level, rng, taken)
        assert C.is_valid_hex(r.color) and r.meta["tier"] == level
        diffs = [abs(a - b) for a, b in zip(C.to_rgb(r.color), C.to_rgb("#808080")) if a != b]
        assert diffs and all(lo <= d <= hi for d in diffs)
        taken.add(r.color)


def test_color_colision_y_reserva():
    parent = "#808080"
    # Ocupa TODO el cubo de radio 90 alrededor del padre: la variación no puede tener éxito
    taken = {C.to_hex((128 + dr, 128 + dg, 128 + db)) for dr in range(-90, 91) for dg in range(-90, 91) for db in range(-90, 91)}
    r = C.mutate_color(parent, "small", random.Random(1), taken, max_attempts=20)
    assert r.meta["strategy"] == "nearest_free_scan" and r.color not in taken and C.is_valid_hex(r.color)
    assert max(abs(a - 128) for a in C.to_rgb(r.color)) == 91  # el libre más cercano


def test_color_escala_si_pequena_colisiona():
    taken = {"#808080"}
    for ch in range(3):  # bloquea todas las variaciones pequeñas posibles
        pass
    taken |= {C.to_hex((128 + a, 128 + b, 128 + c)) for a in range(-3, 4) for b in range(-3, 4) for c in range(-3, 4)}
    r = C.mutate_color("#808080", "small", random.Random(2), taken, max_attempts=30)
    assert r.meta["escalated"] and r.meta["tier"] in ("medium", "large")


# ---------------- mutación ----------------
def test_mutacion_no_altera_al_padre_y_es_reproducible():
    parent = SuperTatetiNet().state_dict()
    h0 = modelstore.weights_hash(parent)
    a = mutation.mutate_state_dict(parent, DEFAULT_MUTATION_LEVELS["large"], 7)
    b = mutation.mutate_state_dict(parent, DEFAULT_MUTATION_LEVELS["large"], 7)
    c = mutation.mutate_state_dict(parent, DEFAULT_MUTATION_LEVELS["large"], 8)
    assert modelstore.weights_hash(parent) == h0
    assert modelstore.weights_hash(a) == modelstore.weights_hash(b) != modelstore.weights_hash(c)
    assert modelstore.weights_hash(a) != h0
    assert torch.equal(a["value_head.0.weight"], parent["value_head.0.weight"])
    for k in parent:  # copia independiente: no comparte memoria
        assert a[k].data_ptr() != parent[k].data_ptr()


def test_niveles_de_mutacion_ordenados_por_magnitud():
    parent = SuperTatetiNet().state_dict()
    dist = []
    for lv in ("small", "medium", "large"):
        ch = mutation.mutate_state_dict(parent, DEFAULT_MUTATION_LEVELS[lv], 5)
        dist.append(sum(float((ch[k] - parent[k]).norm()) for k in parent))
    assert dist[0] < dist[1] < dist[2]


def test_padres_segun_tabla_anual():
    ranked = [f"#{i:06x}" for i in range(20)]            # ranked[0] = 1.º de la tabla anual
    p = mutation.parents_by_rank(ranked, (5, 3, 2, 2, 1, 0, 0))
    assert len(p) == 13
    assert Counter(p) == {ranked[0]: 5, ranked[1]: 3, ranked[2]: 2, ranked[3]: 2, ranked[4]: 1}
    assert ranked[5] not in p and ranked[6] not in p and not any(x in p for x in ranked[7:])  # 8.º, 9.º... jamás
    assert p == mutation.parents_by_rank(ranked, (5, 3, 2, 2, 1, 0, 0))                       # determinista
    with pytest.raises(ValueError):
        mutation.parents_by_rank(ranked, (5, 3, 2, 2, 0, 0, 0))                              # no suma 13


def test_politica_de_hijos_configurable_y_validada():
    assert load_config({}).parent_policy.children_per_rank == (5, 3, 2, 2, 1, 0, 0)
    import json, tempfile
    f = Path(tempfile.mkdtemp()) / "c.json"
    f.write_text(json.dumps({"parent_policy": {"children_per_rank": [4, 3, 2, 2, 1, 1, 0]}}))
    assert load_config({"EVO_CONFIG_FILE": str(f)}).parent_policy.children_per_rank == (4, 3, 2, 2, 1, 1, 0)
    f.write_text(json.dumps({"parent_policy": {"children_per_rank": [4, 3, 2, 2, 1, 0, 0]}}))
    with pytest.raises(ValueError):
        load_config({"EVO_CONFIG_FILE": str(f)})


# ---------------- calendarios ----------------
def test_liga_38_fechas_380_partidas_38_por_ia():
    fx = schedule.league_fixtures()
    assert len(fx) == 38 and sum(len(d) for d in fx) == 380
    per_ia, pairs = Counter(), Counter()
    for day in fx:
        assert sorted(s for p in day for s in p) == list(range(1, 21))  # cada IA juega 1 vez por fecha
        for x, o in day:
            per_ia[x] += 1; per_ia[o] += 1; pairs[(x, o)] += 1
    assert set(per_ia.values()) == {38}
    for (x, o), n in pairs.items():  # ida y vuelta con inicio alternado
        assert n == 1 and pairs[(o, x)] == 1
    assert len(pairs) == 380


def test_grupos():
    fx = schedule.group_fixtures()
    assert len(fx) == 10 and all(1 <= md <= 12 for md, _ in fx) and len({md for md, _ in fx}) == 10
    per, pairs = Counter(), Counter()
    for _, rnd in fx:
        assert len(rnd) == 2
        for x, o in rnd:
            per[x] += 1; per[o] += 1; pairs[(x, o)] += 1
    assert sum(len(r) for _, r in fx) == 20 and set(per.values()) == {8} and len(per) == 5
    assert all(pairs[(o, x)] == 1 for x, o in list(pairs))
    g = schedule.draw_groups([f"#{i:06x}" for i in range(20)], random.Random(1))
    assert sorted(g) == list("ABCD") and all(len(v) == 5 for v in g.values())


def test_cruces_de_octavos_exactos():
    r16 = [(t["slot_a"], t["slot_b"]) for t in schedule.knockout_bracket() if t["round"] == "R16"]
    assert r16 == [("1A", "4B"), ("2A", "3B"), ("3A", "2B"), ("4A", "1B"),
                   ("1C", "4D"), ("2C", "3D"), ("3C", "2D"), ("4C", "1D")]
    days = {t["round"]: t["matchday"] for t in schedule.knockout_bracket()}
    assert days == {"R16": 18, "QF": 23, "SF": 30, "F": 37}
    assert Counter(t["round"] for t in schedule.knockout_bracket()) == {"R16": 8, "QF": 4, "SF": 2, "F": 1}


# ---------------- reglas ----------------
def test_puntos_de_copa_y_liga():
    assert rules.CUP_REWARD == {"champion": 24, "runner_up": 17, "SF": 12, "QF": 7, "R16": 3, "group": 0}
    assert rules.LEAGUE_REWARD[:3] == [30, 24, 21] and rules.LEAGUE_REWARD[-3:] == [-3, -3, -3] and len(rules.LEAGUE_REWARD) == 20
    assert [rules.match_points("x", "x"), rules.match_points("x", "o"), rules.match_points("draw", "o")] == [3, 0, 1]


def test_eliminatoria():
    L = lambda *r: [("A", "B", r[0]), ("B", "A", r[1])]
    assert rules.evaluate_two_legs("A", "B", L("x", "o")) == ("A", "two_wins")        # A gana la ida; en la vuelta A (O) gana
    assert rules.evaluate_two_legs("A", "B", L("x", "draw")) == ("A", "win_and_draw")
    assert rules.evaluate_two_legs("A", "B", L("draw", "x")) == ("B", "win_and_draw")
    assert rules.evaluate_two_legs("A", "B", L("x", "x"))[0] is None                  # 1-1
    assert rules.evaluate_two_legs("A", "B", L("draw", "draw"))[0] is None            # 0-0
    assert rules.evaluate_third_game("A", "B", "x") == ("A", "third_game")
    assert rules.evaluate_third_game("A", "B", "draw")[0] == "B"                      # gana quien comenzó segundo


def test_liga_desempate_por_fecha_y_multiple():
    pids = ["#a", "#b", "#c", "#d"]
    fp = {"#a": 10, "#b": 10, "#c": 10, "#d": 5}
    hist = {"#a": [3, 6, 10], "#b": [1, 10, 10], "#c": [4, 4, 10], "#d": [1, 2, 5]}
    draws = []
    # b alcanza 10 en la fecha 2, a y c en la 3: b primero; a y c por sorteo reproducible
    o1 = rules.league_final_order(pids, fp, hist, rf(1), draws)
    assert o1[0] == "#b" and o1[3] == "#d" and set(o1[1:3]) == {"#a", "#c"}
    assert rules.league_final_order(pids, fp, hist, rf(1), []) == o1
    assert len(draws) == 1 and draws[0].members == ["#a", "#c"]
    # empate múltiple de 3 en la misma fecha: orden no depende del orden de entrada
    fp3 = {p: 9 for p in pids[:3]}; h3 = {p: [3, 6, 9] for p in pids[:3]}
    base = rules.league_final_order(pids[:3], fp3, h3, rf(2), [])
    assert rules.league_final_order(pids[:3][::-1], fp3, h3, rf(2), []) == base
    assert set(base) == set(pids[:3])


def test_recopa_conteo_de_empates():
    g = [("C", "L", "x"), ("L", "C", "draw"), ("C", "L", "o"), ("L", "C", "o")]
    wc, wl, d, w = rules.recopa_outcome("C", "L", g)
    assert (wc, wl, d, w) == (2, 1, 1, "C")
    assert rules.recopa_outcome("C", "L", [("C", "L", "draw")] * 10)[3] is None
    assert rules.recopa_outcome("C", "L", [("C", "L", "x"), ("L", "C", "x")])[3] is None  # 1-1


def test_puntuacion_global_suma_y_desempates():
    P = ["#a", "#b", "#c"]
    ls = {"#a": 30, "#b": 24, "#c": 21}; cs = {"#a": 0, "#b": 24, "#c": 17}; rs = {"#a": 0, "#b": 5, "#c": 0}
    pos = {"#a": 1, "#b": 2, "#c": 3}
    assert rules.global_order(P, ls, cs, rs, pos, rf(), []) == ["#b", "#c", "#a"]
    cs2 = {"#a": 24, "#b": 0, "#c": 0}; rs2 = {p: 0 for p in P}  # a=54, b=24, c=21
    assert rules.global_order(P, ls, cs2, rs2, pos, rf(), [])[0] == "#a"
    # empate de totales: gana la mejor posición de Liga
    ls3 = {"#a": 30, "#b": 24, "#c": 0}; cs3 = {"#a": 0, "#b": 6, "#c": 0}
    assert rules.global_order(["#b", "#a"], ls3, cs3, {p: 0 for p in P}, pos, rf(), [])[0] == "#a"


# ---------------- motor / MCTS / modelos ----------------
def test_ia_solo_juega_legal_y_sin_mcts():
    sys.modules.pop("mcts", None)
    net = SuperTatetiNet().eval()
    for seed in range(5):
        torch.manual_seed(seed)
        n1, n2 = SuperTatetiNet().eval(), SuperTatetiNet().eval()
        g = SuperTateti()
        while g.winner is None:
            mask = g.get_valid_moves()
            mv = engine.select_move(n1 if g.current_player == 1 else n2, g)
            assert mask[mv] == 1
            assert g.make_move(mv // 9, mv % 9)
    assert "mcts" not in sys.modules


def test_paquete_evolution_no_importa_mcts():
    for f in (ROOT / "evolution").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any("mcts" in n or "trainer" in n for n in names), f


def test_partida_determinista():
    torch.manual_seed(1)
    a, b = SuperTatetiNet().eval(), SuperTatetiNet().eval()
    assert engine.play_match(a, b).moves == engine.play_match(a, b).moves


def test_modelos_guardan_cargan_y_no_sobrescriben(tmp_path):
    sd = SuperTatetiNet().state_dict()
    p = tmp_path / "ab12cd.pth"
    digest = modelstore.save_model(p, sd, {"color": "#ab12cd"})
    assert modelstore.load_state_dict(p, digest).keys() == sd.keys()
    assert modelstore.weights_hash(modelstore.load_state_dict(p)) == digest
    net = SuperTatetiNet(); net.load_state_dict(torch.load(p, weights_only=True))  # compatible con play.py
    with pytest.raises(modelstore.ModelExistsError):
        modelstore.save_model(p, sd, {})
    with pytest.raises(modelstore.ModelIntegrityError):
        modelstore.load_state_dict(p, "0" * 64)
    import json
    info = json.loads(modelstore.sidecar_path(p).read_text())
    assert info["architecture"] == "SuperTatetiNet" and info["format_version"] == 1


def test_config_probabilidades_por_entorno():
    cfg = load_config({"MUTATION_PROBS": "small:0.2,medium:0.3,large:0.5", "MAX_SEASONS": "3"})
    assert cfg.level_probs["large"] == 0.5 and cfg.max_seasons == 3
    with pytest.raises(ValueError):
        load_config({"MUTATION_PROBS": "small:0.2,medium:0.3,large:0.1"})