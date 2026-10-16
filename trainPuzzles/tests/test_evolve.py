"""Pruebas del entrenador evolutivo de puzzles (torch, sin base de datos)."""
import math
import random

import pytest
import torch

from model import SuperTatetiNet
from trainPuzzles import evo_config as C
from trainPuzzles import evolve_puzzles as E
from trainPuzzles import train as T
from trainPuzzles.model_cnn import SuperTatetiCNN


def _net_sd():
    return SuperTatetiNet().state_dict()


def _cnn_sd():
    return SuperTatetiCNN().state_dict()


class _P:
    """Puzzle falso: ``sample_eval``/``bootstrap`` solo necesitan ``key``."""

    def __init__(self, key):
        self.key = key


# ---------------------- configuración ----------------------
def test_children_per_rank_reparte_el_resto_arriba():
    assert C.children_per_rank(10, 20) == (2,) * 10
    assert C.children_per_rank(3, 4) == (2, 1, 1)
    assert sum(C.children_per_rank(7, 13)) == 13


def test_config_defaults_y_policy():
    cfg = C.EvoConfig()
    assert (cfg.population, cfg.survivors, cfg.children) == (30, 10, 20)
    assert C.policy_tuple(cfg) == (2,) * 10
    C.validate(cfg)                                            # no lanza


def test_config_arch_valida_y_roundtrip():
    assert C.EvoConfig().arch == "mlp"
    with pytest.raises(ValueError):
        C.validate(C.EvoConfig(arch="rnn"))
    cfg = C.EvoConfig(arch="cnn")
    back = C.from_dict(cfg.snapshot())
    assert back.arch == "cnn" and back.out is None


def test_parse_args_arch_cnn():
    cfg = E.parse_args(["--arch", "cnn"])
    assert cfg.arch == "cnn" and cfg.out is None


def test_config_valida_invariantes():
    with pytest.raises(ValueError):
        C.validate(C.EvoConfig(population=5, survivors=6))
    with pytest.raises(ValueError):
        C.validate(C.EvoConfig(mutation_probs={"small": 0.5, "medium": 0.5, "large": 0.5}))
    with pytest.raises(ValueError):
        C.validate(C.EvoConfig(parent_policy=(1, 1)))          # no suma los 20 hijos


def test_config_valida_campos_nuevos():
    with pytest.raises(ValueError):
        C.validate(C.EvoConfig(eval_pool_size=10))             # < eval_size + early_stop_size
    with pytest.raises(ValueError):
        C.validate(C.EvoConfig(min_large_frac=1.5))
    with pytest.raises(ValueError):
        C.validate(C.EvoConfig(patience=-1))


def test_config_round_trip_json(tmp_path):
    cfg = C.EvoConfig(population=8, survivors=3, generations=2, levels=(1, 2), seed=7)
    path = tmp_path / "evo.json"
    C.save_config(cfg, path)
    back = C.load_config(path)
    assert back.snapshot() == cfg.snapshot()


# ---------------------- mutación ----------------------
def test_mutate_state_dict_no_toca_al_padre_y_conserva_formas():
    parent = _net_sd()
    before = {k: v.clone() for k, v in parent.items()}
    child = E.mutate_state_dict(parent, C.MutationLevel(rate=1.0, sigma=0.5), seed=1)
    assert set(child) == set(parent)
    assert all(child[k].shape == parent[k].shape for k in parent)
    assert all(torch.equal(parent[k], before[k]) for k in parent)     # padre intacto
    # la cabeza de valor no muta por defecto
    assert torch.equal(child["value_head.0.weight"], before["value_head.0.weight"])
    assert any(not torch.equal(child[k], before[k]) for k in parent if not k.startswith("value_head"))


def test_normalize_state_dict_deja_normales_y_reescala_explosivos():
    sd = _net_sd()
    out, changed = E.normalize_state_dict(sd, threshold=4.0)
    assert changed is False and out is sd

    huge = {k: v.clone() for k, v in sd.items()}
    huge["shared.0.weight"] = huge["shared.0.weight"] * 1e4
    out2, changed2 = E.normalize_state_dict(huge, threshold=4.0)
    assert changed2 is True
    assert torch.isfinite(out2["shared.0.weight"]).all()
    assert float(out2["shared.0.weight"].std()) < 1.0          # volvió a escala razonable


def test_normalize_state_dict_cnn_no_crashea_y_reescala():
    sd = _cnn_sd()
    out, changed = E.normalize_state_dict(sd, threshold=4.0, arch="cnn")
    assert changed is False and out is sd

    huge = {k: v.clone() for k, v in sd.items()}
    huge["conv1.weight"] = huge["conv1.weight"] * 1e4
    out2, changed2 = E.normalize_state_dict(huge, threshold=4.0, arch="cnn")
    assert changed2 is True
    assert torch.isfinite(out2["conv1.weight"]).all()
    assert float(out2["conv1.weight"].std()) < 1.0


def test_mutate_state_dict_cnn_conserva_formas_y_padre():
    parent = _cnn_sd()
    before = {k: v.clone() for k, v in parent.items()}
    child = E.mutate_state_dict(parent, C.MutationLevel(rate=1.0, sigma=0.5), seed=1,
                                norm_threshold=4.0, arch="cnn")
    assert set(child) == set(parent)
    assert all(child[k].shape == before[k].shape for k in parent)
    assert all(torch.equal(parent[k], before[k]) for k in parent)     # padre intacto
    # la cabeza de valor del CNN (value_fc1/value_fc2) no muta por defecto
    assert torch.equal(child["value_fc1.weight"], before["value_fc1.weight"])
    assert torch.equal(child["value_fc2.bias"], before["value_fc2.bias"])
    assert any(not torch.equal(child[k], before[k]) for k in parent if not k.startswith("value"))


def test_initial_population_cnn_completa_sin_crashear():
    cfg = C.EvoConfig(population=4, survivors=2, arch="cnn")
    sd = _cnn_sd()
    pop = E.initial_population(cfg, sd, random.Random(0))
    assert len(pop) == 4
    assert all(set(ind.sd) == set(sd) for ind in pop)


def test_pick_level_respeta_probabilidades():
    rng = random.Random(0)
    assert E.pick_level({"small": 1.0, "medium": 0.0, "large": 0.0}, rng) == "small"
    assert E.pick_level({"small": 0.0, "medium": 0.0, "large": 1.0}, rng) == "large"


# ---------------------- población ----------------------
def test_initial_population_tamano_y_semilla_intacta():
    cfg = C.EvoConfig(population=6, survivors=2)
    sd = _net_sd()
    pop = E.initial_population(cfg, sd, random.Random(0))
    assert len(pop) == 6
    assert pop[0].id == 0 and pop[0].level == "seed"
    assert all(torch.equal(pop[0].sd[k], sd[k]) for k in sd)


def test_next_generation_conserva_supervivientes_y_suma_poblacion():
    cfg = C.EvoConfig(population=6, survivors=2, epochs_per_gen=0)
    pop = E.initial_population(cfg, _net_sd(), random.Random(0))
    for i, ind in enumerate(pop):
        ind.fitness = float(i)
    ranked = sorted(pop, key=lambda x: x.fitness, reverse=True)
    nxt = E.next_generation(ranked, cfg, gen=2, rng=random.Random(1))
    assert len(nxt) == cfg.population
    ids = {p.id for p in nxt}
    assert {ranked[0].id, ranked[1].id} <= ids                 # eliten intactos
    assert len(ids) == cfg.population                          # ids únicos
    assert all(p.gen == 2 for p in nxt)


def test_next_generation_respeta_min_large_frac():
    cfg = C.EvoConfig(population=10, survivors=5, min_large_frac=0.6, epochs_per_gen=0)
    pop = E.initial_population(cfg, _net_sd(), random.Random(0))
    for i, ind in enumerate(pop):
        ind.fitness = float(i)
    ranked = sorted(pop, key=lambda x: x.fitness, reverse=True)
    nxt = E.next_generation(ranked, cfg, gen=2, rng=random.Random(3))
    children = nxt[cfg.survivors:]
    assert len(children) == cfg.children
    n_large = sum(1 for p in children if p.level == "large")
    assert n_large >= math.ceil(cfg.min_large_frac * len(children))


# ---------------------- validación rotativa ----------------------
def test_sample_eval_tamano_sin_repetidos_y_cap():
    pool = {1: [_P(k) for k in range(20)]}
    got = E.sample_eval(pool, 5, random.Random(0))
    assert len(got[1]) == 5
    assert len({p.key for p in got[1]}) == 5                   # sin repetidos dentro del set
    cap = E.sample_eval({1: [_P(0), _P(1)]}, 50, random.Random(0))
    assert len(cap[1]) == 2                                    # no supera el pool
    assert E.sample_eval(pool, 0, random.Random(0)) == {}      # size <= 0


def test_sample_eval_rota_y_evita_excluidos():
    pool = {1: [_P(k) for k in range(40)]}
    rng = random.Random(0)
    seen = set()
    for _ in range(8):
        seen |= {p.key for p in E.sample_eval(pool, 5, rng)[1]}
    assert len(seen) > 5                                       # cambia entre llamadas

    first = E.sample_eval(pool, 5, rng)
    excl = {p.key for p in first[1]}
    second = E.sample_eval(pool, 5, rng, exclude=set(excl))
    assert not (excl & {p.key for p in second[1]})             # disjunto de lo excluido


# ---------------------- búsqueda local / bucle ----------------------
def test_local_search_sin_epochs_devuelve_acierto_valido():
    cfg = C.EvoConfig(epochs_per_gen=0, levels=(1,))
    from trainPuzzles.tests.test_train import mate_in_1_position
    from trainPuzzles.puzzles import make_puzzle
    p = make_puzzle(mate_in_1_position(), 1, [2 * 9 + 2])
    sd, acc, ep = E.local_search(_net_sd(), cfg, [p, p], {1: [p]}, random.Random(0))
    assert ep == 0 and 0.0 <= acc <= 1.0
    assert set(sd) == set(_net_sd())


def test_run_una_generacion_minima(tmp_path):
    seed_path = tmp_path / "seed.pth"
    T.save_net(SuperTatetiNet(), seed_path)
    out = tmp_path / "best.pth"
    out_test = tmp_path / "best_test.pth"
    report = tmp_path / "evo_report.json"
    cfg = C.EvoConfig(
        seed_model=seed_path, out=out, out_test=out_test, report=report,
        population=3, survivors=1, generations=1, epochs_per_gen=0,
        levels=(1,), eval_pool_size=6, eval_size=2, early_stop_size=2,
        test_size=2, puzzles_per_level=3,
        games_budget=8000, test_every=1, seed=0,
    )
    rc = E.run(cfg)
    assert rc == 0
    assert out.exists() and report.exists() and out_test.exists()
    data = __import__("json").loads(report.read_text(encoding="utf-8"))
    assert data["generations_run"] == 1
    assert 0.0 <= data["best_fitness"] <= 1.0
    assert "test_accuracy" in data
    assert data["best_test_id"] is not None
