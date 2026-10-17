"""Regresión del fallo de madrugada: pesos que crecen por generación -> NaN -> 'La red eligió un movimiento ilegal: 0'."""
import random
from pathlib import Path

import pytest
import torch

from evolution import additions, engine, modelstore, mutation
from evolution.config import DEFAULT_LEVEL_PROBS, DEFAULT_MUTATION_LEVELS, load_config
from game import SuperTateti
from model import SuperTatetiNet, hidden_from_state_dict
from tests.helpers import ROOT, TEST_URL, Env

BASE = modelstore.load_state_dict(ROOT / "model.pth")
LEVELS = DEFAULT_MUTATION_LEVELS


def _net(sd=None, double=False):
    if sd is not None:
        try:
            n = SuperTatetiNet(hidden=hidden_from_state_dict(sd))
        except ValueError:
            n = SuperTatetiNet()
    else:
        n = SuperTatetiNet()
    n = n.eval()
    if sd is not None:
        n.load_state_dict(sd)
    if double:
        n = n.double()
    for p in n.parameters():
        p.requires_grad_(False)
    return n


def _positions(k=120, seed=0):
    rng, out = random.Random(seed), []
    while len(out) < k:
        g = SuperTateti()
        for _ in range(rng.randrange(0, 40)):
            if g.winner is not None:
                break
            idx = [i for i in range(81) if g.get_valid_moves()[i] == 1]
            i = rng.choice(idx)
            g.make_move(i // 9, i % 9)
        if g.winner is None:
            out.append(g)
    return out


POSITIONS = _positions()


def _moves64(sd):
    """Referencia en float64 (no desborda hasta 1e308)."""
    net = _net({k: v.double() for k, v in sd.items()}, double=True)
    out = []
    for g in POSITIONS:
        st = torch.from_numpy(g.get_state()).double().unsqueeze(0)
        mk = torch.from_numpy(g.get_valid_moves()).double().unsqueeze(0)
        pol, _ = net(st, mk)
        out.append(int(torch.argmax(pol.masked_fill(mk == 0, -1.0), -1)))
    return out


def _moves32(sd):
    net = _net(sd)
    return [engine.select_move(net, g) for g in POSITIONS]


# ====================== el motor nunca se cae ======================
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_red_con_valores_no_finitos_juega_legal_sin_caerse(bad):
    sd = {k: torch.full_like(v, bad) for k, v in SuperTatetiNet().state_dict().items()}
    net = _net(sd)
    assert not engine.net_is_finite(net)
    for g in POSITIONS[:40]:
        mv = engine.select_move(net, g)
        assert g.get_valid_moves()[mv] == 1
        first = next(i for i in range(81) if g.get_valid_moves()[i] == 1)
        assert mv == first                                           # respaldo determinista: primer movimiento legal
    out = engine.play_match(net, _net(BASE))                         # una partida completa, sin excepciones
    assert out.result in ("x", "o", "draw") and out.moves == engine.play_match(net, _net(BASE)).moves


def test_redes_sanas_juegan_exactamente_igual_que_antes():
    for seed in range(4):
        torch.manual_seed(seed)
        net = _net()
        for g in POSITIONS[:60]:
            st = torch.from_numpy(g.get_state()).unsqueeze(0)
            mk = torch.from_numpy(g.get_valid_moves()).unsqueeze(0)
            with torch.no_grad():
                pol, _ = net(st, mk)
            assert engine.select_move(net, g) == int(torch.argmax(pol, -1))   # el cálculo anterior
    assert engine.net_is_finite(_net(BASE))


# ====================== reescalado que conserva la jugada ======================
def _layer_scaled(sd, m):
    """Escalado EXACTO por capas: W*=m, sesgo*=escala acumulada, cabeza de valor compensada."""
    out, S = dict(sd), 1.0
    for n in ("shared.0", "shared.2", "shared.4", "policy_head"):
        S *= m
        out[f"{n}.weight"] = sd[f"{n}.weight"].double() * m
        out[f"{n}.bias"] = sd[f"{n}.bias"].double() * S
        if n == "shared.4":
            out["value_head.0.weight"] = sd["value_head.0.weight"].double() / S
    return {k: (v.float() if torch.is_floating_point(v) else v) for k, v in out.items()}


def test_reescalado_conserva_la_jugada():
    ref = _moves64(BASE)
    assert _moves32(BASE) == ref
    for m in (1e3, 1e8):
        big = _layer_scaled(BASE, m)
        assert _moves32(big) == ref                                   # red grande pero finita: misma jugada
        norm, did = mutation.normalize_state_dict(big, 4.0)
        assert did and _moves32(norm) == ref                          # reescalada: misma jugada
        for n, fan_in in (("shared.0", 100), ("shared.2", 256), ("shared.4", 256), ("policy_head", 128)):
            assert float(norm[f"{n}.weight"].std()) == pytest.approx(fan_in ** -0.5, rel=1e-3)


@pytest.mark.parametrize("c", [30.0, 1e6, 1e10])
def test_reescalado_de_redes_como_las_del_linaje(c):
    """Redes con todos los parámetros x c (el crecimiento real): la reescalada coincide con la grande en float64."""
    grown = {k: (v * c if torch.is_floating_point(v) else v) for k, v in BASE.items()}
    norm, did = mutation.normalize_state_dict(grown, 4.0)
    assert did and _moves32(norm) == _moves64(grown) and engine.net_is_finite(_net(norm))
    if c >= 1e10:
        assert not engine.net_is_finite(_net(grown))                  # esta SÍ desborda float32 (el caso del fallo)


def test_reescalado_no_toca_la_cabeza_de_valor():
    """La cabeza de valor no interviene en la jugada: no se modifica (compensarla la haría crecer sin límite)."""
    grown = {k: (v * 1e6 if torch.is_floating_point(v) else v) for k, v in BASE.items()}
    norm, did = mutation.normalize_state_dict(grown, 4.0)
    assert did and all(torch.equal(norm[k], grown[k]) for k in grown if k.startswith("value_head."))


def test_redes_sanas_no_se_tocan_y_umbral_cero_desactiva():
    same, did = mutation.normalize_state_dict(BASE, 4.0)
    assert not did and all(torch.equal(same[k], BASE[k]) for k in BASE)
    grown = {k: (v * 1e6 if torch.is_floating_point(v) else v) for k, v in BASE.items()}
    off, did = mutation.normalize_state_dict(grown, 0.0)
    assert not did and all(torch.equal(off[k], grown[k]) for k in grown)
    a = mutation.mutate_state_dict(BASE, LEVELS["medium"], 5)
    b = mutation.mutate_state_dict(BASE, LEVELS["medium"], 5, False, 0.0)
    assert modelstore.weights_hash(a) == modelstore.weights_hash(b)  # comportamiento anterior intacto por defecto


def test_hijo_de_un_padre_desbordado_es_sano_y_el_padre_no_cambia():
    grown = {k: (v * 1e10 if torch.is_floating_point(v) else v) for k, v in BASE.items()}
    h = modelstore.weights_hash(grown)
    assert not engine.net_is_finite(_net(grown))
    child = mutation.mutate_state_dict(grown, LEVELS["small"], 7, False, 4.0)
    assert engine.net_is_finite(_net(child)) and modelstore.weights_hash(grown) == h
    assert max(float(v.abs().max()) for k, v in child.items() if not k.startswith("value_head.")) < 100   # lo que se juega


def test_padre_con_pesos_no_finitos_se_sanea():
    broken = {k: v.clone() for k, v in BASE.items()}
    broken["shared.2.weight"][0, :5] = float("nan")
    broken["shared.0.bias"][3] = float("inf")
    child = mutation.mutate_state_dict(broken, LEVELS["small"], 1, False, 4.0)
    assert all(torch.isfinite(v).all() for v in child.values()) and engine.net_is_finite(_net(child))


# ====================== linajes largos ======================
def _chain(threshold, gens, seed=0):
    rng, sd = random.Random(seed), BASE
    net = _net(sd)
    g0 = SuperTateti()
    st, mk = torch.from_numpy(g0.get_state()).unsqueeze(0), torch.from_numpy(g0.get_valid_moves()).unsqueeze(0)
    for gen in range(1, gens + 1):
        lv = mutation.pick_level(DEFAULT_LEVEL_PROBS, rng)
        sd = mutation.mutate_state_dict(sd, LEVELS[lv], rng.getrandbits(40), False, threshold)
        if gen % 50 == 0:
            net.load_state_dict(sd)
            with torch.no_grad():
                pol, _ = net(st, mk)
            if not torch.isfinite(pol).all():
                return gen, sd
    return None, sd


def test_el_bug_original_reproducido_y_corregido():
    gen, _ = _chain(0.0, 1700)                      # sin reescalado (comportamiento anterior): NaN hacia la generación ~1350
    assert gen is not None and 1000 < gen < 1700
    gen, sd = _chain(4.0, 1800)                     # con reescalado: sin NaN y pesos acotados
    assert gen is None and max(float(v.abs().max()) for v in sd.values()) < 10


# ====================== configuración y registro ======================
def test_configuracion_del_umbral():
    assert load_config({}).mutation_norm_threshold == 4.0
    assert load_config({"MUTATION_NORM_THRESHOLD": "0"}).mutation_norm_threshold == 0.0
    with pytest.raises(ValueError):
        load_config({"MUTATION_NORM_THRESHOLD": "-1"})


# ====================== de punta a punta ======================
def test_temporada_con_una_red_nan_no_se_cae_y_los_hijos_nuevos_registran_el_umbral(tmp_path):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    e = Env(tmp_path)
    try:
        nan_sd = {k: torch.full_like(v, float("nan")) for k, v in SuperTatetiNet().state_dict().items()}
        torch.save(nan_sd, tmp_path / "rota.pth")
        db = e.db(); db.connect_with_retries(); db.init_schema()
        additions.register_additions(db, e.cfg(), [str(tmp_path / "rota.pth")]); db.close()
        e.trainer(max_seasons=1).run()                                           # antes: IllegalMoveError a mitad de temporada
        txt = e.out.getvalue()
        assert "AVISO: 1 IA tienen pesos tan grandes" in txt and "FIN DE LA TEMPORADA 1" in txt
        assert e.q("SELECT count(*) c FROM matches WHERE status<>'finished'")[0]["c"] == 0
        broken = e.q("SELECT p.final_position FROM season_participants p JOIN individuals i ON i.id=p.individual_id WHERE i.kind='imported'")[0]
        assert broken["final_position"] >= 8                                     # la red rota no llega al top 7
        kids = e.q("SELECT mutation_meta FROM individuals WHERE kind='mutant'")
        assert kids and all(k["mutation_meta"]["norm_threshold"] == 4.0 for k in kids)
    finally:
        e.close()


def test_hijos_anteriores_a_la_correccion_se_regeneran_igual(tmp_path):
    """Una fila sin 'norm_threshold' (creada antes del arreglo) se reproduce con el algoritmo anterior."""
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL")
    from evolution import repo
    e = Env(tmp_path)
    try:
        e.trainer(max_seasons=1).run()
        kid = e.q("""SELECT i.id, i.model_file, i.mutation_level, i.mutation_seed, i.mutation_meta, p.model_file AS parent_file,
                            p.model_sha256 AS parent_sha FROM individuals i JOIN individuals p ON p.id = i.parent_id
                     WHERE i.kind='mutant' AND i.model_pruned_at IS NULL LIMIT 1""")[0]
        meta = dict(kid["mutation_meta"]); meta.pop("norm_threshold")
        expected = modelstore.weights_hash(mutation.mutate_state_dict(
            modelstore.load_state_dict(e.models / kid["parent_file"], kid["parent_sha"]),
            LEVELS[kid["mutation_level"]].__class__(meta["rate"], meta["sigma"]), kid["mutation_seed"], meta["mutate_value_head"]))
        (e.models / kid["model_file"]).unlink(); (e.models / kid["model_file"]).with_suffix(".json").unlink()
        e.admin.execute(f'SET search_path TO "{e.schema}"')
        e.admin.execute("UPDATE individuals SET model_sha256=NULL, mutation_meta=%s::jsonb WHERE id=%s",
                        (__import__("json").dumps(meta), kid["id"]))
        db = e.db(); db.connect()
        repo.materialize_models(db, e.cfg()); db.close()
        assert e.q("SELECT model_sha256 FROM individuals WHERE id=%s", (kid["id"],))[0]["model_sha256"] == expected
    finally:
        e.close()