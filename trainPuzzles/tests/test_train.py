"""Pruebas del entrenador por puzzles (torch, sin base de datos)."""
import random

import torch

from evolution import modelstore
from model import SuperTatetiNet
from trainPuzzles import canon
from trainPuzzles import puzzles as P
from trainPuzzles import train as T
from trainPuzzles.puzzles import make_puzzle
from trainPuzzles.tests.test_puzzles import mate_in_1_position


def test_policy_target_uniforme_sobre_optimas():
    t = T.policy_target((0, 5, 8))
    assert abs(float(t.sum()) - 1.0) < 1e-6
    assert set(t.nonzero()[0].tolist()) == {0, 5, 8}
    assert all(abs(float(t[i]) - 1 / 3) < 1e-6 for i in (0, 5, 8))


def test_train_epoch_cambia_pesos_y_pierde_finito():
    g = mate_in_1_position()
    p = make_puzzle(g, 1, [2 * 9 + 2])
    torch.manual_seed(0)
    net = SuperTatetiNet()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    trainer = T.PuzzleTrainer(net, lr=1e-2, seed=0)
    loss, ploss, vloss = trainer.train_epoch([p, p, p, p], batch_size=2)
    assert all(x == x and abs(x) != float("inf") for x in (loss, ploss, vloss))
    assert any(not torch.equal(net.state_dict()[k], before[k]) for k in before)


def test_accuracy_es_1_si_toda_jugada_es_optima():
    g = mate_in_1_position()
    all_moves = tuple(int(m) for m in g.get_valid_moves().nonzero()[0])
    p = make_puzzle(g, 1, all_moves)
    net = SuperTatetiNet().eval()
    assert T.accuracy(net, [p, p]) == 1.0
    assert T.accuracy(net, []) == 0.0


def test_guardado_compatible_con_modelstore(tmp_path):
    net = SuperTatetiNet()
    out = tmp_path / "puzzle_net.pth"
    T.save_net(net, out)
    sd = modelstore.load_state_dict(out)                      # valida arquitectura + hash
    assert set(sd) == set(net.state_dict())
    loaded = modelstore.load_net(out)
    assert isinstance(loaded, SuperTatetiNet)


def test_classify_delta_las_cinco_categorias():
    big, small = 0.20, 0.05
    assert T.classify_delta(0.20, big, small) == T.IMPROVED_A_LOT
    assert T.classify_delta(0.50, big, small) == T.IMPROVED_A_LOT
    assert T.classify_delta(0.05, big, small) == T.IMPROVED_LITTLE
    assert T.classify_delta(0.19, big, small) == T.IMPROVED_LITTLE
    assert T.classify_delta(0.0, big, small) == T.NO_CLEAR
    assert T.classify_delta(0.049, big, small) == T.NO_CLEAR
    assert T.classify_delta(-0.049, big, small) == T.NO_CLEAR
    assert T.classify_delta(-0.05, big, small) == T.WORSENED_LITTLE
    assert T.classify_delta(-0.19, big, small) == T.WORSENED_LITTLE
    assert T.classify_delta(-0.20, big, small) == T.WORSENED_A_LOT
    assert T.classify_delta(-0.90, big, small) == T.WORSENED_A_LOT
    assert T.classify_delta(float("nan"), big, small) == "sin datos"


def test_mastery_step_exige_umbral_min_epochs_y_retencion():
    kwargs = dict(threshold=0.95, ret_threshold=0.90, min_epochs=15)
    # cumple todo: suma racha
    assert T.mastery_step(0.96, 0.91, epoch=15, streak=0, **kwargs) == 1
    assert T.mastery_step(0.96, 0.91, epoch=20, streak=1, **kwargs) == 2
    # por debajo del umbral del nivel
    assert T.mastery_step(0.94, 0.99, epoch=20, streak=3, **kwargs) == 0
    # retención insuficiente
    assert T.mastery_step(0.99, 0.89, epoch=20, streak=3, **kwargs) == 0
    # todavía no cumplió min-epochs
    assert T.mastery_step(0.99, 0.99, epoch=14, streak=3, **kwargs) == 0


def test_build_report_agrega_veredicto_y_promedio(tmp_path):
    args = T.parse_args(["--report", str(tmp_path / "r.json")])
    summaries = [
        {"level": 1, "baseline": 0.40, "final": 0.96, "delta": 0.56,
         "epochs": 18, "mastered": True, "reached": True, "best": 0.96, "best_epoch": 18},
        {"level": 2, "baseline": 0.30, "final": 0.32, "delta": 0.02,
         "epochs": 40, "mastered": False, "reached": True, "best": 0.32, "best_epoch": 40},
    ]
    report = T.build_report(args, tmp_path / "in.pth", tmp_path / "out.pth",
                            tmp_path / "r.json", summaries, args.threshold)
    assert report["levels_reached"] == 2 and report["levels_mastered"] == 1
    assert abs(report["avg_delta"] - 0.29) < 1e-9                  # (0.56 + 0.02) / 2
    assert report["verdict"] == T.IMPROVED_A_LOT                   # 0.29 >= big
    assert report["levels"][0]["verdict"] == T.IMPROVED_A_LOT
    assert report["levels"][1]["verdict"] == T.NO_CLEAR            # 0.02 dentro de +-0.05


def test_build_report_sin_datos(tmp_path):
    args = T.parse_args([])
    report = T.build_report(args, tmp_path / "in.pth", tmp_path / "out.pth",
                            tmp_path / "r.json", [], args.threshold)
    assert report["verdict"] == "sin datos"
    assert report["levels_reached"] == 0
    assert report["avg_delta"] != report["avg_delta"]           # NaN


def test_train_level_se_detiene_con_el_flag_de_parada():
    g = mate_in_1_position()
    p = make_puzzle(g, 1, [2 * 9 + 2])
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, seed=0)
    args = T.parse_args(["--max-epochs", "3"])
    T._STOP.set()
    try:
        res = T.train_level(net, trainer, args, 1, [p, p], [p], [p], 0.0, 0.95)
    finally:
        T._STOP.clear()
    assert res["stopped"] is True
    assert res["epochs"] == 0
    assert res["best_state"] is None


def test_train_level_respeta_tope_de_epochs():
    g = mate_in_1_position()
    p = make_puzzle(g, 1, [2 * 9 + 2])
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, seed=0)
    args = T.parse_args(["--max-epochs", "2", "--min-epochs", "99"])
    res = T.train_level(net, trainer, args, 1, [p, p], [p], [p], 0.0, 0.95)
    assert res["capped"] is True
    assert res["epochs"] == 2
    assert res["mastered"] is False


def test_revision_phase_reporta_retencion_por_nivel():
    g = mate_in_1_position()
    p = make_puzzle(g, 1, [2 * 9 + 2])
    net = SuperTatetiNet()
    args = T.parse_args(["--revision-epochs", "2", "--revision-puzzles", "10"])
    trainer = T.PuzzleTrainer(net, seed=0)
    rng = random.Random(0)
    ret = T.revision_phase(net, trainer, args, {1: [p, p]}, [1], {1: [p]}, rng)
    assert set(ret) == {1}
    assert 0.0 <= ret[1] <= 1.0


def test_revision_phase_sin_niveles_no_hace_nada():
    net = SuperTatetiNet()
    args = T.parse_args([])
    trainer = T.PuzzleTrainer(net, seed=0)
    rng = random.Random(0)
    assert T.revision_phase(net, trainer, args, {}, [], {}, rng) == {}


def test_log_phase_agrega_etiqueta(capsys):
    T.log_phase("[GEN]", "hola")
    assert "[GEN] hola" in capsys.readouterr().out


def test_generate_eval_set_loguea_gen(capsys):
    args = T.parse_args(["--eval-size", "2", "--games-budget", "300", "--levels", "1"])
    T.generate_eval_set(1, args, set())
    assert "[GEN]" in capsys.readouterr().out


def test_revision_phase_loguea_rev(capsys):
    g = mate_in_1_position()
    p = make_puzzle(g, 1, [2 * 9 + 2])
    net = SuperTatetiNet()
    args = T.parse_args(["--revision-epochs", "1", "--revision-puzzles", "4"])
    trainer = T.PuzzleTrainer(net, seed=0)
    T.revision_phase(net, trainer, args, {1: [p, p]}, [1], {1: [p]}, random.Random(0))
    assert "[REV]" in capsys.readouterr().out


def test_accuracy_por_lotes_coincide_con_una_a_una_y_usa_entrada_canonica():
    from trainPuzzles.tests.test_canon import sample_mates
    ones, _ = sample_mates(6, 0)
    ps = [make_puzzle(g, 1, P.MateSolver().optimal_moves(g, 1)) for g in ones]
    torch.manual_seed(0)
    net = SuperTatetiNet().eval()
    hits = 0
    for p in ps:
        s = torch.from_numpy(canon.canonical_state(p.state)).unsqueeze(0)
        m = torch.from_numpy(p.mask).unsqueeze(0)
        with torch.no_grad():
            pol, _ = net(s, m)
        mv = int(torch.argmax(pol.masked_fill(m == 0, -1.0), dim=-1).item())
        hits += int(mv in p.best_moves)
    assert T.accuracy(net, ps) == hits / len(ps)


def test_train_epoch_corre_con_y_sin_aumento_y_valor_apagado_por_defecto():
    p = make_puzzle(mate_in_1_position(), 1, [2 * 9 + 2])
    for aug in (True, False):
        torch.manual_seed(0)
        net = SuperTatetiNet()
        trainer = T.PuzzleTrainer(net, lr=1e-2, seed=0, augment=aug)
        loss, ploss, vloss = trainer.train_epoch([p, p, p, p], batch_size=2)
        assert ploss == ploss and abs(loss - ploss) < 1e-6        # value_coef = 0 por defecto