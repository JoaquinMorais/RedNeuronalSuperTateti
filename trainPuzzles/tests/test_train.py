"""Pruebas del entrenador por puzzles (torch, sin base de datos)."""
import random

import numpy as np
import torch

from evolution import modelstore
from model import SuperTatetiNet
from trainPuzzles import layout as L
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


def test_parse_args_acepta_start_level_y_stop_at_level():
    args = T.parse_args(["--start-level", "2", "--stop-at-level", "2"])
    assert args.start_level == 2
    assert args.stop_at_level == 2
    assert T.parse_args([]).start_level == 1
    assert T.parse_args([]).stop_at_level == 0


def test_scheduled_levels_normaliza_start_y_levels():
    assert T.scheduled_levels(T.parse_args([])) == (1, 4)
    assert T.scheduled_levels(T.parse_args(["--levels", "3"])) == (1, 3)
    assert T.scheduled_levels(T.parse_args(["--start-level", "2"])) == (2, 4)
    # --levels menor que start-level se eleva para no dejar afuera el nivel inicial
    assert T.scheduled_levels(T.parse_args(["--start-level", "3", "--levels", "1"])) == (3, 3)


def test_build_report_incluye_start_level(tmp_path):
    args = T.parse_args(["--start-level", "2"])
    report = T.build_report(args, tmp_path / "in.pth", tmp_path / "out.pth",
                            tmp_path / "r.json", [], args.threshold)
    assert report["start_level"] == 2


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


def test_train_level_domina_con_eval_every_mayor_a_uno():
    # Regresión: con --eval-every > 1 la racha de dominio debe acumularse entre
    # epochs evaluados y no reiniciarse en los no evaluados (antes nunca dominaba).
    g = mate_in_1_position()
    all_moves = tuple(int(m) for m in g.get_valid_moves().nonzero()[0])
    p = make_puzzle(g, 1, all_moves)
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, seed=0)
    args = T.parse_args(["--max-epochs", "20", "--min-epochs", "1",
                         "--patience", "2", "--eval-every", "2", "--threshold", "0.9"])
    res = T.train_level(net, trainer, args, 1, [p, p], [p], [p], 0.0, 0.9)
    assert res["mastered"] is True
    assert res["capped"] is False
    assert res["epochs"] <= 20


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


def test_train_level_loguea_cada_n_epochs(capsys):
    g = mate_in_1_position()
    all_moves = tuple(int(m) for m in g.get_valid_moves().nonzero()[0])
    p = make_puzzle(g, 1, all_moves)              # acierto siempre 1.0: sin mejoras nuevas
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, seed=0)
    args = T.parse_args(["--max-epochs", "30", "--min-epochs", "99", "--log-every", "10"])
    T.train_level(net, trainer, args, 1, [p, p], [p], [p], 0.0, 0.95)
    assert capsys.readouterr().out.count("N1 ep") == 4        # epoch 1 (mejor) + 10, 20 y 30


def test_accuracy_por_lotes_coincide_con_una_a_una_y_usa_entrada_canonica():
    from trainPuzzles.tests.test_canon import sample_mates
    ones, _ = sample_mates(6, 0)
    ps = [make_puzzle(g, 1, P.MateSolver().optimal_moves(g, 1)) for g in ones]
    torch.manual_seed(0)
    net = SuperTatetiNet().eval()
    hits = 0
    for p in ps:
        s = torch.from_numpy(p.state).unsqueeze(0)          # estado crudo: forward canonicaliza
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


def test_dropout_no_cambia_el_state_dict():
    a = SuperTatetiNet(dropout=0.0)
    b = SuperTatetiNet(dropout=0.2)
    assert set(a.state_dict()) == set(b.state_dict())
    assert all(a.state_dict()[k].shape == b.state_dict()[k].shape for k in a.state_dict())


def test_dropout_en_eval_es_determinista():
    torch.manual_seed(0)
    net = SuperTatetiNet(dropout=0.5).eval()
    p = make_puzzle(mate_in_1_position(), 1, [2 * 9 + 2])
    s = torch.from_numpy(p.state).unsqueeze(0)
    m = torch.from_numpy(p.mask).unsqueeze(0)
    with torch.no_grad():
        pa, _ = net(s, m)
        pb, _ = net(s, m)
    assert torch.equal(pa, pb)


def test_puzzle_trainer_aplica_weight_decay():
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, lr=1e-2, seed=0, weight_decay=1e-4)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_train_level_plateau_stop_corta_el_nivel():
    g = mate_in_1_position()
    all_moves = tuple(int(m) for m in g.get_valid_moves().nonzero()[0])
    p = make_puzzle(g, 1, all_moves)              # acierto siempre 1.0: nunca mejora
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, seed=0)
    args = T.parse_args(["--min-epochs", "99", "--max-epochs", "0", "--plateau-stop", "2"])
    res = T.train_level(net, trainer, args, 1, [p, p], [p], [p], 0.0, 0.95)
    assert res["plateau_stopped"] is True
    assert res["mastered"] is False
    assert res["epochs"] == 3


def test_eval_seed_por_defecto_igual_a_seed():
    a = T.parse_args(["--seed", "5"])
    assert T._eval_seed(a) == 5
    b = T.parse_args(["--seed", "5", "--eval-seed", "99"])
    assert T._eval_seed(b) == 99


# ---------------------- técnicas de refuerzo de entrenamiento ----------------------
def _puzzles_mate1(n=4):
    g = mate_in_1_position()
    all_moves = tuple(int(m) for m in g.get_valid_moves().nonzero()[0])
    return [make_puzzle(g, 1, all_moves) for _ in range(n)]


def test_topk_accuracy_agrega_las_k():
    ps = _puzzles_mate1(5)
    torch.manual_seed(0)
    net = SuperTatetiNet().eval()
    tk = T.topk_accuracy(net, ps, ks=(1, 2, 3))
    assert set(tk.keys()) == {1, 2, 3}
    assert all(0.0 <= v <= 1.0 for v in tk.values())
    assert tk[1] <= tk[2] <= tk[3]


def test_accuracy_con_tta_corre():
    ps = _puzzles_mate1(4)
    torch.manual_seed(0)
    net = SuperTatetiNet().eval()
    a = T.accuracy(net, ps, tta=True)
    assert 0.0 <= a <= 1.0
    tk = T.topk_accuracy(net, ps, ks=(1, 2), tta=True)
    assert all(0.0 <= v <= 1.0 for v in tk.values())


def test_tta_mapea_la_politica_de_vuelta_con_IDX81():
    from trainPuzzles import canon
    o = 40
    pol_old = np.zeros((1, 81), np.float32)
    pol_old[0, o] = 1.0
    states = np.zeros((1, 100), np.float32)
    masks = np.ones((1, 81), np.float32)
    for k in range(len(canon.SYM_MAPS)):
        _, _, pol_new = canon.apply_sym(states, masks, pol_old, k)
        p_back = np.zeros_like(pol_new)
        p_back[:, canon.IDX81[k]] = pol_new
        assert int(np.argmax(p_back[0])) == o


def test_per_sample_correct_da_un_bool_por_puzzle():
    ps = _puzzles_mate1(3)
    torch.manual_seed(0)
    net = SuperTatetiNet().eval()
    c = T.per_sample_correct(net, ps)
    assert c.shape == (len(ps),)
    assert c.dtype == bool


def test_compute_weights_solo_actua_fuera_del_warmup():
    ps = _puzzles_mate1(4)
    net = SuperTatetiNet().eval()
    w_warm = T.compute_weights(net, ps, boost=3.0, warmup_epoch=5, current_epoch=2, mine_every=1)
    assert all(v == 1.0 for v in w_warm)
    w_on = T.compute_weights(net, ps, boost=3.0, warmup_epoch=0, current_epoch=1, mine_every=1)
    assert all(v >= 1.0 for v in w_on)


def test_ema_mantiene_la_misma_estructura_que_el_state_dict():
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, lr=1e-3, seed=0, ema_decay=0.9)
    assert set(trainer.ema_state) == set(net.state_dict())
    assert all(trainer.ema_state[k].shape == net.state_dict()[k].shape for k in net.state_dict())


def test_lr_cosine_decae():
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, lr=1e-3, seed=0, lr_schedule="cosine",
                              lr_final_frac=0.01, lr_decay_epochs=10)
    assert trainer.scheduler is not None
    l0 = trainer.optimizer.param_groups[0]["lr"]
    for _ in range(3):
        trainer.step_scheduler()
    assert trainer.optimizer.param_groups[0]["lr"] < l0


def test_lr_schedule_none_no_crea_scheduler():
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, lr=1e-3, seed=0, lr_schedule="none")
    assert trainer.scheduler is None
    trainer.step_scheduler()   # no debe fallar


def test_ranking_loss_corre_y_es_finita():
    ps = _puzzles_mate1(4)
    torch.manual_seed(0)
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, lr=1e-3, seed=0, augment=False,
                              rank_coef=0.5, rank_margin=1.0)
    loss, ploss, vloss = trainer.train_epoch(ps, batch_size=2)
    assert loss == loss and ploss == ploss


def test_train_epoch_con_pesos_upsamplea_dificiles():
    ps = _puzzles_mate1(6)
    torch.manual_seed(0)
    net = SuperTatetiNet()
    trainer = T.PuzzleTrainer(net, lr=1e-3, seed=0, augment=False)
    w = np.zeros(len(ps)); w[0] = 1.0        # solo el primero tiene masa
    trainer.train_epoch(ps, batch_size=3, weights=w)   # no debe fallar y usa solo el índice 0


# ---------------------- layout de checkpoints ----------------------
def test_layout_defaults_init_y_out_apuntan_al_canon():
    args = T.parse_args([])
    canon_pth, canon_report = L.canon_paths(T.Path(args.out))
    assert T.Path(args.init) == canon_pth
    assert T.Path(args.out) == canon_pth
    assert canon_pth.name == L.CANON_PTH
    assert canon_report.name == L.CANON_REPORT


def test_layout_next_run_dir_incremental_y_tag(tmp_path):
    assert L.next_run_dir(tmp_path / "puzzle_canon.pth", "puzzle").name == "0001_puzzle"
    (tmp_path / L.RUNS_DIRNAME / "0001_puzzle").mkdir(parents=True)
    (tmp_path / L.RUNS_DIRNAME / "0002_otro").mkdir(parents=True)
    assert L.next_run_dir(tmp_path / "puzzle_canon.pth", "retry1").name == "0003_retry1"


def test_layout_sanitize_tag_y_run_paths(tmp_path):
    assert L.sanitize_tag("retry 1/2") == "retry-1-2"
    assert L.sanitize_tag("") == "puzzle"
    run_dir = tmp_path / "runs" / "0001_puzzle"
    final, report, levels = L.run_paths(run_dir, "puzzle")
    assert final == run_dir / "puzzle.pth"
    assert report == run_dir / "puzzle_report.json"
    assert levels == run_dir / "levels"


def test_layout_canon_deriva_del_directorio_de_out(tmp_path):
    canon_pth, canon_report = L.canon_paths(tmp_path / "sub" / "algo.pth")
    assert canon_pth == tmp_path / "sub" / L.CANON_PTH
    assert canon_report == tmp_path / "sub" / L.CANON_REPORT