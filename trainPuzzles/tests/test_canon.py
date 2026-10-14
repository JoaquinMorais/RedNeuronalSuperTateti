"""Pruebas de la entrada canónica y las simetrías D4 (solo numpy; sin torch ni base de datos)."""
import random

import numpy as np

from game import SuperTateti
from trainPuzzles import canon
from trainPuzzles import puzzles as P
from trainPuzzles.tests.test_puzzles import mate_in_1_position


def sym_game(g: SuperTateti, k: int) -> SuperTateti:
    """La misma partida vista a través de la simetría ``k`` (tablero grande y casillas con la misma r)."""
    r, inv = canon.SYM_MAPS[k], canon.INV9[k]
    h = g.clone()
    h.board = np.array([[g.board[r[b]][r[c]] for c in range(9)] for b in range(9)], dtype=g.board.dtype)
    h.small_boards = np.array([g.small_boards[r[b]] for b in range(9)], dtype=g.small_boards.dtype)
    h.active_board = None if g.active_board is None else int(inv[g.active_board])
    h.squares_played = [int(inv[b]) for b in g.squares_played]
    return h


def sample_mates(n1: int, n2: int, seed: int = 3):
    """Posiciones reales (clones) con mate en 1 y mate en 2, por juego aleatorio."""
    rng, solver = random.Random(seed), P.MateSolver(max_nodes=60000)
    ones, twos = [], []
    for _ in range(4000):
        for g in P.play_random_game(rng):
            if g.turn < 16 or g.winner is not None:
                continue
            if len(ones) < n1 and P.immediate_win_moves(g):
                ones.append(g)
            elif len(twos) < n2:
                solver.reset()
                try:
                    if solver.min_mate(g, 2) == 2:
                        twos.append(g)
                except P.NodeBudgetExceeded:
                    pass
        if len(ones) >= n1 and len(twos) >= n2:
            break
    return ones, twos


def test_mueve_x_queda_igual_y_mueve_o_se_invierte():
    g = mate_in_1_position()                                  # turno 30 (par): mueve X
    s = g.get_state()
    assert np.array_equal(canon.canonical_state(s)[:99], s[:99])
    h = g.clone()
    h.board, h.small_boards, h.current_player, h.turn = -g.board, np.where(np.abs(g.small_boards) == 1, -g.small_boards, g.small_boards), -1, 31
    c = canon.canonical_state(h.get_state())
    assert np.array_equal(c[:99], canon.canonical_state(s)[:99])   # gemelo con colores cambiados -> misma entrada


def test_el_0_5_de_tablero_en_juego_no_cambia_de_signo():
    g = mate_in_1_position()
    g.turn = 31                                                # mueve O
    c = canon.canonical_state(g.get_state())
    assert c[81 + 2] == 0.5 and c[81 + 0] == -1.0 and c[81 + 3] == 0.0


def test_canonical_batch_no_toca_el_original():
    s = np.stack([mate_in_1_position().get_state()] * 3)
    s[1, 99] = 31 / 81
    before = s.copy()
    out = canon.canonical_batch(s)
    assert np.array_equal(s, before) and out.dtype == np.float32 and out.shape == (3, 100)


def test_las_8_simetrias_son_distintas_y_cierran_el_grupo():
    keys = {tuple(r) for r in canon.SYM_MAPS}
    assert len(keys) == 8 and tuple(range(9)) in keys


def test_simetria_coincide_con_el_juego_transformado():
    """apply_sym sobre (estado, máscara) == get_state/get_valid_moves de la partida transformada."""
    ones, _ = sample_mates(4, 0)
    for g in ones:
        for k in range(8):
            h = sym_game(g, k)
            s, m, _ = canon.apply_sym(g.get_state()[None, :], g.get_valid_moves()[None, :],
                                      np.zeros((1, 81), np.float32), k)
            assert np.array_equal(s[0][:99], h.get_state()[:99])
            assert np.array_equal(m[0], h.get_valid_moves())


def test_las_jugadas_optimas_se_transforman_con_la_simetria():
    """Prueba clave: el solver sobre la partida transformada da las jugadas optimas transformadas (mate 1 y 2)."""
    ones, twos = sample_mates(4, 3)
    assert ones and twos
    solver = P.MateSolver(max_nodes=0)
    for level, games in ((1, ones), (2, twos)):
        for g in games:
            solver.reset()
            best = solver.optimal_moves(g, level)
            assert best
            for k in range(8):
                inv = canon.INV9[k]
                expected = {int(inv[m // 9]) * 9 + int(inv[m % 9]) for m in best}
                solver.reset()
                assert set(solver.optimal_moves(sym_game(g, k), level)) == expected


def test_augment_conserva_objetivos_dentro_de_la_mascara_y_no_toca_el_original():
    ones, _ = sample_mates(6, 0)
    s = np.stack([g.get_state() for g in ones]); m = np.stack([g.get_valid_moves() for g in ones])
    t = np.zeros((len(ones), 81), np.float32)
    for i, g in enumerate(ones):
        best = P.MateSolver().optimal_moves(g, 1)
        t[i, best] = 1.0 / len(best)
    s0, m0, t0 = s.copy(), m.copy(), t.copy()
    s2, m2, t2 = canon.augment(s, m, t, np.random.default_rng(0))
    assert np.array_equal(s, s0) and np.array_equal(m, m0) and np.array_equal(t, t0)
    assert np.allclose(t2.sum(1), 1.0) and np.all(t2 <= m2)
    assert np.allclose(m2.sum(1), m.sum(1))