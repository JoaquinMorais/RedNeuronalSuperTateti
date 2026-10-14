"""Pruebas del oráculo y del generador de puzzles (sin torch, sin base de datos)."""
import numpy as np

from game import SuperTateti
from trainPuzzles import puzzles as P


def mate_in_1_position() -> SuperTateti:
    """X (jugador 1) ganó los tableros 0 y 1; el tablero 2 tiene X en 0 y 1."""
    g = SuperTateti()
    g.small_boards[:] = [1, 1, 0.5, 0, 0, 0, 0, 0, 0]
    g.board[2][0] = 1
    g.board[2][1] = 1
    g.active_board = 2
    g.current_player = 1
    g.turn = 30
    g.winner = None
    return g


def test_mate_en_1_detectado_y_jugada_optima():
    g = mate_in_1_position()
    s = P.MateSolver()
    assert s.min_mate(g, 3) == 1
    assert s.can_win(g, 1) is True
    assert s.can_win(g, 0) is False
    best = s.optimal_moves(g, 1)
    assert (2 * 9 + 2) in best                       # jugar la esquina que cierra la línea
    assert all(g.get_valid_moves()[m] == 1 for m in best)


def test_mate_inmediato_coincide_con_simulacion():
    """La detección rápida debe coincidir con simular cada jugada legal."""
    g = mate_in_1_position()
    me = g.current_player
    slow = []
    for mv in P.legal_moves(g):
        child = g.clone()
        child.make_move(mv // 9, mv % 9)
        if child.winner == me:
            slow.append(mv)
    assert sorted(P.immediate_win_moves(g)) == sorted(slow) == [2 * 9 + 2]


def test_posicion_inicial_sin_mate():
    g = SuperTateti()
    s = P.MateSolver(max_nodes=100000)
    assert s.min_mate(g, 1) is None
    assert s.optimal_moves(g, 1) == []


def test_harvest_reproducible_y_legal():
    targets = {1: 3}
    a = P.harvest(targets, seed=7, max_n=1, games_budget=2000)
    b = P.harvest(targets, seed=7, max_n=1, games_budget=2000)
    assert [p.key for p in a[1]] == [p.key for p in b[1]]
    assert a[1], "el muestreo aleatorio debería encontrar algún mate en 1"
    for p in a[1]:
        assert p.level == 1 and p.best_moves
        assert p.state.shape == (100,) and p.mask.shape == (81,)
        assert all(p.mask[m] == 1 for m in p.best_moves)          # óptimas legales
        assert np.count_nonzero(p.mask) >= 1


def test_harvest_respeta_exclude():
    a = P.harvest({1: 3}, seed=11, max_n=1, games_budget=2000)
    keys = {p.key for p in a[1]}
    assert keys
    b = P.harvest({1: 3}, seed=99, max_n=1, games_budget=2000, exclude=keys)
    assert not (keys & {p.key for p in b[1]})


def test_harvest_respeta_should_stop():
    got = P.harvest({2: 50}, seed=5, max_n=2, games_budget=10000, min_turn=14,
                    solver_max_nodes=60000, should_stop=lambda: True)
    assert got[2] == []


def _game_from_puzzle(p):
    g = SuperTateti()
    board, small, player, active = p.key
    g.board = np.frombuffer(board, dtype=np.float32).reshape(9, 9).copy()
    g.small_boards = np.frombuffer(small, dtype=np.float32).copy()
    g.current_player = player
    g.active_board = active
    g.winner = None
    return g


def test_mate_en_2_consistente_con_el_solver():
    got = P.harvest({2: 3}, seed=5, max_n=2, games_budget=4000, min_turn=14,
                    solver_max_nodes=60000)
    ps = got[2]
    assert ps, "el muestreo aleatorio debería encontrar algún mate en 2"
    s = P.MateSolver(max_nodes=60000)
    for p in ps:
        g = _game_from_puzzle(p)
        s.reset()
        assert s.min_mate(g, 2) == 2
        assert s.can_win(g, 1) is False
        assert set(s.optimal_moves(g, 2)) == set(p.best_moves)

