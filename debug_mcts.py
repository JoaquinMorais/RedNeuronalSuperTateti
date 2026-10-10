import torch
import numpy as np
from game import SuperTateti
from model import SuperTatetiNet
from mcts import MCTS

net  = SuperTatetiNet()
mcts = MCTS(net, simulations=200, batch_size=8)
game = SuperTateti()

print("=== DEBUG: qué hace el MCTS en cada turno ===")
for turno in range(5):
    probs = mcts.search(game)
    move_flat = np.argmax(probs)
    board_idx = move_flat // 9
    cell_idx  = move_flat  % 9

    # Ver distribución de visitas
    visitas = sorted(
        [(mf, c.N) for mf, c in mcts._last_root.children.items() if c.N > 0],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    print(f"\nTurno {turno+1}:")
    print(f"  Tablero activo: {game.active_board}")
    print(f"  Movimientos válidos: {int(game.get_valid_moves().sum())}")
    print(f"  Top 5 visitas (move_flat, N): {visitas}")
    print(f"  Total visitas: {sum(c.N for c in mcts._last_root.children.values())}")
    print(f"  Movimiento elegido: tablero {board_idx} casilla {cell_idx}")

    game.make_move(board_idx, cell_idx)