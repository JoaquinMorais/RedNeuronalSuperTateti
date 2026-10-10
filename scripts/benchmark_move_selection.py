"""Compara el tiempo por partida: MCTS (camino ANTIGUO) vs selección directa con la red (NUEVO).

Uso:  python scripts/benchmark_move_selection.py [partidas_mcts] [partidas_directas]
Este script es de medición: es el único lugar que importa ``mcts``; el entrenamiento nuevo no.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from evolution import engine, modelstore  # noqa: E402
from game import SuperTateti  # noqa: E402
from mcts import MCTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
net = modelstore.load_net(ROOT / "model.pth")
n_mcts = int(sys.argv[1]) if len(sys.argv) > 1 else 2
n_direct = int(sys.argv[2]) if len(sys.argv) > 2 else 50


def play_mcts(sims):
    m = MCTS(net, simulations=sims)
    g, moves, t0 = SuperTateti(), 0, time.perf_counter()
    while g.winner is None:
        mv = int(np.argmax(m.search(g)))
        g.make_move(mv // 9, mv % 9)
        moves += 1
    return time.perf_counter() - t0, moves


for sims in (50, 200):  # 50 = play.py/test_ias.py, 200 = trainer.py
    r = [play_mcts(sims) for _ in range(n_mcts)]
    t, mv = sum(x[0] for x in r), sum(x[1] for x in r)
    print(f"MCTS {sims:>3} sims : {t / n_mcts * 1000:9.1f} ms/partida  {t / mv * 1000:7.2f} ms/jugada  ({n_mcts} partidas)")

r = [engine.play_match(net, net) for _ in range(n_direct)]
t, mv = sum(x.duration_ms for x in r), sum(x.num_moves for x in r)
print(f"Directo (red)  : {t / n_direct:9.1f} ms/partida  {t / mv:7.2f} ms/jugada  ({n_direct} partidas)")
