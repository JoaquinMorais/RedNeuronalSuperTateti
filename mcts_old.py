import numpy as np
import math
from game import SuperTateti
import copy

C_PUCT = 1.4  # constante de exploración, 1.4 es un buen valor inicial

class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior=0.0):
        self.game   = game        # estado del juego en este nodo
        self.parent = parent      # nodo padre
        self.move   = move        # movimiento que llevó a este nodo (board_idx, cell_idx)
        self.prior  = prior       # probabilidad que la red asignó a este movimiento

        self.children = {}        # move → MCTSNode
        self.N = 0                # veces visitado
        self.W = 0.0              # victorias acumuladas

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self, parent_N):
        """
        UCB1: balance entre explotar (W/N) y explorar (prior * sqrt(N_padre) / N+1)
        """
        if self.N == 0:
            return float('inf')  # nodos no visitados tienen prioridad máxima
        q = self.W / self.N
        u = C_PUCT * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return q + u

    def best_child(self):
        """Elige el hijo con mayor UCB score."""
        return max(
            self.children.values(),
            key=lambda child: child.ucb_score(self.N)
        )

    def expand(self, policy):
        """
        Crea los nodos hijos para todos los movimientos válidos.
        policy: array de 81 probabilidades de la red neuronal
        """
        valid_moves = self.game.get_valid_moves()
        for move_flat in range(81):
            if valid_moves[move_flat] == 1:
                board_idx = move_flat // 9
                cell_idx  = move_flat  % 9

                new_game = copy.deepcopy(self.game)
                new_game.make_move(board_idx, cell_idx)

                self.children[move_flat] = MCTSNode(
                    game   = new_game,
                    parent = self,
                    move   = (board_idx, cell_idx),
                    prior  = policy[move_flat],
                )

    def backpropagate(self, value):
        """
        Sube el resultado por el árbol hasta la raíz.
        value: 1 si ganó el jugador que hizo el movimiento, -1 si perdió.
        """
        self.N += 1
        self.W += value
        if self.parent:
            # invertir el valor porque el padre es el jugador contrario
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, net, simulations=50, batch_size = None):
        self.net         = net
        self.simulations = simulations

    def search(self, game):
        """
        Corre N simulaciones desde el estado actual.
        Retorna las probabilidades de visita para cada movimiento (81 valores).
        """
        import torch
        root = MCTSNode(game=copy.deepcopy(game))

        for _ in range(self.simulations):
            node = root

            # 1. SELECT: bajar hasta una hoja
            while not node.is_leaf() and node.game.winner is None:
                node = node.best_child()

            # 2. Si el juego terminó en esta hoja, backprop con el resultado real
            if node.game.winner is not None:
                if node.game.winner == 0:
                    value = 0.0
                else:
                    # el value es desde la perspectiva del jugador que VA A MOVER
                    # como el juego ya terminó, el jugador actual perdió
                    # (el que ganó fue el que hizo el último movimiento)
                    value = -1.0
                node.backpropagate(value)
                continue

            # 3. EXPAND: pedir política a la red y crear hijos
            state = torch.FloatTensor(node.game.get_state()).unsqueeze(0)
            mask  = torch.FloatTensor(node.game.get_valid_moves()).unsqueeze(0)

            with torch.no_grad():
                policy, value = self.net(state, mask)

            policy_np = policy.squeeze(0).numpy()
            value_f   = value.item()

            node.expand(policy_np)

            # 4. BACKPROP: subir el valor estimado por la red
            node.backpropagate(value_f)

        # Calcular probabilidades de visita de los hijos de la raíz
        visit_counts = np.zeros(81, dtype=np.float32)
        for move_flat, child in root.children.items():
            visit_counts[move_flat] = child.N

        # Normalizar
        total = visit_counts.sum()
        if total > 0:
            visit_counts /= total

        return visit_counts