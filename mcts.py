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

                new_game = self.game.clone()
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
    def __init__(self, net, simulations=50):
        self.net         = net
        self.simulations = simulations
        self.cache       = {}  # estado → (policy, value)

    def _evaluate(self, game):
        """Evalúa un estado con la red, usando caché si ya fue evaluado."""
        import torch
        
        state_key = game.board.tobytes() + game.small_boards.tobytes()
        
        if state_key not in self.cache:
            state = torch.FloatTensor(game.get_state()).unsqueeze(0)
            mask  = torch.FloatTensor(game.get_valid_moves()).unsqueeze(0)
            with torch.no_grad():
                policy, value = self.net(state, mask)
            self.cache[state_key] = (
                policy.squeeze(0).numpy(),
                value.item()
            )
        
        return self.cache[state_key]

    def search(self, game):
        # Limpiar caché entre búsquedas para no usar memoria infinita
        self.cache = {}
        
        root = MCTSNode(game=game.clone())

        for _ in range(self.simulations):
            node = root

            while not node.is_leaf() and node.game.winner is None:
                node = node.best_child()

            if node.game.winner is not None:
                value = 0.0 if node.game.winner == 0 else -1.0
                node.backpropagate(value)
                continue

            policy_np, value_f = self._evaluate(node.game)  # ← usa caché
            node.expand(policy_np)
            node.backpropagate(value_f)

        visit_counts = np.zeros(81, dtype=np.float32)
        for move_flat, child in root.children.items():
            visit_counts[move_flat] = child.N

        total = visit_counts.sum()
        if total > 0:
            visit_counts /= total

        return visit_counts
    

if __name__ == "__main__":
    import torch
    from model import SuperTatetiNet

    net  = SuperTatetiNet()
    mcts = MCTS(net, simulations=50)
    game = SuperTateti()

    print("Corriendo MCTS...")
    probs = mcts.search(game)

    print("Shape:", probs.shape)           # (81,)
    print("Sum:", probs.sum())             # ~1.0
    print("Movimientos con visitas > 0:", (probs > 0).sum())  # varios
    print("Movimiento más visitado:", probs.argmax())
    print("Probabilidad más alta:", probs.max())