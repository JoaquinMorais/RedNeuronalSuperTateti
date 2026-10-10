import numpy as np
import math
import torch
from game import SuperTateti

C_PUCT = 1.4

class MCTSNode:
    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent  = parent
        self.move    = move
        self.prior   = prior
        self.game    = None
        self.children  = {}
        self.N = 0
        self.W = 0.0
        self._expanded = False

    def is_leaf(self):
        return not self._expanded

    def ucb_score(self, parent_N):
        if self.N == 0:
            return C_PUCT * self.prior * math.sqrt(parent_N + 1e-8)
        q = self.W / self.N
        u = C_PUCT * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return q + u

    def ensure_game(self):
        if self.game is None:
            board_idx, cell_idx = self.move
            self.game = self.parent.game.clone()
            self.game.make_move(board_idx, cell_idx)
        return self.game

    def expand(self, policy, valid_moves):
        self._expanded = True
        for move_flat in range(81):
            if valid_moves[move_flat] == 1:
                self.children[move_flat] = MCTSNode(
                    parent = self,
                    move   = (move_flat // 9, move_flat % 9),
                    prior  = float(policy[move_flat]),
                )

    def best_child(self):
        return max(self.children.values(), key=lambda c: c.ucb_score(self.N))

    def backpropagate(self, value):
        self.N += 1
        self.W += value
        if self.parent:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, net, simulations=200, batch_size=8):
        self.net         = net
        self.simulations = simulations
        self.cache       = {}

    def _evaluate(self, game):
        key = game.board.tobytes() + game.small_boards.tobytes() + bytes([game.current_player > 0])
        if key not in self.cache:
            state = torch.FloatTensor(game.get_state()).unsqueeze(0)
            mask  = torch.FloatTensor(game.get_valid_moves()).unsqueeze(0)
            with torch.no_grad():
                policy, value = self.net(state, mask)
            self.cache[key] = (policy.squeeze(0).numpy(), value.item())
        return self.cache[key]

    def search(self, game):
        self.cache = {}
        root = MCTSNode()
        root.game = game.clone()

        # Expandir raíz primero para poder agregar ruido
        policy_np, _ = self._evaluate(root.game)
        valid_moves  = root.game.get_valid_moves()
        root.expand(policy_np, valid_moves)

        # Ruido de Dirichlet en los priors de la raíz
        valid_children = list(root.children.values())
        noise = np.random.dirichlet([0.3] * len(valid_children))
        for child, n in zip(valid_children, noise):
            child.prior = 0.75 * child.prior + 0.25 * n

        for _ in range(self.simulations):
            node = root

            # SELECT
            while not node.is_leaf():
                node = node.best_child()
                node.ensure_game()
                if node.game.winner is not None:
                    break

            # TERMINAL
            if node.game is not None and node.game.winner is not None:
                value = 0.0 if node.game.winner == 0 else -1.0
                node.backpropagate(value)
                continue

            # EXPAND + EVALUATE
            policy_np, value_f = self._evaluate(node.game)
            valid_moves = node.game.get_valid_moves()
            node.expand(policy_np, valid_moves)

            # Mezclar valor de red con heurística
            heuristic = -node.game.heuristic_value()
            combined  = 0.5 * value_f + 0.5 * heuristic
            node.backpropagate(combined)

        visit_counts = np.zeros(81, dtype=np.float32)
        for move_flat, child in root.children.items():
            visit_counts[move_flat] = child.N

        total = visit_counts.sum()
        if total > 0:
            visit_counts /= total

        self._last_root = root
        return visit_counts