import numpy as np
import math
import torch
from game import SuperTateti

C_PUCT = 1.4

class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior=0.0):
        self.game   = game
        self.parent = parent
        self.move   = move
        self.prior  = prior

        self.children = {}  # move_flat → MCTSNode
        self.N = 0
        self.W = 0.0
        
        # Guardamos la política y los movimientos válidos
        # pero NO creamos los hijos todavía
        self._policy    = None
        self._expanded  = False

    def is_leaf(self):
        return not self._expanded

    def ucb_score(self, parent_N):
        if self.N == 0:
            return float('inf')
        q = self.W / self.N
        u = C_PUCT * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return q + u

    def best_child(self):
        return max(
            self.children.values(),
            key=lambda c: c.ucb_score(self.N)
        )

    def expand(self, policy, valid_moves):
        """
        Expansión lazy: solo registra qué movimientos son válidos
        y sus probabilidades. Los nodos hijos se crean al visitarlos.
        """
        self._policy   = policy
        self._expanded = True
        
        for move_flat in range(81):
            if valid_moves[move_flat] == 1:
                # Crear nodo hijo con juego lazy (sin clonar todavía)
                self.children[move_flat] = MCTSNode(
                    game   = None,  # se asigna al visitar
                    parent = self,
                    move   = (move_flat // 9, move_flat % 9),
                    prior  = policy[move_flat],
                )

    def get_or_create_game(self, move_flat):
        """Crea el estado del juego para este nodo solo cuando se necesita."""
        if self.game is None:
            # Recrear desde el padre
            board_idx, cell_idx = self.move
            self.game = self.parent.game.clone()
            self.game.make_move(board_idx, cell_idx)
        return self.game

    def backpropagate(self, value):
        self.N += 1
        self.W += value
        if self.parent:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, net, simulations=50, batch_size=8):
        self.net         = net
        self.simulations = simulations
        self.batch_size  = batch_size  # simulaciones en paralelo
        self.cache       = {}

    def _evaluate_batch(self, games):
        """Evalúa múltiples estados en una sola llamada a la red."""
        # Filtrar los que ya están en caché
        to_evaluate = []
        keys = []
        results = {}
        
        for game in games:
            key = game.board.tobytes() + game.small_boards.tobytes()
            keys.append(key)
            if key not in self.cache:
                to_evaluate.append((key, game))

        # Evaluar los que no están en caché en un solo batch
        if to_evaluate:
            states = torch.FloatTensor(np.array([
                g.get_state() for _, g in to_evaluate
            ]))
            masks = torch.FloatTensor(np.array([
                g.get_valid_moves() for _, g in to_evaluate
            ]))
            
            with torch.no_grad():
                policies, values = self.net(states, masks)
            
            for i, (key, _) in enumerate(to_evaluate):
                self.cache[key] = (
                    policies[i].numpy(),
                    values[i].item()
                )

        return [self.cache[k] for k in keys]

    def search(self, game):
        self.cache = {}
        root = MCTSNode(game=game.clone())

        # Correr simulaciones en grupos de batch_size
        sims_done = 0
        while sims_done < self.simulations:
            current_batch = min(self.batch_size, self.simulations - sims_done)
            
            # Fase SELECT: bajar por el árbol en cada simulación
            leaves = []
            for _ in range(current_batch):
                node = root
                while not node.is_leaf():
                    node = node.best_child()
                    node.get_or_create_game(
                        node.move[0] * 9 + node.move[1] if node.move else 0
                    )
                    if node.game and node.game.winner is not None:
                        break
                leaves.append(node)

            # Separar terminales de no terminales
            terminal_leaves     = [n for n in leaves if n.game and n.game.winner is not None]
            nonterminal_leaves  = [n for n in leaves if not (n.game and n.game.winner is not None)]

            # Backprop de terminales
            for node in terminal_leaves:
                value = 0.0 if node.game.winner == 0 else -1.0
                node.backpropagate(value)

            # Evaluar no terminales en batch
            if nonterminal_leaves:
                games_to_eval = []
                for node in nonterminal_leaves:
                    g = node.game if node.game else node.get_or_create_game(0)
                    games_to_eval.append(g)

                batch_results = self._evaluate_batch(games_to_eval)

                for node, (policy_np, value_f) in zip(nonterminal_leaves, batch_results):
                    g = node.game
                    valid_moves = g.get_valid_moves()
                    node.expand(policy_np, valid_moves)
                    node.backpropagate(value_f)

            sims_done += current_batch

        visit_counts = np.zeros(81, dtype=np.float32)
        for move_flat, child in root.children.items():
            visit_counts[move_flat] = child.N

        total = visit_counts.sum()
        if total > 0:
            visit_counts /= total

        return visit_counts
    

# Agregá esto temporalmente al final de mcts.py para medir
if __name__ == "__main__":
    import time
    from model import SuperTatetiNet

    net = SuperTatetiNet()
    game = SuperTateti()

    for sims in [50, 100, 200, 400]:
        mcts = MCTS(net, simulations=sims, batch_size=8)
        times = []
        for _ in range(3):
            start = time.time()
            probs = mcts.search(game)
            times.append(time.time() - start)
        avg = sum(times) / len(times)
        # Una partida dura ~50 turnos
        partida = avg * 50
        # Con 8hs disponibles
        partidas_8hs = int(8 * 3600 / partida)
        print(f"sims={sims:>4} | {avg:.3f}s/búsqueda | {partida:.1f}s/partida | {partidas_8hs} partidas en 8hs")