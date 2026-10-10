import torch
import numpy as np
from game import SuperTateti
from model import SuperTatetiNet
from mcts import MCTS

def play_vs_random(mcts, ia_player=1, verbose=False):
    game = SuperTateti()
    
    while game.winner is None:
        if game.current_player == ia_player:
            probs = mcts.search(game)
            move_flat = np.argmax(probs)  # determinista, elige el mejor
        else:
            valid = np.where(game.get_valid_moves() == 1)[0]
            move_flat = np.random.choice(valid)
        
        board_idx = move_flat // 9
        cell_idx  = move_flat  % 9
        
        if verbose:
            print(f"Jugador {game.current_player} juega tablero {board_idx} casilla {cell_idx}")
        
        game.make_move(board_idx, cell_idx)
    
    return game.winner

# Test 1: red sin entrenar vs random
print("=== Red sin entrenar vs Random (100 partidas) ===")
net  = SuperTatetiNet()
net.load_state_dict(torch.load('model.pth', weights_only=True))
mcts = MCTS(net, simulations=200, batch_size=8)

wins, losses, draws = 0, 0, 0
for i in range(100):
    ia_player = 1 if i % 2 == 0 else -1
    winner = play_vs_random(mcts, ia_player=ia_player)
    if winner == ia_player:
        wins += 1
    elif winner == 0:
        draws += 1
    else:
        losses += 1

print(f"Wins: {wins} | Losses: {losses} | Draws: {draws}")
print(f"Winrate: {wins/(wins+losses)*100:.1f}%")

# Test 2: ver qué tan concentradas están las probabilidades
print("\n=== Concentración de probabilidades MCTS ===")
game = SuperTateti()
probs = mcts.search(game)
valid = game.get_valid_moves()
valid_probs = probs[valid == 1]
print(f"Movimientos válidos: {int(valid.sum())}")
print(f"Prob máxima: {probs.max():.4f}")
print(f"Prob mínima (válidos): {valid_probs.min():.4f}")
print(f"Prob promedio (válidos): {valid_probs.mean():.4f}")
print(f"Top 3 movimientos: {np.argsort(probs)[-3:][::-1]}")