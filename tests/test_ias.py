import torch
import numpy as np

from game import SuperTateti
from model import SuperTatetiNet, hidden_from_state_dict
from mcts import MCTS


def load_ai(model_path, simulations=50):
    state = torch.load(model_path, weights_only=True)
    try:
        net = SuperTatetiNet(hidden=hidden_from_state_dict(state))
    except ValueError:
        net = SuperTatetiNet()
    net.load_state_dict(state)
    net.eval()

    return MCTS(net, simulations=simulations)


def play_game(ai_x, ai_o):
    game = SuperTateti()

    while game.winner is None:

        if game.current_player == 1:
            probs = ai_x.search(game)
        else:
            probs = ai_o.search(game)


        temperature = 0.25

        probs = probs ** (1 / temperature)
        probs /= probs.sum()

        move_flat = np.random.choice(
            81,
            p=probs
        )   

        #np.argmax(probs)

        board_idx = move_flat // 9
        cell_idx  = move_flat % 9

        game.make_move(board_idx, cell_idx)

    return game.winner


def evaluate(model_a, model_b, games=10):
    print(f'Modelo A {model_a}')
    print(f'Modelo B {model_b}')
    ai_a = load_ai(model_a)
    ai_b = load_ai(model_b)

    wins_a = 0
    wins_b = 0
    draws  = 0

    for i in range(games):

        # alternar colores
        if i % 2 == 0:
            winner = play_game(ai_a, ai_b)
            #print(winner)
            if winner == 1:
                wins_a += 1
            elif winner == -1:
                wins_b += 1
            else:
                draws += 1

        else:
            winner = play_game(ai_b, ai_a)

            if winner == 1:
                wins_b += 1
            elif winner == -1:
                wins_a += 1
            else:
                draws += 1

        if (i + 1) % 50 == 0:
            print(
                f"{i+1}/{games} | "
                f"A:{wins_a} "
                f"B:{wins_b} "
                f"D:{draws}"
            )

    print()
    print("RESULTADO FINAL")
    print("A:", wins_a)
    print("B:", wins_b)
    print("Empates:", draws)

    total = wins_a + wins_b

    if total > 0:
        print("Winrate A:", wins_a / total * 100)
        print("Winrate B:", wins_b / total * 100)


if __name__ == "__main__":

    evaluate(
        "model.pth",
        "models/model_easy.pth",
        #"training/9-6/checkpoint_100.pth",
        
        games=500
    )