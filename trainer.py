import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from game import SuperTateti
from model import SuperTatetiNet
from mcts import MCTS



import os
from datetime import datetime

# Crear carpeta logs si no existe
os.makedirs("logs", exist_ok=True)

# Nombre del archivo: log_2026-06-10_14-30.txt
LOG_FILE = f"logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"

def log_print(text):
    msg = f"{datetime.now().strftime('%Y/%m/%d-%H:%M')} - {text}"

    print(text)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


class Trainer:
    def __init__(self, simulations=50):
        self.net  = SuperTatetiNet()
        self.mcts = MCTS(self.net, simulations=simulations)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)

    def play_one_game(self):
        game    = SuperTateti()
        history = []

        while game.winner is None:
            # MCTS decide en vez de la red sola
            mcts_probs = self.mcts.search(game)

            # Guardar estado antes de mover
            history.append((
                game.get_state(),
                game.get_valid_moves(),
                mcts_probs,
                game.current_player,
            ))

            # Elegir movimiento sampleando de las probabilidades MCTS
            move_flat = np.random.choice(81, p=mcts_probs)
            board_idx = move_flat // 9
            cell_idx  = move_flat  % 9
            game.make_move(board_idx, cell_idx)

        # Asignar resultado
        for state, mask, probs, player in history:
            if game.winner == 0:
                outcome = 0.0
            else:
                outcome = 1.0 if game.winner == player else -1.0
            self.memory.append((state, mask, probs, outcome))

        return game.winner

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, masks, policies, outcomes = zip(*batch)

        states   = torch.FloatTensor(np.array(states))
        masks    = torch.FloatTensor(np.array(masks))
        policies = torch.FloatTensor(np.array(policies))
        outcomes = torch.FloatTensor(outcomes).unsqueeze(1)

        pred_policy, pred_value = self.net(states, masks)

        value_loss  = nn.MSELoss()(pred_value, outcomes)
        policy_loss = -torch.sum(policies * torch.log(pred_policy + 1e-8), dim=1).mean()
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), value_loss.item(), policy_loss.item()

    def save(self, path="model.pth"):
        torch.save(self.net.state_dict(), path)
        log_print(f"Modelo guardado en {path}")

    def load(self, path="model.pth"):
        self.net.load_state_dict(torch.load(path))
        log_print(f"Modelo cargado desde {path}")

    def train(self, iterations=100, games_per_iter=10, batch_size=64):
        for i in range(iterations):
            results = []
            for _ in range(games_per_iter):
                winner = self.play_one_game()
                results.append(winner)

            loss_info = self.train_step(batch_size)

            wins_x  = results.count(1)
            wins_o  = results.count(-1)
            draws   = results.count(0)

            if loss_info:
                loss, vloss, ploss = loss_info
                log_print(
                    f"Iter {i+1:>4}/{iterations} | "
                    f"X:{wins_x} O:{wins_o} ={draws} | "
                    f"Loss={loss:.4f} (V={vloss:.4f} P={ploss:.4f})"
                )
            else:
                log_print(f"Iter {i+1:>4}/{iterations} | Acumulando experiencias...")

            # Guardar cada 100 iteraciones
            if (i + 1) % 100 == 0:
                self.save(f"checkpoint_{i+1}.pth")

        self.save()


if __name__ == "__main__":
    trainer = Trainer(simulations=50)
    import os
    if os.path.exists("model.pth"):
        trainer.load("model.pth")
        log_print("Continuando entrenamiento desde model.pth")
    trainer.train(iterations=1920, games_per_iter=10, batch_size=64)