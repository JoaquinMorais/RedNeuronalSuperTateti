import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from datetime import datetime
from game import SuperTateti
from model import SuperTatetiNet
from mcts import MCTS

os.makedirs("logs", exist_ok=True)
LOG_FILE = f"logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"

def log_print(text):
    msg = f"{datetime.now().strftime('%Y/%m/%d-%H:%M')} - {text}"
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def play_random_move(game):
    """Elige un movimiento completamente al azar."""
    valid = game.get_valid_moves()
    valid_indices = np.where(valid == 1)[0]
    move_flat = np.random.choice(valid_indices)
    return move_flat // 9, move_flat % 9


class Trainer:
    def __init__(self, simulations=200):
        self.net  = SuperTatetiNet()
        self.mcts = MCTS(self.net, simulations=simulations, batch_size=8)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0003)  # antes 0.001
        self.memory    = deque(maxlen=10000)                            # antes 5000

    # ── Fase 1: IA vs oponente random ─────────────────────────────
    def play_vs_random(self, ia_player=1):
        """
        La IA juega contra un oponente que mueve al azar.
        ia_player: 1 = IA juega como X, -1 = IA juega como O
        """
        game    = SuperTateti()
        history = []

        while game.winner is None:
            if game.current_player == ia_player:
                # IA decide con MCTS
                mcts_probs = self.mcts.search(game)
                move_flat  = np.random.choice(81, p=mcts_probs)
            else:
                # Oponente random
                mcts_probs = game.get_valid_moves()
                total = mcts_probs.sum()
                mcts_probs = mcts_probs / total  # normalizar para guardar
                valid = game.get_valid_moves()
                valid_indices = np.where(valid == 1)[0]
                move_flat = np.random.choice(valid_indices)

            board_idx = move_flat // 9
            cell_idx  = move_flat  % 9

            history.append((
                game.get_state(),
                game.get_valid_moves(),
                mcts_probs,
                game.current_player,
            ))

            game.make_move(board_idx, cell_idx)

        for state, mask, probs, player in history:
            if game.winner == 0:
                outcome = 0.0
            else:
                outcome = 1.0 if game.winner == player else -1.0
            self.memory.append((state, mask, probs, outcome))

        return game.winner

    # ── Fase 2: self-play normal ───────────────────────────────────
    def play_one_game(self):
        game    = SuperTateti()
        history = []

        while game.winner is None:
            mcts_probs = self.mcts.search(game)
            move_flat  = np.random.choice(81, p=mcts_probs)
            board_idx  = move_flat // 9
            cell_idx   = move_flat  % 9

            history.append((
                game.get_state(),
                game.get_valid_moves(),
                mcts_probs,
                game.current_player,
            ))
            game.make_move(board_idx, cell_idx)

        for state, mask, probs, player in history:
            if game.winner == 0:
                outcome = 0.0
            else:
                outcome = 1.0 if game.winner == player else -1.0
            self.memory.append((state, mask, probs, outcome))

        return game.winner

    # ── Entrenamiento ──────────────────────────────────────────────
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
        self.net.load_state_dict(torch.load(path, weights_only=True))
        log_print(f"Modelo cargado desde {path}")

    # ── Curriculum ─────────────────────────────────────────────────
    def train_curriculum(self, horas=6, carpeta=''):
        """
        Distribuye el tiempo disponible entre las dos fases:
        - Fase 1 (vs random): 40% del tiempo
        - Fase 2 (self-play): 60% del tiempo
        """
        import time

        segundos_totales = horas * 3600
        segundos_fase1   = int(segundos_totales * 0.4)
        segundos_fase2   = int(segundos_totales * 0.6)

        log_print(f"Tiempo total: {horas}hs")
        log_print(f"  Fase 1 (vs random):  {segundos_fase1/60:.0f} min")
        log_print(f"  Fase 2 (self-play):  {segundos_fase2/60:.0f} min")

        # ── FASE 1: vs random ──────────────────────────────────────────
        log_print("=== FASE 1: IA vs Random ===")
        self.memory.clear()
        iter_f1 = 0
        t_inicio = time.time()

        while time.time() - t_inicio < segundos_fase1:
            results = []
            for _ in range(5):
                ia_player = 1 if iter_f1 % 2 == 0 else -1
                winner = self.play_vs_random(ia_player=ia_player)
                results.append(winner)

            last_loss = None
            for _ in range(4):
                last_loss = self.train_step(batch_size=256)

            iter_f1 += 1
            transcurrido = time.time() - t_inicio
            restante     = max(0, segundos_fase1 - transcurrido)

            if last_loss:
                loss, vloss, ploss = last_loss
                wins_ia = results.count(1) + results.count(-1)
                draws   = results.count(0)
                log_print(
                    f"F1 Iter {iter_f1:>4} | "
                    f"IA:{wins_ia} ={draws} | "
                    f"Loss={loss:.4f} (V={vloss:.4f} P={ploss:.4f}) | "
                    f"Restante: {restante/60:.1f}min"
                )

            if iter_f1 % 50 == 0:
                self.save(f"{carpeta}/fase1_checkpoint_{iter_f1}.pth")

        self.save(f"{carpeta}/fase1_final.pth")
        self.memory.clear()
        log_print(f"Fase 1 completada en {iter_f1} iteraciones. Memoria limpiada.")

        # ── FASE 2: self-play ──────────────────────────────────────────
        log_print("=== FASE 2: Self-play ===")
        iter_f2  = 0
        t_inicio = time.time()

        while time.time() - t_inicio < segundos_fase2:
            results = []
            for _ in range(5):
                winner = self.play_one_game()
                results.append(winner)

            last_loss = None
            for _ in range(4):
                last_loss = self.train_step(batch_size=64)

            iter_f2 += 1
            transcurrido = time.time() - t_inicio
            restante     = max(0, segundos_fase2 - transcurrido)

            if last_loss:
                loss, vloss, ploss = last_loss
                wins_x = results.count(1)
                wins_o = results.count(-1)
                draws  = results.count(0)
                log_print(
                    f"F2 Iter {iter_f2:>4} | "
                    f"X:{wins_x} O:{wins_o} ={draws} | "
                    f"Loss={loss:.4f} (V={vloss:.4f} P={ploss:.4f}) | "
                    f"Restante: {restante/60:.1f}min"
                )

            if iter_f2 % 50 == 0:
                self.save(f"{carpeta}/fase2_checkpoint_{iter_f2}.pth")

        self.save()
        log_print(f"Fase 2 completada en {iter_f2} iteraciones.")
        log_print("Entrenamiento completado.")


if __name__ == "__main__":
    trainer = Trainer(simulations=200)

    if os.path.exists("model.pth"):
        trainer.load("model.pth")
        log_print("Continuando desde model.pth")
    else:
        log_print("Entrenando desde cero")

    import os
    from datetime import datetime

    fecha = datetime.now().strftime("%m-%d")

    version = 1
    while True:
        carpeta = f"training/{fecha}-v{version}"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
            break
        version += 1

    log_print(carpeta)

    trainer.train_curriculum(horas=16,carpeta = carpeta)  # ← cambiá este número