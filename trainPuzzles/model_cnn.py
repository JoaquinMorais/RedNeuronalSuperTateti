"""Versión CNN de ``SuperTatetiNet`` para entrenar puzzles de mate en N.

Existencia y alcance
--------------------
El MLP (``model.py``) trata las 81 casillas como features independientes: no tiene
sesgo espacial. Este modelo aislado explota que ``game.get_state()`` es, en esencia,
una imagen 9x9 (cada bloque 3x3 es un tablero chico) y aplica convoluciones sobre
esa grilla.

Compatibilidad
--------------
- **No** reemplaza a ``SuperTatetiNet``: vive solo en ``trainPuzzles/`` y se usa con el
  flag ``--arch cnn`` de ``trainPuzzles/train.py``.
- ``forward`` tiene la MISMA firma que ``SuperTatetiNet`` (``(B, 100)`` ->
  ``(policy(81), value(1))``) y el ``state_dict`` es estándar, así que todo el pipeline
  de entrenamiento (TTA, ranking, hard-mining, EMA, currículo) funciona sin cambios.
- La canonicalización se delega en ``model._canonicalize`` para que sea EXACTAMENTE la
  misma que la del MLP.

Encoding de entrada (confirmado en ``game.py:get_state``)
---------------------------------------------------------
- ``[0:81]``  tablero 9x9 aplanado (X=+1, O=-1, 0=vacío) -> imagen 9x9.
- ``[81:90]`` ``small_boards``: estado de cada tablero chico (1/-1/0/0.5) -> mapa 3x3.
- ``[90:99]`` tablero activo one-hot (todos 1 si hay libertad) -> mapa 3x3.
- ``[99]``    ``turno/81`` (escalar, se inyecta en la cabeza de valor).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import _canonicalize

ARCHITECTURE_NAME = "SuperTatetiCNN"

# Canales del cuerpo convolucional ("Media": ~132k params, comparable al MLP de ~143k).
# Para subir capacidad, p. ej. (64, 128, 128) (~240k).
DEFAULT_CHANNELS = (48, 96, 96)
IN_CHANNELS = 3  # ch0 tablero (con signo), ch1 small_boards, ch2 active_board


def _expand_macro(planes: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) -> (B, 9, 9): cada celda macro llena su bloque 3x3.

    La celda macro ``(r, c)`` ocupa las casillas ``3r..3r+2`` x ``3c..3c+2`` del
    tablero grande, de modo que el mapa queda alineado con ``ch0``.
    """
    b = planes.shape[0]
    return planes.view(b, 3, 1, 3, 1).expand(b, 3, 3, 3, 3).reshape(b, 9, 9)


class SuperTatetiCNN(nn.Module):
    def __init__(self, dropout: float = 0.0, channels: tuple[int, int, int] = DEFAULT_CHANNELS):
        super().__init__()

        # Dropout funcional (no agrega módulos) para no cambiar claves del state_dict.
        self.dropout_p = float(dropout)

        c1, c2, c3 = channels

        self.conv1 = nn.Conv2d(IN_CHANNELS, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)

        # Política: 1 logit por casilla, alineado con la grilla 9x9.
        self.policy_head = nn.Conv2d(c3, 1, kernel_size=1)

        # Valor: pool global + turno -> MLP -> tanh.
        self.value_fc1 = nn.Linear(c3 + 1, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Construye el tensor (B, 3, 9, 9) a partir del estado crudo (B, 100)."""
        board = x[:, 0:81].reshape(-1, 9, 9)
        small = _expand_macro(x[:, 81:90].reshape(-1, 3, 3))
        active = _expand_macro(x[:, 90:99].reshape(-1, 3, 3))
        return torch.stack((board, small, active), dim=1)

    def forward(self, x, valid_moves_mask=None):
        x = _canonicalize(x)
        feats = self._features(x)

        h = F.relu(self.conv1(feats))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        if self.dropout_p > 0.0:
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        # Política: (B, 1, 9, 9) -> (B, 81)
        policy_logits = self.policy_head(h).reshape(x.shape[0], 81)
        if valid_moves_mask is not None:
            policy_logits = policy_logits.masked_fill(valid_moves_mask == 0, float("-inf"))
        policy = F.softmax(policy_logits, dim=-1)

        # Valor: (B, C) + turno -> (B, 1)
        pooled = F.adaptive_avg_pool2d(h, 1).reshape(x.shape[0], -1)
        turn = x[:, 99:100]
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(torch.cat((pooled, turn), dim=1)))))

        return policy, value


if __name__ == "__main__":
    from game import SuperTateti

    game = SuperTateti()
    state = torch.FloatTensor(game.get_state()).unsqueeze(0)
    mask = torch.FloatTensor(game.get_valid_moves()).unsqueeze(0)

    net = SuperTatetiCNN()
    policy, value = net(state, mask)
    n_params = sum(p.numel() for p in net.parameters())

    print("Policy shape:", tuple(policy.shape))
    print("Policy sum:", float(policy.sum()))
    print("Value shape:", tuple(value.shape), "value:", float(value))
    print("Parámetros:", n_params)
