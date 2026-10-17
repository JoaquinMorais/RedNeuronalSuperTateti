import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture import MLP_HIDDEN


def hidden_from_state_dict(sd: dict) -> tuple:
    """Deriva las dimensiones ocultas del MLP a partir de un ``state_dict``.

    Lee, en orden, la ``out_features`` de cada capa Linear de ``shared`` (índices
    pares: ``shared.0``, ``shared.2``, ...). Permite reconstruir la red con su tamaño
    real al cargar un checkpoint, sin depender del default de ``architecture.py``.
    """
    dims = []
    i = 0
    while f"shared.{i}.weight" in sd:
        dims.append(int(sd[f"shared.{i}.weight"].shape[0]))
        i += 2
    if not dims:
        raise ValueError("state_dict sin capas 'shared.*': no se pudo inferir la arquitectura del MLP")
    return tuple(dims)


def _canonicalize(x: torch.Tensor) -> torch.Tensor:
    """Reescribe el estado desde la perspectiva de quien mueve (mis fichas = +1).

    ``x[:, 0:81]`` es el tablero ABSOLUTO (X=+1, O=-1) y ``x[:, 81:90]`` el resultado de cada
    tablero chico (1 / -1 / 0 = empatado / 0.5 = en juego). ``x[:, 99]`` es ``turno/81``: mueve X
    con turno par. Si mueve O, se multiplica tablero y tableros ganados por -1 para que la red vea
    siempre "mis fichas = +1"; el 0.5 de "en juego" no se toca. Así un patrón táctico se aprende una
    sola vez en vez de una por color.

    No permuta índices (la jugada sigue siendo la misma casilla absoluta), por eso la máscara de
    movimientos válidos sigue alineada. Ver ``trainPuzzles/canon.py``.
    """
    turn = torch.round(x[:, 99] * 81.0).to(torch.int64)
    odd = (turn % 2).to(x.dtype).unsqueeze(1)     # 0.0 si mueve X, 1.0 si mueve O (mismo device)
    me = 1.0 - 2.0 * odd
    out = x.clone()
    out[:, 0:81] = x[:, 0:81] * me
    won = x[:, 81:90]
    out[:, 81:90] = torch.where(won.abs() == 1.0, won * me, won)
    return out


class SuperTatetiNet(nn.Module):
    def __init__(self, hidden=MLP_HIDDEN, dropout: float = 0.0):
        super().__init__()

        # Dropout de entrenamiento (0 = desactivado). Se aplica en `forward` de forma
        # FUNCIONAL (no se agregan módulos `nn.Dropout`) para que el `state_dict` siga
        # teniendo exactamente las mismas claves y formas: no rompe checkpoints ni la
        # firma de arquitectura de `evolution.modelstore`.
        self.dropout_p = float(dropout)

        # Dimensiones de las capas ocultas (fuente única: `architecture.MLP_HIDDEN`).
        self.hidden = tuple(int(h) for h in hidden)

        # Capas compartidas: una Linear + ReLU por cada dimensión de `hidden`.
        layers = []
        prev = 100
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.shared = nn.Sequential(*layers)

        # Rama de política
        self.policy_head = nn.Linear(prev, 81)

        # Rama de valor
        self.value_head = nn.Sequential(
            nn.Linear(prev, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
    
    def forward(self, x, valid_moves_mask=None):
        """
        x:                tensor de forma (batch, 100) (estado crudo de ``game.get_state()``)
        valid_moves_mask: tensor de forma (batch, 81) con 1=válido, 0=inválido
        
        La entrada se canonicaliza (perspectiva de quien mueve) antes de la red: el encoding
        canónico vive acá y por eso todos los que juegan pasan el estado crudo sin transformarlo.

        Retorna:
          policy: probabilidades de cada movimiento (batch, 81)
          value:  evaluación de la posición       (batch, 1)
        """
        x = _canonicalize(x)
        shared = x
        for m in self.shared:
            shared = m(shared)
            if self.dropout_p > 0.0:
                shared = F.dropout(shared, p=self.dropout_p, training=self.training)
        
        # Rama de valor
        value = self.value_head(shared)
        
        # Rama de política con máscara de movimientos válidos
        policy_logits = self.policy_head(shared)
        
        if valid_moves_mask is not None:
            # Poner -infinito en los movimientos inválidos
            # así el softmax les asigna probabilidad 0
            policy_logits = policy_logits.masked_fill(valid_moves_mask == 0, float('-inf'))
        
        policy = F.softmax(policy_logits, dim=-1)
        
        return policy, value
    


if __name__ == "__main__":
    import numpy as np
    from game import SuperTateti
    
    game = SuperTateti()
    
    # Convertir estado a tensor
    state = torch.FloatTensor(game.get_state()).unsqueeze(0)  # (1, 100)
    mask  = torch.FloatTensor(game.get_valid_moves()).unsqueeze(0)  # (1, 81)
    
    net = SuperTatetiNet()
    policy, value = net(state, mask)
    
    print("Policy shape:", policy.shape)   # (1, 81)
    print("Policy sum:",   policy.sum().item())  # debe ser ~1.0
    print("Value shape:",  value.shape)    # (1, 1)
    print("Value:",        value.item())   # número entre -1 y 1
    
    # Verificar que los movimientos inválidos tienen prob 0
    # Al inicio todos los 81 son válidos así que todos > 0
    print("Movimientos con prob > 0:", (policy > 0).sum().item())  # 81
    
    # Simular un turno y verificar que la máscara funciona
    game.make_move(4, 0)  # activo = tablero 0, 8 casillas libres
    state2 = torch.FloatTensor(game.get_state()).unsqueeze(0)
    mask2  = torch.FloatTensor(game.get_valid_moves()).unsqueeze(0)
    policy2, _ = net(state2, mask2)
    print("Movimientos válidos después de turno 1:", (policy2 > 0).sum().item())  # debe ser 9