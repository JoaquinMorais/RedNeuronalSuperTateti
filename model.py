import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperTatetiNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Capas compartidas
        self.shared = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Rama de política
        self.policy_head = nn.Linear(128, 81)
        
        # Rama de valor
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
    
    def forward(self, x, valid_moves_mask=None):
        """
        x:                tensor de forma (batch, 100)
        valid_moves_mask: tensor de forma (batch, 81) con 1=válido, 0=inválido
        
        Retorna:
          policy: probabilidades de cada movimiento (batch, 81)
          value:  evaluación de la posición       (batch, 1)
        """
        shared = self.shared(x)
        
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