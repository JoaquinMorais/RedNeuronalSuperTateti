import torch
import numpy as np
from game import SuperTateti
from model import SuperTatetiNet
from mcts import MCTS

def print_board(game):
    """Imprime el tablero de forma legible."""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    
    print("\n   " + "  |  ".join([f"  {i}  " for i in range(3)]))
    print("   " + "-----+-------+-----")
    
    for big_row in range(3):
        for small_row in range(3):
            row_str = ""
            for big_col in range(3):
                big_idx = big_row * 3 + big_col
                
                # Si el tablero pequeño está terminado, mostrarlo diferente
                if game.small_boards[big_idx] == 1:
                    if small_row == 1:
                        row_str += " [  X  ] "
                    else:
                        row_str += " [     ] "
                elif game.small_boards[big_idx] == -1:
                    if small_row == 1:
                        row_str += " [  O  ] "
                    else:
                        row_str += " [     ] "
                elif game.small_boards[big_idx] == 0:
                    if small_row == 1:
                        row_str += " [ --- ] "
                    else:
                        row_str += " [     ] "
                else:
                    # Tablero en juego: mostrar casillas
                    cells = ""
                    for small_col in range(3):
                        cell_idx = small_row * 3 + small_col
                        val = game.board[big_idx][cell_idx]
                        cells += symbols[int(val)] + " "
                    
                    # Resaltar si es el tablero activo
                    if game.active_board == big_idx or game.active_board is None:
                        row_str += f"*{cells.strip()}*"
                    else:
                        row_str += f" {cells.strip()} "
                
                if big_col < 2:
                    row_str += " | "
            
            print(f"   {row_str}")
        
        if big_row < 2:
            print("   " + "---------+---------+---------")
    
    print()
    
    # Mostrar tablero grande resumido
    print("Estado tableros grandes:")
    for i in range(9):
        s = game.small_boards[i]
        if s == 1:   sym = 'X'
        elif s == -1: sym = 'O'
        elif s == 0:  sym = '='
        else:         sym = '.'
        print(sym, end=' ')
        if (i+1) % 3 == 0:
            print()
    print()

def get_human_move(game):
    """Pide al humano que ingrese su movimiento."""
    valid = game.get_valid_moves()
    
    if game.active_board is not None:
        print(f"Debés jugar en el tablero {game.active_board}")
    else:
        print("Podés jugar en cualquier tablero")
    
    while True:
        try:
            if not game.active_board:
                board_idx = int(input("Tablero (0-8): "))
            else:
                board_idx = game.active_board
            cell_idx  = int(input("Casilla (0-8): "))
            
            move_flat = board_idx * 9 + cell_idx
            
            if valid[move_flat] == 1:
                return board_idx, cell_idx
            else:
                print("Movimiento inválido, intentá de nuevo")
        except (ValueError, IndexError):
            print("Ingresá números entre 0 y 8")

def play():
    # Cargar modelo
    net = SuperTatetiNet()
    try:
        net.load_state_dict(torch.load("model.pth", weights_only=True))
        net.eval()
        print("Modelo cargado correctamente")
    except FileNotFoundError:
        print("No se encontró model.pth, usando red sin entrenar")
    
    mcts = MCTS(net, simulations=50)
    
    # Elegir lado
    print("\n¿Querés jugar como X (primero) u O (segundo)?")
    lado = input("X/O: ").strip().upper()
    human_player = 1 if lado == 'X' else -1
    
    game = SuperTateti()
    
    print("\nTablero: los tableros activos están marcados con *")
    print("Las casillas se numeran así:")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    
    while game.winner is None:
        print_board(game)
        
        if game.current_player == human_player:
            print("=== TU TURNO ===")
            board_idx, cell_idx = get_human_move(game)
        else:
            print("=== TURNO DE LA IA ===")
            probs = mcts.search(game)
            move_flat = np.argmax(probs)  # la IA elige el mejor movimiento
            board_idx = move_flat // 9
            cell_idx  = move_flat  % 9
            print(f"La IA juega en tablero {board_idx}, casilla {cell_idx}")
        
        game.make_move(board_idx, cell_idx)
    
    print_board(game)
    
    if game.winner == 0:
        print("Empate!")
    elif game.winner == human_player:
        print("Ganaste!")
    else:
        print("Ganó la IA!")

if __name__ == "__main__":
    play()