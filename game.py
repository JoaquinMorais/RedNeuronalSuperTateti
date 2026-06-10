import numpy as np

# Índices de las combinaciones ganadoras en un tablero 3x3
WIN_COMBOS = [
    (0,1,2), (3,4,5), (6,7,8),  # filas
    (0,3,6), (1,4,7), (2,5,8),  # columnas
    (0,4,8), (2,4,6),            # diagonales
]

class SuperTateti:
    def __init__(self):
        # Tablero principal: 9 tableros x 9 casillas
        # 1 = jugador actual, -1 = rival, 0 = vacío
        self.board = np.zeros((9, 9), dtype=np.float32)
        
        # Estado de cada tablero pequeño
        # 1 = ganado jugador actual, -1 = ganado rival
        # 0 = empatado, 0.5 = en juego
        self.small_boards = np.full(9, 0.5, dtype=np.float32)
        
        self.current_player = 1   # 1 o -1
        self.active_board = None  # None = cualquier tablero
        self.squares_played = []  # historial de tableros jugados
        self.turn = 0
        self.winner = None        # 1, -1, 0 (empate), None (en curso)
    
    def check_small_winner(self, board_idx):
        """
        Verifica si hay ganador en un tablero pequeño.
        Retorna: 1 (gana jugador actual), -1 (gana rival), 
                 0 (empate), 0.5 (sigue en juego)
        """
        board = self.board[board_idx]
        
        for a, b, c in WIN_COMBOS:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]  # 1 o -1
        
        # Si no hay ganador, verificar si está lleno (empate)
        if not np.any(board == 0):
            return 0  # empatado
        
        return 0.5  # sigue en juego

    def check_big_winner(self):
        """
        Verifica si hay ganador en el tablero grande.
        Retorna: 1, -1, 0 (empate), o None (sigue en juego)
        """
        for a, b, c in WIN_COMBOS:
            sa, sb, sc = self.small_boards[a], self.small_boards[b], self.small_boards[c]
            # Los tres tienen que estar ganados por el mismo jugador
            if sa == sb == sc and sa in (1.0, -1.0):
                return sa
        
        # Si no quedan tableros en juego, es empate global
        if not np.any(self.small_boards == 0.5):
            return 0
        
        return None  # sigue en juego

    def get_next_active_board(self, cell_idx):
        """
        Dado el índice de la casilla jugada dentro del tablero pequeño,
        determina cuál es el próximo tablero activo.
        Replica exactamente la lógica de get_actual_square del JS.
        """
        # Si ese tablero está disponible, ir ahí directamente
        if self.small_boards[cell_idx] == 0.5:
            return cell_idx
        
        # Si no, recorrer el historial de atrás para adelante
        # buscando el último tablero disponible
        for board_idx in reversed(self.squares_played):
            if self.small_boards[board_idx] == 0.5:
                return board_idx
        
        # Si no hay ninguno disponible, el juego debería haber terminado
        return None

    def make_move(self, board_idx, cell_idx):
        """
        Realiza un movimiento.
        board_idx: índice del tablero pequeño (0-8)
        cell_idx:  índice de la casilla dentro del tablero (0-8)
        
        Retorna True si el movimiento es válido, False si no.
        """
        # Validaciones
        if self.winner is not None:
            return False  # juego terminado
        if self.active_board is not None and board_idx != self.active_board:
            return False  # tablero incorrecto
        if self.small_boards[board_idx] != 0.5:
            return False  # tablero ya terminado
        if self.board[board_idx][cell_idx] != 0:
            return False  # casilla ocupada

        # Realizar el movimiento
        self.board[board_idx][cell_idx] = self.current_player

        # Actualizar historial
        if not self.squares_played or self.squares_played[-1] != board_idx:
            self.squares_played.append(board_idx)

        # Verificar si se ganó o empató el tablero pequeño
        result = self.check_small_winner(board_idx)
        if result != 0.5:
            self.small_boards[board_idx] = result

        # Verificar si se ganó el juego completo
        big_result = self.check_big_winner()
        if big_result is not None:
            self.winner = big_result

        # Calcular el próximo tablero activo
        self.active_board = self.get_next_active_board(cell_idx)

        # Cambiar de jugador y avanzar turno
        self.current_player *= -1
        self.turn += 1

        return True
    
    def get_valid_moves(self):
        """
        Retorna un array de 81 valores.
        1 = movimiento válido, 0 = inválido.
        """
        mask = np.zeros(81, dtype=np.float32)
        
        if self.winner is not None:
            return mask  # juego terminado, ningún movimiento válido
        
        # Si hay tablero activo, solo se puede jugar ahí
        if self.active_board is not None:
            boards_to_check = [self.active_board]
        else:
            # Primer turno: cualquier tablero disponible
            boards_to_check = [i for i in range(9) if self.small_boards[i] == 0.5]
        
        for board_idx in boards_to_check:
            for cell_idx in range(9):
                if self.board[board_idx][cell_idx] == 0:
                    mask[board_idx * 9 + cell_idx] = 1.0
        
        return mask

    def get_state(self):
        """
        Convierte el estado actual en el vector de 100 entradas para la red.
        
        Estructura:
          [0:81]   casillas del tablero (1=yo, -1=rival, 0=vacío)
          [81:90]  estado tableros pequeños
          [90:99]  one-hot tablero activo (todos 1 si cualquier tablero)
          [99]     turno normalizado
        """
        state = np.zeros(100, dtype=np.float32)
        
        # Casillas: aplanar el tablero 9x9 a 81 valores
        state[0:81] = self.board.flatten()
        
        # Estado de los 9 tableros pequeños
        state[81:90] = self.small_boards
        
        # Tablero activo en one-hot
        if self.active_board is None:
            state[90:99] = 1.0  # todos unos = cualquier tablero
        else:
            state[90 + self.active_board] = 1.0
        
        # Turno normalizado entre 0 y 1
        state[99] = self.turn / 81.0
        
        return state

    def clone(self):
        """Copia rápida del estado sin usar deepcopy."""
        new = SuperTateti.__new__(SuperTateti)
        new.board        = self.board.copy()
        new.small_boards = self.small_boards.copy()
        new.current_player = self.current_player
        new.active_board   = self.active_board
        new.squares_played = self.squares_played.copy()
        new.turn   = self.turn
        new.winner = self.winner
        return new

if __name__ == "__main__":
    game = SuperTateti()
    
    # Turno 1: jugador 1 juega en tablero 4, casilla 0
    game.make_move(4, 0)
    print("Turno 1:")
    print("  Tablero activo:", game.active_board)  # debe ser 0
    print("  Jugador actual:", game.current_player)  # debe ser -1
    print("  Turno:", game.turn)  # debe ser 1
    
    # Turno 2: jugador -1 debe jugar en tablero 0
    game.make_move(0, 4)
    print("Turno 2:")
    print("  Tablero activo:", game.active_board)  # debe ser 4
    
    # Verificar movimiento inválido
    resultado = game.make_move(3, 0)  # tablero incorrecto
    print("Movimiento inválido:", resultado)  # debe ser False
    
    # Verificar get_valid_moves
    moves = game.get_valid_moves()
    print("Movimientos válidos (sum):", moves.sum())  # debe ser 8 (tablero 4, casilla 0 ocupada)
    
    # Verificar get_state
    state = game.get_state()
    print("Shape del estado:", state.shape)  # debe ser (100,)
    print("Turno normalizado:", state[99])   # debe ser 2/81

