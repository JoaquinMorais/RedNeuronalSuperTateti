import numpy as np

# Combinaciones ganadoras en una cuadrícula de 3x3
WIN_COMBOS = [
    (0,1,2), (3,4,5), (6,7,8),  # filas
    (0,3,6), (1,4,7), (2,5,8),  # columnas
    (0,4,8), (2,4,6),            # diagonales
]

class SuperTateti:
    def __init__(self):
        # 1 = X, -1 = O, 0 = vacío
        self.board = np.zeros((9, 9), dtype=np.float32)
        
        # Estado macro del juego. 1 = Gana X, -1 = Gana O, 0 = Empate, 0.5 = En juego
        self.small_boards = np.full(9, 0.5, dtype=np.float32)
        
        self.current_player = 1   # Comienza el jugador 1
        self.active_board = None  # None significa libertad de jugar en cualquier tablero libre
        self.squares_played = []  # Historial de tableros jugados
        self.turn = 0
        self.winner = None        # 1, -1, 0 (empate), None (en progreso)
    
    def check_small_winner(self, board_idx):
        board = self.board[board_idx]
        for a, b, c in WIN_COMBOS:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        if not np.any(board == 0):
            return 0  # Tablero chico empatado
        return 0.5

    def check_big_winner(self):
        for a, b, c in WIN_COMBOS:
            sa, sb, sc = self.small_boards[a], self.small_boards[b], self.small_boards[c]
            if sa == sb == sc and sa in (1.0, -1.0):
                return sa
        if not np.any(self.small_boards == 0.5):
            return 0  # Macro-tablero empatado
        return None

    def get_next_active_board(self, cell_idx):
        if self.small_boards[cell_idx] == 0.5:
            return cell_idx
        # Si el tablero de destino está cerrado, busca el último tablero activo libre en el historial
        for board_idx in reversed(self.squares_played):
            if self.small_boards[board_idx] == 0.5:
                return board_idx
        return None

    def make_move(self, board_idx, cell_idx):
        if self.winner is not None:
            return False
        if self.active_board is not None and board_idx != self.active_board:
            return False
        if self.small_boards[board_idx] != 0.5:
            return False
        if self.board[board_idx][cell_idx] != 0:
            return False

        self.board[board_idx][cell_idx] = self.current_player

        if not self.squares_played or self.squares_played[-1] != board_idx:
            self.squares_played.append(board_idx)

        result = self.check_small_winner(board_idx)
        if result != 0.5:
            self.small_boards[board_idx] = result

        big_result = self.check_big_winner()
        if big_result is not None:
            self.winner = big_result

        self.active_board = self.get_next_active_board(cell_idx)

        self.current_player *= -1
        self.turn += 1
        return True
    
    def get_valid_moves(self):
        mask = np.zeros(81, dtype=np.float32)
        if self.winner is not None:
            return mask
        
        if self.active_board is not None:
            boards_to_check = [self.active_board]
        else:
            boards_to_check = [i for i in range(9) if self.small_boards[i] == 0.5]
        
        for board_idx in boards_to_check:
            for cell_idx in range(9):
                if self.board[board_idx][cell_idx] == 0:
                    mask[board_idx * 9 + cell_idx] = 1.0
        return mask

    def get_state(self):
        state = np.zeros(100, dtype=np.float32)
        state[0:81] = self.board.flatten()
        state[81:90] = self.small_boards
        if self.active_board is None:
            state[90:99] = 1.0
        else:
            state[90 + self.active_board] = 1.0
        state[99] = self.turn / 81.0
        return state

    def clone(self):
        # Clonación rápida y limpia sin romper referencias a listas internas
        new = SuperTateti.__new__(SuperTateti)
        new.board        = self.board.copy()
        new.small_boards = self.small_boards.copy()
        new.current_player = self.current_player
        new.active_board   = self.active_board
        new.squares_played = list(self.squares_played)
        new.turn   = self.turn
        new.winner = self.winner
        return new
    
    def heuristic_value(self):
        """
        Calcula una ventaja numérica simple estricta para self.current_player.
        Retorna un valor acotado en el rango [-1, 1].
        """
        score = 0.0
        player = self.current_player

        # 1. Puntos por tableros chicos ganados / perdidos
        for i in range(9):
            if self.small_boards[i] == player:
                score += 1.0
            elif self.small_boards[i] == -player:
                score -= 1.0

        # 2. Relevancia táctica del centro macro
        center = 4
        if self.small_boards[center] == player:
            score += 0.5
        elif self.small_boards[center] == -player:
            score -= 0.5

        # 3. Relevancia de las casillas centrales de cada tablero pequeño
        for board_idx in range(9):
            if self.small_boards[board_idx] == 0.5:
                center_cell = self.board[board_idx][4]
                if center_cell == player:
                    score += 0.1
                elif center_cell == -player:
                    score -= 0.1

        # 4. Amenazas directas de dos en raya en el tablero macro
        for a, b, c in WIN_COMBOS:
            line = [self.small_boards[a], self.small_boards[b], self.small_boards[c]]
            if line.count(player) == 2 and line.count(0.5) == 1:
                score += 0.8
            if line.count(-player) == 2 and line.count(0.5) == 1:
                score -= 0.8

        return max(-1.0, min(1.0, score / 5.0))

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

