import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    # Comprueba si todos los movimientos legales han sido explorados
    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    # Selecciona un hijo basado en el UCB1 con el factor de exploración dado
    def select_child(self, exploration_factor):
        if not self.children:
            return None
        else:
            children_ucb = []
            for i, child in enumerate(self.children):
                if child.visits == 0:
                    children_ucb.append(float('inf'))
                else:
                    ucb = (child.wins / child.visits) + exploration_factor * np.sqrt(2 * np.log(self.visits) / child.visits)
                    children_ucb.append(ucb)
            selected_index = np.argmax(children_ucb)
            print("Índice del hijo seleccionado:", selected_index)  # Agregar esta línea para imprimir el índice del hijo seleccionado
            return self.children[selected_index]

    # Añade un hijo con el estado dado al nodo actual
    def add_child(self, child_state):
        child = Node(child_state, parent=self)
        self.children.append(child)
        child.state.selected_positions = self.state.selected_positions.copy()
        child.state.selected_positions.add(child_state.last_action)
        return child

class TicTacToeState:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 para vacío, 1 para X, -1 para O
        self.player_to_move = 1  # Jugador 1 comienza el juego
        self.last_action = None
        self.selected_positions = set()  # Almacena las posiciones seleccionadas

    # Devuelve una lista de movimientos legales disponibles en el estado actual del juego
    def get_legal_moves(self):
        all_legal_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    all_legal_moves.append((i, j))
        return [move for move in all_legal_moves if move not in self.selected_positions]

    # Realiza el movimiento dado en el estado actual del juego y devuelve el nuevo estado resultante
    def take_action(self, action):
        i, j = action
        new_state = TicTacToeState()
        new_state.board = np.copy(self.board)
        new_state.board[i][j] = self.player_to_move
        new_state.player_to_move = -self.player_to_move  # Cambia de jugador
        new_state.last_action = action
        new_state.selected_positions = set(self.selected_positions)  # Copia las posiciones seleccionadas
        new_state.selected_positions.add(action)  # Agrega la acción actual
        return new_state

    # Determina si el estado actual del juego es terminal y devuelve el resultado del juego
    def is_terminal(self):
        # Comprueba filas, columnas y diagonales para una victoria
        for i in range(3):
            if self.board[i].sum() == 3 or self.board[:, i].sum() == 3:
                return True, 1  # Jugador 1 gana
            if self.board[i].sum() == -3 or self.board[:, i].sum() == -3:
                return True, -1  # Jugador -1 (O) gana
        if self.board.trace() == 3 or np.fliplr(self.board).trace() == 3:
            return True, 1  # Jugador 1 gana
        if self.board.trace() == -3 or np.fliplr(self.board).trace() == -3:
            return True, -1  # Jugador -1 (O) gana
        # Comprueba si hay empate
        if len(self.get_legal_moves()) == 0:
            return True, 0  # Empate
        return False, None

    # Devuelve el resultado del juego en el estado actual
    def get_result(self):
        terminal, winner = self.is_terminal()
        return winner

# Ejecuta el algoritmo MCTS para encontrar el mejor movimiento
def mcts(root_node, iterations):
    for _ in range(iterations):
        node = root_node
        while not node.is_fully_expanded() and node.children:
            node = node.select_child(exploration_factor=1.0)
        expanded_node = expand(node)
        if expanded_node is None:
            continue
        result = simulate_random_game(expanded_node.state)
        backpropagate(expanded_node, result)

    # Elige el mejor movimiento basado en las visitas de los hijos del nodo raíz
    best_child = max(root_node.children, key=lambda x: x.visits)
    best_action = best_child.state.last_action

    print("Hijo seleccionado:", best_child.state.board)  # Agregar esta línea para imprimir el hijo seleccionado

    return best_action

# Expande el nodo seleccionado
def expand(node):
    legal_moves = node.state.get_legal_moves()
    for move in legal_moves:
        new_state = node.state.take_action(move)
        child_node = node.add_child(new_state)
        print("Estado del hijo:", child_node.state.board)  # Agregar esta línea para verificar el estado del nodo hijo
    print("Hijos agregados:", len(node.children))  # Agregar esta línea para verificar el número de hijos agregados
    return node.children[0] if node.children else None

# Simula un juego aleatorio desde el estado dado
def simulate_random_game(state):
    while not state.is_terminal()[0]:
        legal_moves = state.get_legal_moves()

        print("Movimientos legales:", legal_moves)  # Imprime los movimientos válidos
        print("Posiciones seleccionadas por la IA:", state.selected_positions)  # Imprime las posiciones seleccionadas por la IA

        flattened_moves = [move[0] * 3 + move[1] for move in legal_moves]  # Aplana los movimientos legales

        action = np.random.choice(flattened_moves)

        row = action // 3
        col = action % 3
        state = state.take_action((row, col))

    return state.get_result()

# Retropropaga el resultado de la simulación hacia arriba en el árbol
def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.wins += result
        node = node.parent

# Función principal para jugar el juego
def play_game():
    state = TicTacToeState()
    root_node = Node(state)

    while True:
        print("Tablero actual:")
        print(state.board)

        # Turno del jugador
        if state.player_to_move == 1:
            while True:
                try:
                    row = int(input("Ingrese la fila (0-2): "))
                    col = int(input("Ingrese la columna (0-2): "))
                    if (row, col) in state.get_legal_moves():
                        break
                    else:
                        print("¡Movimiento inválido! Por favor elija una celda vacía.")
                except ValueError:
                    print("¡Entrada inválida! Por favor ingrese un número.")

            action = (row, col)
            state.selected_positions.add(action)  # Actualiza las posiciones seleccionadas después del movimiento del jugador
            state = state.take_action(action)
            root_node = Node(state)  # Actualiza el nodo raíz con el nuevo estado
        # Turno del algoritmo MCTS
        else:
            best_action = mcts(root_node, iterations=10000)
            action = best_action
            state.selected_positions.add(action)  # Actualiza las posiciones seleccionadas después del movimiento de la IA
            state = state.take_action(action)
            root_node = Node(state)  # Actualiza el nodo raíz con el nuevo estado

        # Comprueba si el juego ha terminado
        terminal, winner = state.is_terminal()
        if terminal:
            print("¡Juego terminado!")
            print("Ganador:", "Jugador 1 (X)" if winner == 1 else "Jugador 2 (O)" if winner == -1 else "Empate")
            break

play_game()
