import numpy as np


class RandomAgent:
  def best_move(self, board):
    while True:
      x, y = np.random.randint(3), np.random.randint(3)
      if board.can_place(x, y):
        return x, y


class RLAgent:
  """Offline training, assuming the RL Agent has its opponent has attribute."""
  def __init__(self, sym='o', step=0.2, eps=0.1):
    self.step = step
    self.V = {}
    self.eps = eps
    self.sym = sym

  def get_board_id(self, board):
    return board.__str__()

  def random_move(self, board):
    while True:
      x, y = np.random.randint(3), np.random.randint(3)
      if board.can_place(x, y):
        return x, y

  def best_move(self, board):
    max_value = -np.inf
    for x in range(3):
      for y in range(3):
        if board.can_place(x, y):
          value = self.get_move_value(board, x, y)
          if value > max_value:
            best_move = (x, y)
            max_value = value
    return best_move

  def get_move_value(self, board, x, y):
    board.do_move(x, y)
    value = self.get_value(board)
    board.undo_move(x, y)
    return value

  def get_value(self, board):
    state_id = board.__str__()
    if state_id in self.V:
      return self.V[state_id]
    self.V[state_id] = board.result(max_player=self.sym)
    return self.V[state_id]

  def update_value(self, s_t, s_tp1):
    self.V[s_t] = self.V[s_t] + self.step * (self.V[s_tp1] - self.V[s_t])

  def eps_greedy(self, board):
    if np.random.random() < self.eps:
      return "eps", self.random_move(board)
    else:
      return "greedy", self.best_move(board)

  def train(self, board, opponent=RandomAgent(), n_episodes=1000):
    for _ in range(n_episodes):
      board.reset()
      while not board.is_end_state():
        s_t = board.__str__()
        self.get_value(board)  # to initialize V(s_t) if it doesn't exist yet
        board.do_move(*opponent.best_move(board))
        if not board.is_end_state():
          case, move = self.eps_greedy(board)
          board.do_move(*move)
          s_tp1 = board.__str__()  # idem with V(s_{t+1})
          self.get_value(board)
          if case == "greedy":  # only update from non-exploratory moves
            self.update_value(s_t, s_tp1)

  def get_possible_move_values(self, board):
    d = {}
    for x in range(3):
      for y in range(3):
        if board.can_place(x, y):
          d[f"({x}, {y})"] = self.get_move_value(board, x, y)
    return d
