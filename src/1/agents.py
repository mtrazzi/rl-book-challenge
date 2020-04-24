from board import TicTacToeBoard
import numpy as np


class RandomAgent:
  def __init__(self, size=3, sym='x'):
    self.sym = sym
    self.size = size

  def best_move(self, board):
    while True:
      x, y = np.random.randint(self.size), np.random.randint(self.size)
      if board.can_place(x, y):
        return x, y

  def train(self, opponent, n_episodes):
    pass


class RLAgent:
  """Offline training, assuming the RL Agent has its opponent as attribute."""
  def __init__(self, size=3, sym='o', step=0.2, eps=0.1, eps_decay=1):
    self.step = step
    self.V = {}
    self.eps = eps
    self.sym = sym
    self.size = size
    self.eps_decay = eps_decay
    self.original_eps = eps

  def get_board_id(self, board):
    return board.__str__()

  def random_move(self, board):
    while True:
      x, y = np.random.randint(board.size), np.random.randint(board.size)
      if board.can_place(x, y):
        return x, y

  def best_move(self, board):
    values = [self.get_move_value(board, i // self.size, i % self.size)
              if board.can_place(i // self.size, i % self.size) else -np.inf
              for i in range(self.size ** 2)]
    idx = np.random.choice(np.flatnonzero(np.isclose(values, max(values))))
    return idx // self.size, idx % self.size

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

  def train_one_step(self, board, opponent=RandomAgent()):
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

  def train(self, opponent=RandomAgent(), n_episodes=1000):
    board = TicTacToeBoard(self.size)
    for episode_nb in range(n_episodes):
      board.reset()
      while not board.is_end_state():
        self.train_one_step(board, opponent)
      if episode_nb % 100 == 0:
        self.eps *= self.eps_decay
        print(self.eps)

  def get_possible_move_values(self, board):
    d = {}
    for x in range(board.size):
      for y in range(board.size):
        if board.can_place(x, y):
          d[f"({x}, {y})"] = self.get_move_value(board, x, y)
    return d
