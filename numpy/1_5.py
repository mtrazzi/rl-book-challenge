import os

import numpy as np

PLAYERS = ['x', 'o']


class TicTacToeBoard:
  def __init__(self):
    self.board = self.new_board()
    self.players = PLAYERS
    self.curr_play_idx = 0

  def is_valid_move(self, x, y):
    return 0 <= x < 3 and 0 <= y < 3 and self.board[x][y] == '.'

  def do_move(self, x, y):
    if self.is_valid_move(x, y):
      self.board[x][y] = self.players[self.curr_play_idx]
      self.curr_play_idx = (self.curr_play_idx + 1) % 2

  def undo_move(self, x, y):
    self.board[x][y] = '.'
    self.curr_play_idx = (self.curr_play_idx + 1) % 2

  def transpose(self):
    return np.array(self.board).T.tolist()

  def diag(self):
    return [[self.board[x_0 + i * dx][y_0 + i * dy] for i in (range(3))]
            for (x_0, y_0, dx, dy) in [(0, 0, 1, 1), (2, 0, -1, 1)]]

  def has_won(self, sym):
    for row in self.board + self.transpose() + self.diag():
      if row[0] == row[1] == row[2] == sym:
        return True
    return False

  def is_over(self):
    for row in self.board:
      if '.' in row:
        return False
    return True

  def is_end_state(self):
    return self.is_over() or self.has_won('x') or self.has_won('o')

  def can_place(self, x, y):
    return self.board[x][y] == '.'

  def new_board(self):
    return [['.' for _ in range(3)] for _ in range(3)]

  def reset(self):
    self.board = self.new_board()

  def opposite(self, sym):
    return 'x' if sym == 'o' else 'o'

  def result(self, max_player):
    if self.has_won(max_player):
      return 1
    elif self.has_won(self.opposite(max_player)):
      return 0
    else:
      return 0.5

  def __str__(self):
    return '\n'.join([' '.join(row) for row in self.board])


class RandomAgent:
  def __init__(self, board):
    self.board = board

  def best_move(self):
    while True:
      x, y = np.random.randint(3), np.random.randint(3)
      if self.board.can_place(x, y):
        return x, y


class RLAgent:
  """Offline training, assuming the RL Agent has its opponent has attribute."""
  def __init__(self, board, sym='o', step=0.2, eps=0.1, opp_type="random"):
    self.step = step
    self.V = {}
    self.eps = eps
    self.sym = sym
    self.board = board
    self.initialize_opponent(opp_type)

  def initialize_opponent(self, opp_type):
    if opp_type == "random":
      self.opponent = RandomAgent(self.board)
    elif opp_type == "RL":
      self.opponent = RLAgent(self.board, sym='x', opp_type="random")
      self.opponent.train(n_episodes=1000)

  def get_board_id(self):
    return self.board.__str__()

  def random_move(self):
    while True:
      x, y = np.random.randint(3), np.random.randint(3)
      if self.board.can_place(x, y):
        return x, y

  def best_move(self):
    max_value = -np.inf
    for x in range(3):
      for y in range(3):
        if self.board.can_place(x, y):
          value = self.get_move_value(x, y)
          if value > max_value:
            best_move = (x, y)
            max_value = value
    return best_move

  def get_move_value(self, x, y):
    self.board.do_move(x, y)
    value = self.get_value()
    self.board.undo_move(x, y)
    return value

  def get_value(self):
    state_id = self.get_board_id()
    if state_id in self.V:
      return self.V[state_id]
    self.V[state_id] = self.board.result(max_player=self.sym)
    return self.V[state_id]

  def update_value(self, s_t, s_tp1):
    self.V[s_t] = self.V[s_t] + self.step * (self.V[s_tp1] - self.V[s_t])

  def eps_greedy(self):
    if np.random.random() < self.eps:
      return "eps", self.random_move()
    else:
      return "greedy", self.best_move()

  def train(self, n_episodes=1000):
    for _ in range(n_episodes):
      self.board.reset()
      while not self.board.is_end_state():
        s_t = self.get_board_id()
        self.get_value()  # to initialize V(s_t) if it doesn't exist yet
        opponent_move = self.opponent.best_move()
        self.board.do_move(*opponent_move)
        if not self.board.is_end_state():
          case, move = self.eps_greedy()
          self.board.do_move(*move)
          s_tp1 = self.get_board_id()  # idem with V(s_{t+1})
          self.get_value()
          if case == "greedy":  # only update from non-exploratory moves
            self.update_value(s_t, s_tp1)

  def get_possible_move_values(self):
    d = {}
    for x in range(3):
      for y in range(3):
        if self.board.can_place(x, y):
          d[f"({x}, {y})"] = self.get_move_value(x, y)
    return d


def play_against_agent(agent):
  agent.board.reset()
  while not agent.board.is_end_state():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(agent.board)
    x, y = [int(elt) for elt in input().split()]
    agent.board.do_move(x, y)
    print(agent.board)
    print(agent.get_possible_move_values())
    input("...")
    agent.board.do_move(*agent.best_move())


class RLBoxAgent:
  """Trained online from playing against opponents."""
  def __init__(self, sym='o', step=0.1, eps=0.1):
    self.sym = sym
    self.step = step
    self.eps = eps
    self.V = {}

  def initialize_values(self):
    pass

  def update(self, board, last_move):
    pass

  def play(self, board):
    pass


def main():
  np.random.seed(42)
  board = TicTacToeBoard()
  agent = RLAgent(board, opp_type="random")
  agent.train(n_episodes=1000)
  while True:
    play_against_agent(agent)


if __name__ == '__main__':
  main()
