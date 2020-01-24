import numpy as np

PLAYERS = ['x', 'o']


class TicTacToeBoard:
  def __init__(self):
    self.board = new_board()
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
      if len(set(row)) == 1 and row[0] == sym:
        return True
    return False

  def is_over(self):
    for row in self.board:
      if '.' in row:
        return False
    return True

  def is_end_state(self):
    return self.is_over() or self.has_won('x') or self.has_won('o')

  def can_place(self, sym, x, y):
    return self.board[x][y] == '.'

  def new_board(self):
    return [['.' for _ in range(3)] for _ in range(3)]

  def reset(self):
    self.board = new_board()

  def __str__(self):
    return '\n'.join([' '.join(row) for row in self.board])


class TicTacToeAgent:
  def __init__(self, board, sym='x', alpha=0.1, eps=0.1):
    self.alpha = alpha
    self.V = {}
    self.eps = eps
    self.sym = sym
    self.board = board

  def random_move(self):
    while True:
      x, y = np.random.randint(3), np.random.randint(3)
      if self.board.can_place(x, y):
        return x, y

  def greedy(self):
    best_move, max_value = (0, 0), 0
    for x in range(3):
      for y in range(3):
        value = self.get_move_value(x, y)
        if value > max_value:
          best_move = (x, y)
          max_value = value
    return best_move

  def get_move_value(self,  x, y):
    self.board.do_move(x, y)
    value = self.get_value()
    self.board.undo_move(x, y)
    return value

  def get_value(self):
    state_id = self.board.__str__()
    if state_id in self.V:
      return self.V[state_id]
    self.V[state_id] = 1 if self.board.has_won(self.sym) else 0.5

  def eps_greedy(self, board):
    if np.random.random() < self.eps:
      return self.random_move()
    else:
      return self.greedy()

  # def train(self, opponent, n_episodes=1000):
  #   for _ in range(n_episodes):
  #     self.board.reset()
  #     # while True:





def main():
  board = TicTacToeBoard()
  agent = TicTacToeAgent(board)
  while (True):
    print(board)
    print(board.has_won('x'))
    x, y = [int(elt) for elt in input().split()]
    board.do_move(x, y)
  # import ipdb; ipdb.set_


if __name__ == '__main__':
  main()
