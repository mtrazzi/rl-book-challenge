import numpy as np

PLAYERS = ['x', 'o']


class TicTacToeBoard:
  def __init__(self, size=3):
    self.size = size
    self.board = self.new_board()
    self.players = PLAYERS
    self.curr_play_idx = 0

  def is_valid_move(self, x, y):
    return 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == '.'

  def do_move(self, x, y):
    if self.is_valid_move(x, y):
      self.board[x][y] = self.players[self.curr_play_idx]
      self.curr_play_idx = (self.curr_play_idx + 1) % 2
    else:
      exit("Tried to do an invalid move")

  def undo_move(self, x, y):
    self.board[x][y] = '.'
    self.curr_play_idx = (self.curr_play_idx + 1) % 2

  def transpose(self):
    return np.array(self.board).T.tolist()

  def diag(self):
    return [[self.board[x_0 + i * dx][y_0 + i * dy] for i in (range(self.size))]
            for (x_0, y_0, dx, dy) in [(0, 0, 1, 1), (self.size - 1, 0, -1, 1)]]

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

  def can_place(self, x, y):
    return self.board[x][y] == '.'

  def new_board(self):
    return [['.' for _ in range(self.size)] for _ in range(self.size)]

  def reset(self):
    self.board = self.new_board()

  def result(self, max_player):
    from utils import opposite
    if self.has_won(max_player):
      return 1
    elif self.has_won(opposite(max_player)):
      return 0
    else:
      return 0.5

  def __str__(self):
    return '\n'.join([' '.join(row) for row in self.board])
