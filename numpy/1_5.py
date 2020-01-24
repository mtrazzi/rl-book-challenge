import numpy as np

BOARD = [
  ['.', '.', '.'],
  ['.', '.', '.'],
  ['.', '.', '.'],
]

PLAYERS = ['x', 'o']

class TicTacToeBoard:
  def __init__(self):
    self.board = BOARD
    self.players = PLAYERS
    self.curr_play_idx = 0

  def do_move(self, x, y):
    self.board[x][y] = self.players[self.curr_play_idx]
    self.curr_play_idx = (self.curr_play_idx + 1) % 2

  def __str__(self):
    return '\n'.join([' '.join(row) for row in self.board])



class TicTacToeAgent:
  def __init__(self, alpha=0.1, eps=0.1):
    self.alpha = alpha
    self.V = {}
    self.eps = eps

  def possible_states(self):
    return []

  def aligned_three(self, state):
    return 0

  def initialize(self):
    for state in self.possible_states():
      self.V[state] = 1 if self.aligned_three(state) else 0.5

  def play(self, state):
    if np.random.random() < self.eps:
      return 0
    else:
      return 0

def main():
  agent = TicTacToeAgent()
  board = TicTacToeBoard()
  board.do_move(0, 0)
  board.do_move(2, 2)
  print(board)

if __name__ == '__main__':
  main()
