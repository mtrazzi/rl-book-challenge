import numpy as np
from mdp import MDP

MOVES = {
  "UP": (-1, 0),
  "DOWN": (1, 0),
  "RIGHT": (0, 1),
  "LEFT": (0, -1),
}

UNIQUE_TERMINAL_STATE = (0, 0)


class Gridworld(MDP):
  def __init__(self, size=4, cost_move=1):
    self.size_val = size
    self.cost_move = cost_move
    super().__init__()

  @property
  def size(self):
    return self.size_val

  @property
  def moves(self):
    return list(MOVES.keys())

  @property
  def states(self):
    return [(x, y) for x in range(self.size)
            for y in range(self.size)]

  @property
  def r(self):
    return [-1, 0]

  def next_s(self, s, a):
    move = MOVES[a]
    candidate = s[0] + move[0], s[1] + move[1]
    if self.is_terminal(s):
      return s
    elif self.is_terminal(candidate):
      return UNIQUE_TERMINAL_STATE  # two grey cells are the same state
    elif not self.is_valid(candidate):  # out of border move
      return s
    else:
      return candidate

  def reward(self, s, a):
    return 0 if self.is_terminal(s) else -self.cost_move

  def is_valid(self, s):
    def is_valid_coord(x):
      return 0 <= x < self.size
    return np.all([is_valid_coord(s[i]) for i in range(len(s))])

  def _p(self, s_p, r, s, a):
    s_next = self.next_s(s, a)
    r_next = self.reward(s, a)
    return 1 if (self.is_valid(s_next)
                 and np.all(s_next == s_p) and
                 (r == r_next)) else 0

  def is_terminal(self, s):
    return ((s[0] == 0 and s[1] == 0) or
            (s[0] == (self.size - 1) and s[1] == (self.size - 1)))
