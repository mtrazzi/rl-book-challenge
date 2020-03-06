import numpy as np

MOVES = {
  "UP": (-1, 0),
  "DOWN": (1, 0),
  "RIGHT": (0, 1),
  "LEFT": (0, -1),
}

UNIQUE_TERMINAL_STATE = (0, 0)


class Gridworld:
  def __init__(self, size=4):
    self.size = size
    self.n_states = self.size ** 2 - 1
    self.moves = list(MOVES.keys())
    self.states = [(x, y) for x in range(self.size)
                   for y in range(self.size)]
    self.r = [-1, 0]

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
    return 0 if self.is_terminal(s) else -1

  def is_valid(self, s):
    def is_valid_coord(x):
      return 0 <= x < self.size
    return np.all([is_valid_coord(s[i]) for i in range(len(s))])

  def p(self, s_p, r, s, a):
    s_next = self.next_s(s, a)
    r_next = self.reward(s, a)
    return 1 if (self.is_valid(s_next)
                 and np.all(s_next == s_p) and
                 (r == r_next)) else 0

  def is_terminal(self, s):
    return ((s[0] == 0 and s[1] == 0) or
            (s[0] == (self.size - 1) and s[1] == (self.size - 1)))

  def p(self, s_p, r, s, a):
    (n1, n2), (n1_p, n2_p), m = s, s_p, a
    if (n1_p < 0 or n2_p < 0 or not (0 <= m <= MAX_CAR_MOVES) or not (0 <= n1 <= MAX_CAR_CAP) or not (0 <= n2 <= MAX_CAR_CAP)):
      return 0
    def proba_move_loc(n_p, n, new_cars, lam_ret, lam_rent):
      args = n_p - n + new_cars, lam_ret, lam_rent
      if n_p != MAX_CAR_CAP:
        return skellman.pmf(*args)
      else:
        return skellman.sf(*args) + skellman.pmf(*args)
    return proba_move_loc(n1_p, n1, -m, )

  def reward(self, s, a):
    req, ret = [[np.random.poisson(lam) for lam in lam_list]
                          for lam_list in [REN_REQ_LAMBDA, RETURNS_LAMBDA]]
    new_nb_cars = [min(s[i] + ret[i], MAX_CAR_CAP) for i in range(NB_LOC)]
    if np.all(np.array(req) <= np.array(new_nb_cars)):
      return abs(a) * CAR_MOVE_COST + np.sum(req) * RENT_BEN
    else:
      return 0
