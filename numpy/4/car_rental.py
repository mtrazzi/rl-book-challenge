import numpy as np

MAX_CAR_NB = 5
MAX_CAR_MOVES = 5
REN_REQ_LAMBDA = [3, 4]
RETURNS_LAMBDA = [3, 2]

class CarRentalEnv:
  def __init__(self):
    self.states = [(x, y) for x in range(MAX_CAR_NB)
                   for y in range(MAX_CAR_NB)]
    self.moves = list(range(-MAX_CAR_MOVES, MAX_CAR_MOVES + 1))
    self.r = [0]
    self.size = MAX_CAR_NB

  def next_s(self, s, a):
    return s[0] - a, s[1] + a

  def reward(self, s, a):
    requests = [np.random.poisson(lam) for lam in REN_REQ_LAMBDA]
    if s[0] < requests[0] or s[1] < requests[1]:
      return 0
    return 0

  def is_valid(self, s):
    return np.all([0 <= nb_cars <= 20 for nb_cars in s])

  def p(self, s_p, r, s, a):
    s_next = self.next_s(s, a)
    r_next = self.reward(s, a)
    return 1 if (self.is_valid(s_next)
                 and np.all(s_next == s_p) and
                 (r == r_next)) else 0

  def is_terminal(self, s):
    return False