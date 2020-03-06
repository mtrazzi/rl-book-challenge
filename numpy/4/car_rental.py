import numpy as np
from scipy.stats import skellam

MAX_CAR_CAP = 4
MAX_CAR_MOVES = 5
REN_REQ_LAMBDA = [3, 4]
RETURNS_LAMBDA = [3, 2]
CAR_MOVE_COST = 2
RENT_BEN = 10
NB_LOC = 2

class CarRentalEnv:
  def __init__(self):
    self.states = [(x, y) for x in range(MAX_CAR_CAP)
                   for y in range(MAX_CAR_CAP)]
    self.moves = list(range(-MAX_CAR_MOVES, MAX_CAR_MOVES + 1))
    self.r = [0]
    self.size = MAX_CAR_CAP

  def is_valid(self, s):
    car_arr = np.array(s)
    return np.all((0 <= car_arr) & (car_arr <= MAX_CAR_CAP))

  def p(self, s_p, r, s, a):
    (n1, n2), (n1_p, n2_p), m = s, s_p, a
    if (n1_p < 0 or n2_p < 0 or not (0 <= m <= MAX_CAR_MOVES) or not (0 <= n1 <= MAX_CAR_CAP) or not (0 <= n2 <= MAX_CAR_CAP)):
      return 0
    def proba_move_loc(n_p, n, new_cars, lam_ret, lam_rent):
      args = n_p - n + new_cars, lam_ret, lam_rent
      if n_p != MAX_CAR_CAP:
        return skellam.pmf(*args)
      else:
        return skellam.sf(*args) + skellam.pmf(*args)
    req, ret = [[np.random.poisson(lam) for lam in lam_list]
                for lam_list in [REN_REQ_LAMBDA, RETURNS_LAMBDA]]
    return (proba_move_loc(n1_p, n1, -m, ret[0], req[0]) * 
            proba_move_loc(n1_p, n1, m, ret[1], req[1]))

  def is_terminal(self, s):
    return not self.is_valid(s)