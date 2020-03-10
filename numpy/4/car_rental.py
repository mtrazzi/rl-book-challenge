import numpy as np
from scipy.stats import skellam
from utils import trans_id

MAX_CAR_CAP = 4
MAX_CAR_MOVES = 5
REQUEST_LAMBDA = [3, 4]
RETURNS_LAMBDA = [3, 2]
CAR_MOVE_COST = 2
RENT_BEN = 10
NB_LOC = 2


class CarRentalEnv:
  def __init__(self):
    self.states = [(x, y) for x in range(MAX_CAR_CAP)
                   for y in range(MAX_CAR_CAP)]
    self.moves = list(range(-MAX_CAR_MOVES, MAX_CAR_MOVES + 1))
    self.r = list(range(-MAX_CAR_MOVES * CAR_MOVE_COST, MAX_CAR_MOVES *
                        CAR_MOVE_COST + 1))
    self.size = MAX_CAR_CAP
    print("starting to compute transitions p...")
    self.p = {trans_id(s_p, r, s, a): self._p(s_p, r, s, a) for a in self.moves
              for s in self.states for r in self.r for s_p in self.states}
    print("done...")

  def is_valid(self, s):
    car_arr = np.array(s)
    return np.all((0 <= car_arr) & (car_arr <= MAX_CAR_CAP))

  def _p(self, s_p, r, s, a):
    (n1, n2), (n1_p, n2_p), m = s, s_p, a
    if (n1_p < 0 or n2_p < 0 or not (0 <= m <= MAX_CAR_MOVES)
        or not (0 <= n1 <= MAX_CAR_CAP) or not (0 <= n2 <= MAX_CAR_CAP)):
      return 0

    def proba_move_loc(n_p, n, new_cars, lam_ret, lam_rent):
      args = n_p - n + new_cars, lam_ret, lam_rent
      if n_p != MAX_CAR_CAP:
        return skellam.pmf(*args)
      else:
        return skellam.sf(*args) + skellam.pmf(*args)
    return (proba_move_loc(n1_p, n1, -m, RETURNS_LAMBDA[0], REQUEST_LAMBDA[0])
            * proba_move_loc(n1_p, n1, m, RETURNS_LAMBDA[1], REQUEST_LAMBDA[1]))

  def is_terminal(self, s):
    return not self.is_valid(s)
