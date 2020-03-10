import numpy as np
from scipy.stats import skellam
from mdp import MDP

MAX_CAR_CAP = 5
MAX_CAR_MOVES = 5
REQUEST_LAMBDA = [1, 1]
RETURNS_LAMBDA = [1, 1]
CAR_MOVE_COST = 2
RENT_BEN = 10
NB_LOC = 2


class CarRentalEnv(MDP):
  def __init__(self):
    self.init_skellman_probs()
    super().__init__()

  @property
  def size(self):
    return MAX_CAR_CAP

  @property
  def moves(self):
    return list(range(-MAX_CAR_MOVES, MAX_CAR_MOVES + 1))

  @property
  def states(self):
    return [(x, y) for x in range(MAX_CAR_CAP + 1)
            for y in range(MAX_CAR_CAP + 1)]

  @property
  def r(self):
    return list(range(-MAX_CAR_MOVES * CAR_MOVE_COST, MAX_CAR_MOVES *
                      CAR_MOVE_COST + 1))

  def is_valid(self, s):
    car_arr = np.array(s)
    return np.all((0 <= car_arr) & (car_arr <= MAX_CAR_CAP))

  def init_skellman_probs(self):
    self.skell_pmfs = {i: {} for i in range(NB_LOC)}
    self.skell_sf_pmfs = {i: {} for i in range(NB_LOC)}
    pmf_range = list(range(-MAX_CAR_CAP - MAX_CAR_MOVES,
                     MAX_CAR_CAP + MAX_CAR_MOVES + 1))
    for i in range(NB_LOC):
      lam_ret, lam_rent = RETURNS_LAMBDA[i], REQUEST_LAMBDA[i]
      self.skell_pmfs[i] = {i: skellam.pmf(i, lam_ret, lam_rent)
                            for i in pmf_range}
      self.skell_sf_pmfs[i] = {i: (skellam.pmf(i, lam_ret, lam_rent) +
                               skellam.sf(i, lam_ret, lam_rent))
                               for i in pmf_range}

  def _p(self, s_p, r, s, a):
    (n1, n2), (n1_p, n2_p), m = s, s_p, a
    if (n1_p < 0 or n2_p < 0 or not (0 <= m <= MAX_CAR_MOVES)
        or not (0 <= n1 <= MAX_CAR_CAP) or not (0 <= n2 <= MAX_CAR_CAP)):
      return 0

    def proba_move_loc(n_p, n, new_cars, location):
      idx = n_p - n + new_cars
      if n_p != MAX_CAR_CAP:
        return self.skell_pmfs[location][idx]
      else:
        return self.skell_sf_pmfs[location][idx]

    return proba_move_loc(n1_p, n1, -m, 0) * proba_move_loc(n1_p, n1, m, 1)

  def is_terminal(self, s):
    return not self.is_valid(s)
