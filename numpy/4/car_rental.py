import numpy as np
from scipy.stats import skellam, poisson
from mdp import MDP
from utils import trans_id

MAX_CAR_MOVES = 1
REQUEST_LAMBDA = [3, 4]
RETURNS_LAMBDA = [3, 2]
CAR_MOVE_COST = 2
RENT_BEN = 10
NB_LOC = 2


class CarRentalEnv(MDP):
  def __init__(self, size):
    self.max_car_cap = size
    self.init_probs()
    super().__init__()
    self.renormalize()

  @property
  def size(self):
    # possible nb of cars can go frmo 0 to max car capacity
    return self.max_car_cap + 1

  @property
  def moves(self):
    return list(range(-MAX_CAR_MOVES, MAX_CAR_MOVES + 1))

  @property
  def states(self):
    return [(x, y) for x in range(self.max_car_cap + 1)
            for y in range(self.max_car_cap + 1)]

  @property
  def r(self):
    return [-CAR_MOVE_COST * car_moves + RENT_BEN * car_solds
            for car_moves in range(MAX_CAR_MOVES + 1)
            for car_solds in range(self.max_car_cap * NB_LOC + 1)]

  def is_valid(self, s):
    car_arr = np.array(s)
    return np.all((0 <= car_arr) & (car_arr <= self.max_car_cap))

  def init_probs(self):
    # nb cars returned yesterday - nb cars rented today is a diff of poisson
    # which follows a skellman distribution
    self.skell_pmfs = {i: {} for i in range(NB_LOC)}
    self.skell_sf_pmfs = {i: {} for i in range(NB_LOC)}
    # can only sell cars up to max capacity * nb of locations
    self.poiss_pmfs = [poisson.pmf(j, sum(REQUEST_LAMBDA))
                       for j in range(max(self.r) + 1)]
    pmf_range = list(range(-self.max_car_cap - MAX_CAR_MOVES,
                     self.max_car_cap + MAX_CAR_MOVES + 1))
    for i in range(NB_LOC):
      lam_ret, lam_rent = RETURNS_LAMBDA[i], REQUEST_LAMBDA[i]
      self.skell_pmfs[i] = {j: skellam.pmf(j, lam_ret, lam_rent)
                            for j in pmf_range}
      self.skell_sf_pmfs[i] = {j: (skellam.pmf(j, lam_ret, lam_rent) +
                               skellam.sf(j, lam_ret, lam_rent))
                               for j in pmf_range}

  def _p(self, s_p, r, s, a):
    (n1, n2), (n1_p, n2_p), m = s, s_p, a
    if (n1_p < 0 or n2_p < 0 or not (0 <= abs(m) <= MAX_CAR_MOVES)
        or not (0 <= n1 <= self.max_car_cap)
        or not (0 <= n2 <= self.max_car_cap)
        or not (0 <= n1 - m) or not (0 <= n2 + m)):
      return 0

    def proba_move_loc(n_p, n, new_cars, location):
      idx = n_p - n + new_cars
      if n_p != self.max_car_cap:
        return self.skell_pmfs[location][idx]
      else:
        return self.skell_sf_pmfs[location][idx]

    p1, p2 = proba_move_loc(n1_p, n1, m, 0), proba_move_loc(n2_p, n2, -m, 1)
    succ_prob = p1 * p2
    # probabilty of reward being r depends on number of requests and moves
    rew_prob = self.poiss_pmfs[min(r + CAR_MOVE_COST * abs(m), max(self.r))]
    return succ_prob * rew_prob

  def renormalize(self):
    for s in self.states:
        for a in self.moves:
            p_sum = sum([self._p(s_p, r, s, a) for s_p in self.states
                        for r in self.r])
            if p_sum > 0:
              for s_p in self.states:
                for r in self.r:
                  self.p[trans_id(s_p, r, s, a)] /= p_sum

  def is_terminal(self, s):
    return not self.is_valid(s)
