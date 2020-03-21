import numpy as np
from scipy.stats import skellam, poisson
from mdp import MDP
from utils import trans_id

REQUEST_LAMBDA = [3, 4]
RETURNS_LAMBDA = [3, 2]
CAR_MOVE_COST = 2
RENT_BEN = 10
NB_LOC = 2
ABSORBING_STATE = (-1, -1)


class CarRentalEnv(MDP):
  def __init__(self, size):
    self.max_car_cap = size
    self.max_car_moves = self.max_car_cap // 5 + 1
    self.init_probs()
    super().__init__()
    # print(f"self.max_car_moves is {self.max_car_moves}")

  @property
  def size(self):
    # possible nb of cars can go frmo 0 to max car capacity
    return self.max_car_cap + 1

  @property
  def moves(self):
    return list(range(-self.max_car_moves, self.max_car_moves + 1))

  @property
  def states(self):
    return [(x, y) for x in range(self.max_car_cap + 1)
            for y in range(self.max_car_cap + 1)] + [ABSORBING_STATE]

  @property
  def r(self):
    return [-CAR_MOVE_COST * car_moves + RENT_BEN * car_solds
            for car_moves in range(self.max_car_moves + 1)
            for car_solds in range(self.max_car_cap * NB_LOC + 1)]

  def is_valid(self, s):
    car_arr = np.array(s)
    return np.all((0 <= car_arr) & (car_arr <= self.max_car_cap))

  def init_probs_old(self):
    # nb cars returned yesterday - nb cars rented today is a diff of poisson
    # which follows a skellman distribution
    self.skell_pmfs = {i: {} for i in range(NB_LOC)}
    self.skell_sf_pmfs = {i: {} for i in range(NB_LOC)}
    # can only sell cars up to max capacity * nb of locations
    self.poiss_pmfs = [poisson.pmf(j, sum(REQUEST_LAMBDA))
                       for j in range(max(self.r) + 1)]
    pmf_range = list(range(-self.max_car_cap - self.max_car_moves,
                     self.max_car_cap + self.max_car_moves + 1))
    for i in range(NB_LOC):
      lam_ret, lam_rent = RETURNS_LAMBDA[i], REQUEST_LAMBDA[i]
      self.skell_pmfs[i] = {j: skellam.pmf(j, lam_ret, lam_rent)
                            for j in pmf_range}
      self.skell_sf_pmfs[i] = {j: (skellam.pmf(j, lam_ret, lam_rent) +
                               skellam.sf(j, lam_ret, lam_rent))
                               for j in pmf_range}

  def init_probs(self):
    # can only sell cars up to max capacity * nb of locations
    sell_range = range(self.max_car_cap * NB_LOC + 1)
    self.req_pmfs = {i: [poisson.pmf(j, REQUEST_LAMBDA[i])
                     for j in sell_range] for i in range(NB_LOC)}
    self.req_cdf = {i: [poisson.cdf(j, REQUEST_LAMBDA[i])
                   for j in sell_range] for i in range(NB_LOC)}
    self.ret_pmfs = {i: [poisson.pmf(j, RETURNS_LAMBDA[i])
                     for j in sell_range] for i in range(NB_LOC)}
    self.ret_sf = {i: [poisson.sf(j, RETURNS_LAMBDA[i])
                   for j in sell_range] for i in range(NB_LOC)}

  def _p(self, s_p, r, s, a):
    (n1, n2), (n1_p, n2_p), m = s, s_p, a
    move_cost = abs(m) * CAR_MOVE_COST
    max_ben = (n1 + n2) * RENT_BEN
    nb_sells = (r + move_cost) // RENT_BEN
    if self.is_terminal(s):
      return (s_p == s) and (r == 0)
    elif self.is_terminal(s_p) and m <= n1 and -m <= n2:
      return (1 - self.req_cdf[0][n1 - m] * self.req_cdf[1][n2 + m]) * (r == 0)
    elif (n1_p < 0 or n2_p < 0 or not (0 <= abs(m) <= self.max_car_moves)
          or not (0 <= n1 <= self.max_car_cap)
          or not (0 <= n2 <= self.max_car_cap)
          or not (0 <= n1 - m) or not (0 <= n2 + m)
          or not (r in range(-move_cost, -move_cost + max_ben + 1, RENT_BEN))
          or s == ABSORBING_STATE):
      return 0

    def p_ret(n_p, n, req, moved_cars, loc):
      idx = n_p - (n - moved_cars) + req
      return ((self.ret_sf[loc][idx] + self.ret_pmfs[loc][idx])
              if n_p == self.max_car_cap else self.ret_pmfs[loc][idx])
    return sum([p_ret(n1_p, n1, k, m, 0)
                * p_ret(n2_p, n2, nb_sells - k, -m, 1)
                * self.req_pmfs[0][k] * self.req_pmfs[1][nb_sells - k]
                for k in range(nb_sells - n2 - m, n1 - m + 1)])

  def is_terminal(self, s):
    return not self.is_valid(s)
