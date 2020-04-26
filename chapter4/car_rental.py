from scipy.stats import poisson
from mdp import MDP
import numpy as np

REQUEST_LAMBDA = [3, 4]
RETURNS_LAMBDA = [3, 2]
CAR_MOVE_COST = 2
RENT_BEN = 10
NB_LOC = 2
ABSORBING_STATE = (-1, -1)
PARKING_COST = 4


class CarRentalEnv(MDP):
  def __init__(self, size, ex_4_7=False):
    self.max_car_cap = size
    self.max_car_moves = self.max_car_cap // 5 + 1
    self.init_probs()
    self.ex_4_7 = ex_4_7  # is the pb modified for exercise 4.7 or not
    # possible nb of cars can go from 0 to max car capacity
    self.size = self.max_car_cap + 1
    self.get_moves()
    self.get_states()
    self.get_r()
    super().__init__()

  def get_moves(self):
    self.moves = list(range(-self.max_car_moves, self.max_car_moves + 1))

  def get_states(self):
    self.states = [(x, y) for x in range(self.max_car_cap + 1)
            for y in range(self.max_car_cap + 1)] + [ABSORBING_STATE]

  def get_r(self):
    if self.ex_4_7:
      self.r = np.unique([self.compute_reward(n1, n2, m, car_sold)
                        for n1 in range(self.max_car_cap + 1)
                        for n2 in range(self.max_car_cap + 1)
                        for m in range(-self.max_car_moves,
                                       self.max_car_moves + 1)
                        for car_sold in range(self.max_car_cap * NB_LOC + 1)])
    self.r = [-CAR_MOVE_COST * car_moves + RENT_BEN * car_sold
            for car_moves in range(self.max_car_moves + 1)
            for car_sold in range(self.max_car_cap * NB_LOC + 1)]

  def init_probs(self):
    """Computing some probabilities in advance to make it go faster."""
    sell_range = range(self.max_car_cap * NB_LOC + 1)
    self.req_pmfs = {i: [poisson.pmf(j, REQUEST_LAMBDA[i])
                     for j in sell_range] for i in range(NB_LOC)}
    self.req_cdf = {i: [poisson.cdf(j, REQUEST_LAMBDA[i])
                    for j in sell_range] for i in range(NB_LOC)}
    self.ret_pmfs = {i: [poisson.pmf(j, RETURNS_LAMBDA[i])
                     for j in sell_range] for i in range(NB_LOC)}
    self.ret_sf_pmfs = {i: [poisson.sf(j, RETURNS_LAMBDA[i]) +
                        self.ret_pmfs[i][j] for j in sell_range]
                        for i in range(NB_LOC)}
    self.req_pmfs_prod = {(k, n_sells - k): (self.req_pmfs[0][k] *
                                             self.req_pmfs[1][n_sells - k])
                          for n_sells in sell_range
                          for k in range(n_sells + 1)}

  def move_cost(self, m):
    """Check the cost of doing m, depending on if we're in ex4.7."""
    if not self.ex_4_7:
      return abs(m) * CAR_MOVE_COST
    return CAR_MOVE_COST * abs(m - 1 if m > 0 else m)

  def is_possible_reward(self, n1, n2, m, r, move_cost, max_ben):
    """Check if reward is possible for given state and problem."""
    if not self.ex_4_7:
      return r in range(-move_cost, -move_cost + max_ben + 1, RENT_BEN)
    for car_sold in range(self.max_car_cap * NB_LOC + 1):
      if self.compute_reward(n1, n2, m, car_sold) == r:
        return True
    return False

  def park_cost(self, n1, n2, m):
    if not self.ex_4_7:
      return 0
    return (PARKING_COST * ((n1 + m) >= self.max_car_cap or
                            (n2 - m) >= self.max_car_cap))

  def nb_car_sold(self, n1, n2, m, r):
    """Compute the number of car sold, according to the given reward."""
    return (r + self.park_cost(n1, n2, m) + self.move_cost(m)) // RENT_BEN

  def compute_reward(self, n1, n2, m, car_sold):
    """
    Compute the reward associated with an action according to the description
    of exercise 4.7.
    """
    return RENT_BEN * car_sold - (self.park_cost(n1, n2, m) + self.move_cost(m))

  def _p(self, s_p, r, s, a):
    """Transition function defined in private because p dictionary in mdp.py."""
    (n1, n2), (n1_p, n2_p), m = s, s_p, a
    move_cost = self.move_cost(m)
    max_ben = (n1 + n2) * RENT_BEN
    if self.is_terminal(s):
      return (s_p == s) and (r == 0)
    elif self.is_terminal(s_p) and m <= n1 and -m <= n2:
      return (1 - self.req_cdf[0][n1 - m] * self.req_cdf[1][n2 + m]) * (r == 0)
    elif not ((0 <= n1 - m <= self.max_car_cap)
              and (0 <= n2 + m <= self.max_car_cap)
              and self.is_possible_reward(n1, n2, m, r, move_cost, max_ben)):
      return 0

    nb_sells = self.nb_car_sold(n1, n2, m, r)

    def p_ret(n_p, n, req, moved_cars, loc):
      idx = n_p - (n - moved_cars) + req
      if req > n:
        return 0
      # print(f"{idx} = {n_p} - ({n} - {moved_cars}) + {req}")
      return ((self.ret_sf_pmfs[loc][idx])
              if n_p == self.max_car_cap else self.ret_pmfs[loc][idx])
    return sum([p_ret(n1_p, n1, k, m, 0)
                * p_ret(n2_p, n2, nb_sells - k, -m, 1)
                * self.req_pmfs_prod[(k, nb_sells - k)]
                for k in range(nb_sells + 1)])

  def is_terminal(self, s):
    return s == ABSORBING_STATE
