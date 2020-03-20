import numpy as np
from scipy.stats import skellam, poisson
from mdp import MDP
from utils import trans_id

REQUEST_LAMBDA = [3, 4]
RETURNS_LAMBDA = [3, 2]
CAR_MOVE_COST = 2
RENT_BEN = 10
NB_LOC = 2


class CarRentalEnv(MDP):
  def __init__(self, size):
    self.max_car_cap = size
    self.max_car_moves = max(self.max_car_cap // 4, 1)
    self.init_probs()
    super().__init__()

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
            for y in range(self.max_car_cap + 1)]

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
    self.req_pmfs = [poisson.pmf(j, sum(REQUEST_LAMBDA))
                     for j in sell_range]
    self.ret_pmfs = {i: [poisson.pmf(j, RETURNS_LAMBDA[i])
                     for j in sell_range] for i in range(NB_LOC)}
    self.ret_sf_pmfs = {i: [poisson.pmf(j, RETURNS_LAMBDA[i]) +
                            poisson.sf(j, RETURNS_LAMBDA[i])
                            for j in sell_range] for i in range(NB_LOC)}

  def _p(self, s_p, r, s, a):
    (n1, n2), (n1_p, n2_p), m = s, s_p, a
    move_cost = abs(m) * CAR_MOVE_COST
    max_ben = (n1 + n2) * RENT_BEN
    nb_sells = (r + move_cost) // RENT_BEN
    # print(f"nb_sells={nb_sells}, n1p={n1_p}, n2p={n2_p},n1={n1}, n2={n2}, m={m}, r={r}")
    if (n1_p < 0 or n2_p < 0 or not (0 <= abs(m) <= self.max_car_moves)
        or not (0 <= n1 <= self.max_car_cap)
        or not (0 <= n2 <= self.max_car_cap)
        or not (0 <= n1 - m) or not (0 <= n2 + m)
        or not (r in range(-move_cost, -move_cost + max_ben + 1, RENT_BEN))):
      # print(f"p({s_p},{r}|{s},{a}) = 0", end='')
      # if not (r in range(-move_cost, -move_cost + max_ben + 1, RENT_BEN)):
      #   print(f" (bc {r} not in {list(range(-move_cost, -move_cost + max_ben + 1, RENT_BEN))})")
      # else:
      #   print()
      return 0

    def p_ret(n_p, n, req, moved_cars, loc):
      if req > (n - moved_cars):
        # print(f"{req} >= ({n} - {moved_cars})")
        return 0
      idx = n_p - (n - moved_cars) + req
      # print(f"(loc {loc}) (idx){idx} = (n_p){n_p} - (n){n} + (req){req} + (moved_cars){moved_cars}")
      p_fn = self.ret_sf_pmfs if n_p == self.max_car_cap else self.ret_pmfs
      return p_fn[loc][idx]

    p_sells = self.req_pmfs[nb_sells]
    p_diff = sum([p_ret(n1_p, n1, k, m, 0)
                  * p_ret(n2_p, n2, nb_sells - k, -m, 1)
                  for k in range(nb_sells + 1)])
    # print(f"p({s_p},{r}|{s},{a}) = {p_sells * p_diff} (= {p_sells} * {p_diff})")
    return p_sells * p_diff

  def is_terminal(self, s):
    return not self.is_valid(s)
