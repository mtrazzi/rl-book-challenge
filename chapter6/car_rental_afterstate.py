import numpy as np

REQ_LAM = [3, 4]
RET_LAM = [3, 2]
CAR_MOVE_COST = 2
RENT_BEN = 10
NB_LOC = 2
S_ABS = (-1, -1)
PARKING_COST = 4

class CarRentalAfterstateEnv:
  def __init__(self, size, ex_4_7=False):
    self.max_car_cap = size
    self.max_car_moves = self.max_car_cap // 5 + 1
    self.init_probs()
    self.ex_4_7 = ex_4_7
    self.size = self.max_car_cap + 1
    self.get_states()
    self.get_moves()
    self.get_moves_d()
    self.get_r()
    #self.compute_p()

  def get_moves(self):
    self.moves = list(range(-self.max_car_moves, self.max_car_moves + 1))

  def get_moves_d(self):
    self.moves_d = {}
    for (n1, n2) in self.states:
      if (n1, n2) == S_ABS:
        self.moves_d[(n1, n2)] = [0]
        continue
      max_move_right = min([abs(n1), self.max_car_cap - n2, self.max_car_moves])
      max_move_left = min([abs(n2), self.max_car_cap - n1, self.max_car_moves])
      self.moves_d[(n1, n2)] = np.arange(-max_move_left, max_move_right + 1)

  def get_states(self):
    self.states = [(x, y) for x in range(self.max_car_cap + 1)
            for y in range(self.max_car_cap + 1)] + [S_ABS]

  def after_state(self, s, a):
    (n1, n2) = s
    #print(f"{str(s)} --({a})--> {str((n1 - a, n2 + a))}")
    return (n1 - a, n2 + a)

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

  def poisson_dist(self, n, lamb):
    exp_term = np.exp(-lamb)
    mult = exp_term
    fact = 1
    dist = [mult / fact]
    dist_sum = dist[0]
    for i in range(1, n):
      mult *= lamb
      fact *= i
      frac = mult / fact
      dist.append(frac)
      dist_sum += frac
    dist.append(1 - dist_sum) 
    return dist

  def init_probs(self):
    self.ret_dis = {i: self.poisson_dist(self.max_car_cap, RET_LAM[i])
                     for i in range(NB_LOC)}
    self.req_dis = {i: self.poisson_dist(self.max_car_cap, REQ_LAM[i])
                     for i in range(NB_LOC)}

  def move_cost(self, m):
    if not self.ex_4_7:
      return abs(m) * CAR_MOVE_COST
    return CAR_MOVE_COST * abs(m - 1 if m > 0 else m)

  def park_cost(self, n1, n2, m):
    if not self.ex_4_7:
      return 0
    return (PARKING_COST * ((n1 + m) >= self.max_car_cap or
                            (n2 - m) >= self.max_car_cap))

  def compute_reward(self, n1, n2, m, car_sold):
    return RENT_BEN * car_sold - (self.park_cost(n1, n2, m) + self.move_cost(m))

  def sample(self, distrib):
    return np.random.choice(np.arange(len(distrib)), p=distrib) 

  def step(self, a):
    (n1, n2), m = self.state, a
    (ret_1, ret_2), (req_1, req_2) = [[self.sample(dis[i]) for i in [0, 1]] 
                                      for dis in [self.ret_dis, self.req_dis]]
    car_sold = req_1 + req_2
    n1_p, n2_p = n1 - m + ret_1 - req_1, n2 + m + ret_2 - req_2
    done = n1_p < 0 or n2_p < 0
    r = self.compute_reward(n1, n2, m, car_sold) if not done else 0
    s_p = S_ABS if done else tuple(map(lambda x: min(x, self.max_car_cap), (n1_p, n2_p)))
    self.state = s_p
    return s_p, r, done, {}

  def reset(self):
    self.state = self.states[np.random.randint(len(self.states))]
    return self.state

  def seed(self, seed):
    np.random.seed(seed)

  def compute_p(self, n_iter=100):
    self.counts = {(s_p, r, s, a): 0
                  for s in self.states for a in self.moves_d[s]
                  for s_p in self.states for r in self.r}
    count = 0
    to_do = sum(len(self.moves_d[s]) for s in self.states)
    for s in self.states:
      for a in self.moves_d[s]:
        print(f"{int(100 * (count / to_do))}%")
        count += 1
        for _ in range(n_iter):
          self.state = s
          s_p, r, _, _ = self.step(a)
          self.counts[(s_p, r, s, a)] += 1

    def p_sum(d, s_p_list, r_list, s_list, a_list):
      return np.sum([d[(s_p, r, s, a)] for s_p in s_p_list
                     for r in r_list for s in s_list for a in a_list])

    self.psa = {(s, a): p_sum(self.counts, self.states, self.r, [s], [a]) for s in self.states for a in self.moves_d[s]}
    self.p = {(s_p, r, s, a): (self.counts[(s_p, r, s, a)] / self.psa[(s, a)] if self.psa[(s, a)] != 0 else 0)
               for s in self.states for a in self.moves_d[s]
               for s_p in self.states for r in self.r}

    self.pr = {(s, a): np.array([p_sum(self.p, self.states, [r], [s], [a])
               for r in self.r]) for s in self.states for a in self.moves_d[s]}
    self.psp = {(s, a): np.array([p_sum(self.p, [s_p], self.r, [s], [a])
                    for s_p in self.states])
                    for s in self.states for a in self.moves_d[s]}


  def is_terminal(self, s):
    return s == S_ABS
