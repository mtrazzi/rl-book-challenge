import matplotlib.pyplot as plt
import numpy as np
from utils import trans_id
import time
from car_rental import CarRentalEnv
from itertools import product

class DynamicProgramming:
  """
  Dynamic Programming algorithms to run the gridworld and car_rental
  examples (fig 4.1 and 4.2).
  """
  def __init__(self, env, pi=None, theta=1e-4, gamma=0.9):
    self.theta = theta
    self.env = env  # environment with transitions p
    self.V = {tuple(s): 0 for s in self.env.states}
    self.gamma = gamma
    self.pi_init = {} if pi is None else pi  # initial policy (optional)
    self.pi = {}  # deterministic pi

  def initialize_deterministic_pi(self, arb_d=None):
    """Initializes a deterministic policy pi."""
    print(arb_d)
    if arb_d is None or not arb_d:
      arb_d = {s: self.env.moves[np.random.randint(len(self.env.moves))]
               for s in self.env.states}
    for s in self.env.states:
      for a in self.env.moves:
        self.pi[(a, s)] = int(a == arb_d[s])

  def print_policy_gridworld(self):
    to_print = [[None] * self.env.size for _ in range(self.env.size)]
    max_length = max([len(move_name) for move_name in self.env.moves])
    for x in range(self.env.size):
      for y in range(self.env.size):
        to_print[x][y] = str(self.deterministic_pi((x, y))).ljust(max_length)
    print("printing policy gridworld")
    print(*to_print, sep='\n')

  def print_policy_car_rental(self):
    fig, ax = plt.subplots()
    X = Y = list(range(self.env.size))
    Z = [[self.deterministic_pi((x, y)) for y in Y] for x in X]
    print("printing policy car rental")
    transposed_Z = [[Z[self.env.size - x - 1][y] for y in Y] for x in X]
    print(*transposed_Z, sep='\n')
    CS = ax.contour(X, Y, transposed_Z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Figure 4.2')
    plt.show()

  def print_policy(self):
    if isinstance(self.env, CarRentalEnv):
      self.print_policy_car_rental()
    else:
      self.print_policy_gridworld()

  def print_values(self):
    np.set_printoptions(2)
    size = self.env.size
    to_print = np.zeros((size, size))
    idxs = list(range(size))
    for x in idxs:
      for y in idxs:
        to_print[x][y] = self.V[(x, y)]
    print("printing value function V")
    if isinstance(self.env, CarRentalEnv):
      to_print_term = [[to_print[size - x - 1][y] for y in idxs] for x in idxs]
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      # coords = list(product(idxs, idxs))
      # def get_j(crds, j): return [crd[j] for crd in crds]
      # X, Y, Z = get_j(coords, 0), get_j(coords, 1), np.ravel(np.array(to_print))  
      (X, Y), Z = np.meshgrid(idxs, idxs), np.array(to_print).T
      ax.set_xlabel('# of cars at second location', fontsize=10)
      ax.set_ylabel('# of cars at first location', fontsize=10)
      ax.set_xticks([idxs[0], idxs[-1]])
      ax.set_yticks([idxs[0], idxs[-1]])
      ax.set_zticks([np.min(Z), np.max(Z)])
      import ipdb; ipdb.set_trace()
      ax.plot_surface(X, Y, Z)
      plt.show()
      print(to_print_term)
    else:
      print(np.array(to_print))

  def expected_value(self, s, a):
    ev = np.sum([self.env.p[trans_id(s_p, r, s, a)] *
                (r + self.gamma * self.V[s_p])
                for s_p in self.env.states for r in self.env.r])
    # print(*[f"({s_p}, {r}|{s},{a}) {self.env.p[trans_id(s_p, r, s, a)]} * ({r} + {self.gamma} * {self.V[s_p]})"
                # for s_p in self.env.states for r in self.env.r], sep="\n")
    return ev

  def policy_evaluation(self):
    """Updates V according to current pi."""
    counter = 0
    while True:
      counter += 1
      # print(f"at the start of iteration #{counter}")
      delta = 0
      # self.print_values()
      for s in self.env.states:
        v = self.V[s]
        self.V[s] = np.sum([self.pi[(a, s)] * self.expected_value(s, a)
                            for a in self.env.moves])
        # print([f"({a}, {s}): {self.pi[(a, s)]} * {self.expected_value(s, a)}"
        #                        for a in self.env.moves], sep=' ')
        delta = max(delta, abs(v-self.V[s]))
      if delta < self.theta:# or counter >= 100:
        break

  def deterministic_pi(self, s):
    return self.env.moves[np.argmax([self.pi[(a, s)] for a in self.env.moves])]

  def update_pi(self, s, a):
    """Sets pi(a|s) = 1 and pi(a'|s) = 0 for a' != a."""
    for a_p in self.env.moves:
      self.pi[(a_p, s)] = (a == a_p)

  def policy_improvement(self):
    """Improves pi according to current V. Returns True if policy is stable."""
    policy_stable = True
    for s in self.env.states:
      old_action = self.deterministic_pi(s)
      self.update_pi(s, self.env.moves[np.argmax([self.expected_value(s, a)
                     for a in self.env.moves])])
      policy_stable = policy_stable and (old_action == self.deterministic_pi(s))
    return policy_stable

  def policy_iteration(self, max_iter=np.inf):
    self.initialize_deterministic_pi(self.pi_init)

    counter = 0
    # self.print_policy()
    # self.print_values()
    while True and counter < max_iter:
      print(f"counter={counter}")
      start = time.time()
      self.policy_evaluation()
      print(f"evaluation took {time.time()-start}s")
      # self.print_values()
      start = time.time()
      if self.policy_improvement():
        return self.V, self.pi
      print(f"improvement took {time.time()-start}s")
      # self.print_policy()   
      counter += 1
