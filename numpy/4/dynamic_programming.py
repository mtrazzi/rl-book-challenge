import matplotlib.pyplot as plt
import numpy as np
from car_rental import CarRentalEnv


class DynamicProgramming:
  """
  Dynamic Programming algorithms to run the gridworld and car_rental
  examples (fig 4.1 and 4.2).
  """
  def __init__(self, env, pi=None, det_pi=None, theta=1e-4, gamma=0.9):
    self.theta = theta
    self.env = env
    self.V = {tuple(s): 0 for s in self.env.states}
    self.gamma = gamma
    self.pi_init = {} if pi is None else pi
    self.initialize_deterministic_pi(det_pi)
    self.compute_pi_vects()

  def initialize_deterministic_pi(self, det_pi_dict=None):
    """Initializes a deterministic policy pi."""
    if det_pi_dict is None or not det_pi_dict:
      det_pi_dict = {s: self.env.moves[np.random.randint(len(self.env.moves))]
                     for s in self.env.states}
    self.pi = {(a, s): int(a == det_pi_dict[s]) for a in self.env.moves
               for s in self.env.states}

  def compute_pi_vects(self):
    """Initializing vectors for pi(.|s) for faster policy evaluation."""
    pi = self.pi if not self.pi_init else self.pi_init
    self.pi_vect = {s: [pi[(a, s)] for a in self.env.moves]
                    for s in self.env.states}

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
    pol_range = list(range(np.min(transposed_Z), np.max(transposed_Z) + 1))
    CS = ax.contour(X, Y, Z, colors='k', levels=pol_range)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Figure 4.2')
    plt.show()

  def print_policy(self):
    if isinstance(self.env, CarRentalEnv):
      self.print_policy_car_rental()
    else:
      self.print_policy_gridworld()

  def print_values(self, show_matplotlib=False):
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
      if show_matplotlib:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        (X, Y), Z = np.meshgrid(idxs, idxs), np.array(to_print).T
        ax.set_xlabel('# of cars at second location', fontsize=10)
        ax.set_ylabel('# of cars at first location', fontsize=10)
        ax.set_xticks([idxs[0], idxs[-1]])
        ax.set_yticks([idxs[0], idxs[-1]])
        ax.set_zticks([np.min(Z), np.max(Z)])
        ax.plot_surface(X, Y, Z)
        plt.show()
      print(np.array(to_print_term))
    else:
      print(np.array(to_print))

  def expected_value(self, s, a):
    V_vect = np.array([self.V[s_p] for s_p in self.env.states])
    return (np.dot(self.env.r, self.env.pr[(s, a)])
            + self.gamma * np.dot(V_vect, self.env.psp[(s, a)]))

  def policy_evaluation(self):
    """Updates V according to current pi."""
    self.compute_pi_vects()  # for faster policy evaluation
    while True:
      delta = 0
      for s in self.env.states:
        v = self.V[s]
        expected_values = [self.expected_value(s, a) for a in self.env.moves]
        self.V[s] = np.dot(self.pi_vect[s], expected_values)
        delta = max(delta, abs(v-self.V[s]))
      if delta < self.theta:
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
    counter = 0
    while True and counter < max_iter:
      self.policy_evaluation()
      if self.policy_improvement():
        return self.V, self.pi
      counter += 1
