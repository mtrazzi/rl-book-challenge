import matplotlib.pyplot as plt
import numpy as np
from car_rental import CarRentalEnv
from gambler import GamblerEnv


class DynamicProgramming:
  """
  Dynamic Programming algorithms to run the gridworld and car_rental
  examples (fig 4.1 and 4.2).
  """
  def __init__(self, env, pi=None, det_pi=None, theta=1e-4, gamma=0.9):
    self.theta = theta
    self.env = env
    self.V = {s: 0 for s in env.states}
    # vect for vectorized computation
    self.V_vect = np.array([self.V[s] for s in self.env.states]).astype(float)
    self.gamma = gamma
    self.pi = self.initialize_deterministic_pi(det_pi) if pi is None else pi
    self.compute_pi_vects()
    # expected reward of s, a
    self.er = {(s, a): np.dot(env.r, env.pr[(s, a)]) for s in env.states
               for a in env.moves}
    # Q is for if we want to use action values instead of state values
    self.Q = {(s, a): 0 for s in self.env.states for a in self.env.moves}
    self.Q_vect = {s: np.array([self.Q[(s, a)]
                   for a in self.env.moves]).astype(float)
                   for s in self.env.states}

  def initialize_deterministic_pi(self, det_pi_dict=None):
    """Initializes a deterministic policy pi."""
    if det_pi_dict is None or not det_pi_dict:
      det_pi_dict = {s: self.env.moves[np.random.randint(len(self.env.moves))]
                     for s in self.env.states}
    return {(a, s): int(a == det_pi_dict[s]) for a in self.env.moves
            for s in self.env.states}

  def compute_pi_vects(self):
    """Initializing vectors for pi(.|s) for faster policy evaluation."""
    self.pi_vect = {s: [self.pi[(a, s)] for a in self.env.moves]
                    for s in self.env.states}

  def print_policy_gridworld(self):
    to_print = [[None] * self.env.size for _ in range(self.env.size)]
    max_length = max([len(move_name) for move_name in self.env.moves])
    for x in range(self.env.size):
      for y in range(self.env.size):
        to_print[x][y] = str(self.deterministic_pi((x, y))).ljust(max_length)
    print("printing policy gridworld")
    print(*to_print, sep='\n')

  def print_policy_car_rental(self, title='Figure 4.2'):
    fig, ax = plt.subplots()
    X = Y = list(range(self.env.size))
    Z = [[self.deterministic_pi((x, y)) for y in Y] for x in X]
    print("printing policy car rental")
    transposed_Z = [[Z[self.env.size - x - 1][y] for y in Y] for x in X]
    print(*transposed_Z, sep='\n')
    pol_range = list(range(np.min(transposed_Z), np.max(transposed_Z) + 1))
    CS = ax.contour(X, Y, Z, colors='k', levels=pol_range)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title(title)
    plt.show()

  def print_policy_gambler(self, title='Figure 4.3'):
    plt.plot([self.deterministic_pi(s) for s in self.env.states[1:-1]])
    plt.title(title)
    plt.show()

  def print_policy(self):
    if isinstance(self.env, CarRentalEnv):
      self.print_policy_car_rental()
    elif isinstance(self.env, GamblerEnv):
      self.print_policy_gambler()
    else:
      self.print_policy_gridworld()

  def print_values(self, show_matplotlib=False, title="Figure 4.3"):
    if isinstance(self.env, GamblerEnv):
      plt.plot([self.V[s] for s in self.env.states[1:-1]])
      plt.title(title)
      plt.show()
      return
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

  def print_Q_values(self):
    for s in self.env.states:
      self.V[s] = np.dot(self.pi_vect[s], self.Q_vect[s])
      print(f"{s}:", end=' ')
      print(*[f"{a}: {self.Q[(s, a)]}" for a in self.env.moves])
    self.print_values()
    self.print_policy()

  def expected_value(self, s, a, arr):
    return self.er[(s, a)] + self.gamma * np.dot(arr, self.env.psp[(s, a)])

  def policy_evaluation(self):
    """Updates V according to current pi."""
    self.compute_pi_vects()  # for faster policy evaluation
    while True:
      delta = 0
      for s in self.env.states:
        v = self.V[s]
        expected_values = [self.expected_value(s, a, self.V_vect)
                           for a in self.env.moves]
        bellman_right_side = np.dot(self.pi_vect[s], expected_values)
        self.V[s] = self.V_vect[self.env.states.index(s)] = bellman_right_side
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
      a_old = self.deterministic_pi(s)
      ev = np.array([self.expected_value(s, a, self.V_vect)
                    for a in self.env.moves])
      a_new = self.env.moves[np.random.choice(np.flatnonzero(ev == ev.max()))]
      self.update_pi(s, a_new)
      policy_stable = policy_stable and (a_old == a_new)
    return policy_stable

  def policy_iteration(self):
    while True:
      self.policy_evaluation()
      if self.policy_improvement():
        return self.V, self.pi

  def policy_iteration_improved(self):
    past_policies_str = [str(self.pi)]
    while True:
      self.policy_evaluation()
      policy_stable = self.policy_improvement()
      pol_str = str(self.pi)
      if policy_stable or pol_str in past_policies_str:
        return self.V, self.pi
      past_policies_str.append(pol_str)

  def policy_evaluation_Q(self):
    """Updates Q according to current pi."""
    self.compute_pi_vects()  # for faster policy evaluation
    while True:
      delta = 0
      for s in self.env.states:
        # computing the sum over a_p of pi(a_p|s_p)Q(s_p, a_p)
        # in advance so Q doesn't change between updates
        expected_Q = [np.dot(self.pi_vect[s_p], self.Q_vect[s_p])
                      for s_p in self.env.states]
        for a in self.env.moves:
          q = self.Q[(s, a)]
          self.Q[(s, a)] = self.expected_value(s, a, expected_Q)
          self.Q_vect[s][self.env.moves.index(a)] = self.Q[(s, a)]
          delta = max(delta, abs(q-self.Q[(s, a)]))
      if delta < self.theta:
        break

  def policy_improvement_Q(self):
    """Improves pi according to current Q. Returns True if policy is stable."""
    policy_stable = True
    for s in self.env.states:
      a_old = self.deterministic_pi(s)
      best_actions = np.flatnonzero(self.Q_vect[s] == self.Q_vect[s].max())
      a_new = self.env.moves[np.random.choice(best_actions)]
      self.update_pi(s, a_new)
      policy_stable = policy_stable and (a_old == a_new)
    return policy_stable

  def policy_iteration_Q(self):
    """Policy iteration using Q values."""
    while True:
      self.policy_evaluation_Q()
      if self.policy_improvement_Q():
        return self.Q, self.pi

  def value_iteration(self):
    import time
    start = time.time()
    while True:
      delta = 0
      for s in self.env.states:
        v = self.V[s]
        expected_values = [self.expected_value(s, a, self.V_vect)
                           for a in self.env.moves]
        self.V[s] = self.V_vect[self.env.states.index(s)] = max(expected_values)
        delta = max(delta, abs(v-self.V[s]))
      if delta < self.theta:
        break
    print(f"finished value iteration after {time.time()-start}s")
    self.policy_improvement()
