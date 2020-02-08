import numpy as np


class DynamicProgramming:
  def __init__(self, env, theta=1e-4, gamma=0.9):
    self.theta = theta
    self.env = env  # environment with transitions p
    self.V = {tuple(s): 0 for s in self.env.states}
    self.gamma = gamma

  def print_values(self):
    np.set_printoptions(2)
    to_print = np.zeros((self.env.size, self.env.size))
    for x in range(self.env.size):
      for y in range(self.env.size):
        to_print[x][y] = self.V[(x, y)]
    print(to_print)

  def policy_evaluation(self, pi):
    while True:
      delta = 0
      for s in self.env.states:
        v = self.V[tuple(s)]
        self.V[tuple(s)] = np.sum([pi(a, s) * np.sum(
                          [self.env.p(s_p, r, s, a) *
                            (r + self.gamma * self.V[tuple(s_p)])
                            for s_p in self.env.states for r in self.env.r])
                            for a in self.env.moves])
        delta = max(delta, abs(v-self.V[tuple(s)]))
      if delta < self.theta:
        break
    self.print_values()

  