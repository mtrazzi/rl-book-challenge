import numpy as np


class DynamicProgramming:
  def __init__(self, env, theta=1e-4, gamma=0.9):
    self.theta = theta
    self.env = env  # environment with transitions p
    self.V = {tuple(s): 0 for s in self.env.states}
    self.gamma = 0.9

  def policy_evaluation(self, pi):
    delta = 0
    for s in self.env.states:
      v = 0
      self.V[tuple(s)] = np.sum([pi(a, s) * np.sum(
                        [self.env.p(s_p, r, s, a) *
                         (r + self.gamma * self.V[tuple(s_p)])
                         for s_p in self.env.states for r in self.env.r])
                         for a in self.env.moves])
      delta = max(delta, abs(v-self.V[tuple(s)]))
      if delta < self.theta:
        break
