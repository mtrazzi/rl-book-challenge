import numpy as np


class Bandit:
  def __init__(self, k=10, mean=0):
    self.k = k
    self.q = np.random.randn(k) + mean
    self.old_q = [q_val for q_val in self.q]  # copy

  def max_action(self):
    return np.argmax(self.q)

  def reward(self, action):
    return np.random.normal(self.q[action])

  def reset(self):
    self.q = self.old_q
