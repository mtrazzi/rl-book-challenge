import numpy as np


class Bandit:
  def __init__(self, k=10):
    self.k = k
    self.q = np.random.randn(k)
    self.max_action = max(self.q)

  def reward(self, action):
    return np.random.normal(self.q[action])
