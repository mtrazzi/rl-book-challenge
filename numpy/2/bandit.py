import numpy as np


class Bandit:
  def __init__(self, k=10, mean=0):
    self.k = k
    self.q = np.random.randn(k) + mean
    self.max_action = np.argmax(self.q)

  def reward(self, action):
    return np.random.normal(self.q[action])
