import numpy as np


class MonteCarlo:
  def __init__(self, env, pi, gamma=0.9):
    self.env = env
    self.pi = pi
    self.gamma = gamma
    self.V = {s: 0 for s in env.states}

  def print_values(self):
    print(self.V)

  def sample_action(self, s):
    pi_dist = [self.pi[(a, s)] for a in self.env.moves]
    return np.random.choice(s, p=pi_dist)

  def generate_trajectory(self):
    trajs = []
    s = self.env.reset()
    a = self.sample_action(s)
    while True:
      s_p, r, done, _ = self.env.step(a)
      trajs.append((s, a, r))
      s = s_p
      if done:
        break


class MonteCarloFirstVisit(MonteCarlo):
  def __init__(self, env, pi, gamma=0.9):
    super().__init__(env, pi, gamma)

  def first_visit_mc_prediction(self):
    pass
