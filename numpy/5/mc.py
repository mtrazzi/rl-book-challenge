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
    return np.random.choice(self.env.moves, p=pi_dist)

  def generate_trajectory(self):
    trajs = []
    s = self.env.reset()
    while True and len(trajs) < 10:
      print(self.env)
      a = self.sample_action(s)
      s_p, r, done, _ = self.env.step(a)
      trajs.append((s, a, r))
      s = s_p
      if done:
        break
    print(trajs)
    return trajs


class MonteCarloFirstVisit(MonteCarlo):
  def __init__(self, env, pi, gamma=0.9):
    super().__init__(env, pi, gamma)
    self.returns = {s: [] for s in env.states}

  def first_visit_mc_prediction(self):
    trajs = self.generate_trajectory()
    G = 0
    states = [s for (s, _, _) in trajs]
    for (i, (s, a, r)) in enumerate(trajs[::-1]):
      G = self.gamma * G + r
      if s not in states[:-(i + 1)]:  # logging only first visits
        self.returns[s].append(G)
        self.V[s] = np.mean(self.returns[s])
