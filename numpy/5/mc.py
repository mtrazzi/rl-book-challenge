class MonteCarlo:
  def __init__(self, env, pi, gamma=0.9):
    self.env = env
    self.pi = pi
    self.gamma = gamma
    self.V = {s: 0 for s in env.states}

  def print_values(self):
    print(self.V)


class MonteCarloFirstVisit(MonteCarlo):
  def __init__(self, env, pi, gamma=0.9):
    super().__init__(env, pi, gamma)

  def first_visit_mc_prediction(self):
    pass
