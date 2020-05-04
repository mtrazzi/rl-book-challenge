from sample_model import SampleModel
from utils import sample

class TabularQ:
  def __init__(self, env, alpha, gamma):
    self.env = env
    self.a = alpha
    self.g = gamma
    self.reset()

  def random_sample_one_step_planning(self, n_updates=100, decay=False):
    model = SampleModel(self.env)
    decay_rate = (1 - 1 / n_updates) if decay else None
    for _ in range(n_updates):
      s = sample(self.env.states)
      moves = self.env.moves_d[s]
      a = sample(moves) 
      s_p, r = model.sample_s_r(s, a)
      Q_max = max(self.Q[(s_p, a_p)] for a_p in moves)
      self.Q[(s, a)] += self.a * (r + self.g * Q_max - self.Q[(s, a)])
      if decay:
        self.a *= decay_rate

  def get_V(self):
    return {s: max(self.Q[(s, a)] for a in self.env.moves_d[s]) for s in self.env.states}

  def reset(self):
    self.Q = {(s,a): 0.0 for s in self.env.states for a in self.env.moves_d[s]}
