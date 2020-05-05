from utils import sample
import numpy as np

class TabularQ:
  def __init__(self, model, alpha, gamma):
    self.model = model
    self.a = alpha
    self.g = gamma
    self.Q = {}
    self.reset_q()

  def rand_sam_one_step_pla(self, n_updates=1, decay=False):
    decay_rate = (1 - 1 / n_updates) if decay else None
    S, A = self.model.states, self.model.moves_d
    s_list = list(S)
    a_dict = {s: list(A[s]) for s in s_list}
    for _ in range(n_updates):
      s = sample(s_list)
      a = sample(a_dict[s]) 
      s_p, r = self.model.sample_s_r(s, a)
      Q_max = max(self.Q[(s_p, a_p)] for a_p in a_dict[s])
      self.Q[(s, a)] += self.a * (r + self.g * Q_max - self.Q[(s, a)])
      if decay:
        self.a *= decay_rate

  def get_V(self):
    return {s: max(self.Q[(s, a)] for a in self.model.moves_d[s]) for s in self.model.states}

  def seed(self, seed):
    np.random.seed(seed) 

  def reset_q(self):
    for s in self.model.states:
      for a in self.model.moves_d[s]:
        self.Q[(s, a)] = 0
