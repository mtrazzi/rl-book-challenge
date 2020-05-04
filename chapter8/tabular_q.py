from utils import sample

class TabularQ:
  def __init__(self, model, alpha, gamma):
    self.model = model
    self.a = alpha
    self.g = gamma
    self.reset()

  def rand_sam_one_step_pla(self, n_updates=1, decay=False):
    decay_rate = (1 - 1 / n_updates) if decay else None
    for _ in range(n_updates):
      s = sample(self.model.states)
      moves = self.model.moves_d[s]
      a = sample(moves) 
      s_p, r = self.model.sample_s_r(s, a)
      Q_max = max(self.Q[(s_p, a_p)] for a_p in moves)
      self.Q[(s, a)] += self.a * (r + self.g * Q_max - self.Q[(s, a)])
      if decay:
        self.a *= decay_rate

  def get_V(self):
    return {s: max(self.Q[(s, a)] for a in self.model.moves_d[s]) for s in self.model.states}

  def reset(self):
    self.Q = {(s,a): 0.0 for s in self.model.states for a in self.model.moves_d[s]}
