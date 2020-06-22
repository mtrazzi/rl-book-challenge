import numpy as np


class SemiGradDP:
  def __init__(self, env, pi, b, w_dim, alpha, gamma, vhat, nab_vhat):
    self.env = env
    self.pi = pi
    self.b = b
    self.a = alpha
    self.g = gamma
    self.w_dim = w_dim
    self.vhat = vhat
    self.nab_vhat = nab_vhat
    self.n_st = len(self.env.states)
    self.reset()

  def exp_val_s_a(self, s, a):
    return np.sum([self.env.p[(s_p, r, s, a)] * (r + self.g *
                                                 self.vhat(s_p, self.w))
                   for s_p in self.env.states for r in self.env.r])

  def exp_val_s(self, s):
    return np.sum([self.pi[(a, s)] * self.exp_val_s_a(s, a)
                  for a in self.env.moves])

  def pol_eva(self, n_sweeps):
    for _ in range(n_sweeps):
      self.w = self.w + ((self.a / self.n_st) *
                         np.sum([(self.exp_val_s(s) -
                                self.vhat(s, self.w)) *
                                self.nab_vhat(s, self.w)
                                for s in self.env.states], axis=0))

  def reset(self):
    self.w = np.zeros(self.w_dim)
