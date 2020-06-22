import numpy as np
from tdc import TDC


class EmphaticTD(TDC):
  def __init__(self, env, pi, b, w_dim, alpha, gamma, vhat, nab_vhat,
               M_0, I_0, feat):
    super().__init__(env, pi, b, w_dim, alpha, None, gamma, vhat, feat)
    self.M = M_0
    self.int = I_0
    self.nab_vhat = nab_vhat
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
      self.mu += 1
      self.w = (self.w + self.M * self.a *
                                np.mean([(self.exp_val_s(s) -
                                self.vhat(s, self.w)) *
                                self.nab_vhat(s, self.w)
                                for s in self.env.states], axis=0))
      self.M = self.g * self.M + self.int
    self.mu /= self.mu.sum()

  def reset(self):
    self.w = np.zeros(self.w_dim)
    self.mu = np.zeros(len(self.env.states))
