import numpy as np
from utils import gen_traj


class ReinforceBaseline:
  def __init__(self, env, alpha_w, alpha_t, gamma, the_d, pi_gen, log_pi, vhat, nab_vhat, w_dim, the_0=None):
    self.env = env
    self.a_w = alpha_w
    self.a_t = alpha_t
    self.g = gamma
    self.the_d = the_d
    self.pi_gen = pi_gen
    self.log_pi = log_pi
    self.the_0 = the_0
    self.vhat = vhat
    self.nab_vhat = nab_vhat
    self.w_dim = w_dim
    self.reset()

  def train(self, n_ep):
    env, w, a_w, a_t, the, g = self.env, self.w, self.a_w, self.a_t, self.the, self.g
    log_pi, pi_gen, vhat, nab_vhat = self.log_pi, self.pi_gen, self.vhat, self.nab_vhat
    tot_rew_l = []
    to_plot = []
    for _ in range(n_ep):
      pi = pi_gen(env, the)
      S, A, R = zip(*gen_traj(env, pi))
      T = len(S)
      g_l = [g ** k for k in range(T)]
      for t in range(T):
        s, a = S[t], A[t]
        G = np.dot(g_l[:T-t], R[:T-t])
        delt = G - vhat(s, w)
        w += self.a_w * delt * nab_vhat(s, w)
        the += alp * g_l[t] * G * log_pi(a, s, pi)
      tot_rew_l.append(np.sum(R))
      to_plot.append(the[0]) 
    return tot_rew_l

  def reset(self):
    self.the = np.random.randn(self.the_d) if self.the_0 is None else self.the_0
    self.w = np.zeros(self.w_dim)

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)
