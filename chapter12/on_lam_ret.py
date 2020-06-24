import numpy as np


class OnLamRet:
  def __init__(self, env, alpha, w_dim, lam, vhat, nab_vhat, gamma):
    self.env = env
    self.a = alpha
    self.lam = lam
    self.d = w_dim
    self.vhat = vhat
    self.nab_vhat = nab_vhat
    self.g = gamma
    self.reset()

  def get_r_values(self, R, i, j):
    return R[i:j]

  def n_step_return(self, t, T, w_idx):
    S, R, n, g_l = self.S, self.R, self.n, self.g_l
    max_idx = min(t + n, T - 1)
    r_vals = self.get_r_values(R, t + 1, max_idx + 1)
    G = np.dot(g_l[:max_idx-t], r_vals)
    if t + n < T:
      G = G + g_l[n] * self.vhat(S[t + n], self.w_l[w_idx])
    return G

  def lam_ret(self, t, h, w_idx):
    n_steps = h - t - 1
    self.g_l = [self.g ** k for k in range(n_steps + 2)]
    G, lam_fac = 0, 1
    for n in range(1, n_steps + 1):
      self.n = n
      G += lam_fac * self.n_step_return(t, h, w_idx)
      lam_fac *= self.lam
    self.n = n_steps + 1
    to_ret = (1 - self.lam) * G + lam_fac * self.n_step_return(t, h, w_idx)
    return to_ret

  def pol_eva(self, pi, n_ep, max_steps=np.inf):
    self.R, self.S = [], []
    for ep in range(n_ep):
      self.reset_weight_tab()
      if ep > 0 and ep % 1 == 0:
        print(f"ep #{ep}")
      self.S.append(self.env.reset())
      t = 0
      while t < max_steps:
        s_p, r, d, _ = self.env.step(self.S[t])
        self.S.append(s_p)
        self.R.append(r)
        if d:
          break
        t += 1
        self.w_l.append(np.zeros(self.d))
        for (k, s) in enumerate(self.S[:-1]):
          self.w_l[k + 1] = (self.w_l[k] +
                             self.a * (self.lam_ret(k, t, k)
                                       - self.vhat(s, self.w_l[k]))
                                    * self.nab_vhat(s, self.w_l[k]))

  def get_value_list(self):
    return [self.vhat(s, self.w_l[-1]) for s in self.env.states]

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset_weight_tab(self):
    self.w_l = [np.zeros(self.d) if self.w_l is None else [self.w_l[-1]]]

  def reset(self):
    self.w_l = None
