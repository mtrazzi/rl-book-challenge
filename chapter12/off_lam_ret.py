import numpy as np

class OffLamRet:
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

  def n_step_return(self, t, T):
    S, R, n, g_l, w = self.S, self.R, self.n, self.g_l, self.w
    max_idx = min(t + n, T - 1)
    r_vals = self.get_r_values(R, t + 1, max_idx + 1)
    G = np.dot(g_l[:max_idx-t], r_vals)
    if t + n < T:
      G = G + g_l[n] * self.vhat(S[t + n], w)
    return G

  def lam_ret(self, t, T):
    n_steps = T - t - 1
    self.g_l = [self.g ** k for k in range(n_steps + 2)]
    G, lam_fac = 0, 1
    for n in range(1, n_steps + 1):
      # print(f"[t={t}, n={n}]")
      self.n = n
      G += lam_fac * self.n_step_return(t, T)
      # print(f"nstep return was: {self.n_step_return(t, T)}")
      lam_fac *= self.lam
    self.n = n_steps + 1
    to_ret = (1 - self.lam) * G + lam_fac * self.n_step_return(t, T)
    # print(f"lambda return is then {to_ret}")
    return to_ret

  def pol_eva(self, pi, n_ep, max_steps=np.inf):
    self.R, self.S = [], []
    for ep in range(n_ep):
      if ep > 0 and ep % 1 == 0:
        print(f"ep #{ep}")
      self.S.append(self.env.reset())
      T = np.inf
      t = 0
      while t < max_steps:
        s_p, r, d, _ = self.env.step(self.S[t])
        self.S.append(s_p)
        self.R.append(r)
        if d:
          T = t + 1
          break
        t += 1
      # w_sum = np.zeros_like(self.w, dtype=np.float64)
      # print(f"R={R} (len(R)={len(self.R)})")
      # np.set_printoptions(1)
      for (t, s) in enumerate(self.S[:-1]):
        # import ipdb; ipdb.set_trace()
        # print(f"w={self.w}")
        self.w += (self.a * (self.lam_ret(t, T) - self.vhat(s, self.w))
                          * self.nab_vhat(s, self.w))

  def get_value_list(self):
    return [self.vhat(s, self.w) for s in self.env.states]

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.d)
