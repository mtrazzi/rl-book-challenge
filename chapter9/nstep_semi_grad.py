import numpy as np

class nStepSemiGrad:
  def __init__(self, env, alpha, w_dim, gamma, n):
    self.env = env
    self.a = alpha
    self.n = n
    self.d = w_dim
    self.g = gamma
    self.reset()

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves_d[s]]
    return self.env.moves_d[s][np.random.choice(np.arange(len(self.env.moves_d[s])), p=pi_dist)]

  def get_r_values(self, R, i, j):
    n = self.n
    orig_mod = mod_idx = i % (n + 1)
    goal = j % (n + 1)
    R_vals = []
    while True:
      R_vals.append(R[mod_idx])
      mod_idx = (mod_idx + 1) % (n + 1)
      if mod_idx == goal:
        return R_vals

  def n_step_return(self, vhat, tau, T):
    S, R, n, g_l = self.S, self.R, self.n, self.g_l
    max_idx = min(tau + n, T)
    r_vals = self.get_r_values(R, tau + 1, max_idx + 1)
    G = np.dot(g_l[:max_idx-tau], r_vals)
    if tau + n < T:
      G = G + g_l[n] * vhat(S[(tau + n) % (n + 1)], self.w)
    return G

  def pol_eva(self, pi, vhat, nab_vhat, n_ep, gamma):
    self.g_l = [gamma ** k for k in range(self.n + 1)]
    n, R, S = self.n, self.R, self.S
    for ep in range(n_ep):
      S[0] = self.env.reset()
      T = np.inf
      t = 0
      while True:
        if t < T:
          S[(t + 1) % (n + 1)], R[(t + 1) % (n + 1)], d, _ = self.env.step(self.sample_action(pi, S[t % (n + 1)]))
          if d:
            T = t + 1
        tau = t - n + 1
        if tau >= 0:
          s = S[tau % (n + 1)]
          G = self.n_step_return(vhat, tau, T)
          self.w += self.a * (G - vhat(s, self.w)) * nab_vhat(s, self.w)
        if tau == (T - 1):
          break
        t += 1

  def get_value_list(self, vhat):
    return [vhat(s, self.w) for s in self.env.states]

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.S = [None for _ in range(self.n + 1)]
    self.R = [None for _ in range(self.n + 1)]
    self.w = np.zeros(self.d)
