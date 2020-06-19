import numpy as np
from semi_grad_sarsa import GradientAlg


class nStepSemiGradSarsa(GradientAlg):
  def __init__(self, env, alpha, w_dim, eps, n):
    self.n = n
    super().__init__(env, alpha, w_dim, eps)
    self.reset()

  def get_r_values(self, R, i, j):
    n = self.n
    mod_idx = i % (n + 1)
    goal = j % (n + 1)
    R_vals = []
    while True:
      R_vals.append(R[mod_idx])
      mod_idx = (mod_idx + 1) % (n + 1)
      if mod_idx == goal:
        return R_vals

  def n_step_return(self, qhat, tau, T):
    S, A, R, n, g_l, w = self.S, self.A, self.R, self.n, self.g_l, self.w
    max_idx = min(tau + n, T)
    r_vals = self.get_r_values(R, tau + 1, max_idx + 1)
    G = np.dot(g_l[:max_idx-tau], r_vals)
    if tau + n < T:
      taum = (tau + n) % (n + 1)
      G = G + g_l[n] * qhat(S[taum], A[taum], w)
    return G

  def pol_eva(self, pi, qhat, nab_qhat, n_ep, gamma, max_steps=np.inf):
    def act(s): return self.eps_gre(s) if pi is None else pi(s)
    self.g_l = [gamma ** k for k in range(self.n + 1)]
    self.qhat = qhat
    n, R, S, A, w = self.n, self.R, self.S, self.A, self.w
    steps_per_ep = []
    for ep in range(n_ep):
      if ep > 0 and ep % 1 == 0:
        print(f"ep #{ep}")
      S[0] = self.env.reset()
      A[0] = act(S[0])
      T = np.inf
      t = 0
      while True and t < max_steps:
        if t < T:
          tp1m, tm = (t + 1) % (n + 1), t % (n + 1)
          S[tp1m], R[tp1m], d, _ = self.env.step(act(S[tm]))
          if d:
            T = t + 1
          else:
            A[tp1m] = act(S[tp1m])
        tau = t - n + 1
        if tau >= 0:
          taum = tau % (n + 1)
          s, a = S[taum], A[taum]
          G = self.n_step_return(qhat, tau, T)
          w += self.a * (G - qhat(s, a, w)) * nab_qhat(s, a, w)
        if tau == (T - 1):
          break
        t += 1
      steps_per_ep.append(t + 1)
    return steps_per_ep

  def reset(self):
    self.S = [None for _ in range(self.n + 1)]
    self.R = [None for _ in range(self.n + 1)]
    self.A = [None for _ in range(self.n + 1)]
    super().reset()
