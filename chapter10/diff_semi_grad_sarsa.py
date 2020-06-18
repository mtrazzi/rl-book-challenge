from semi_grad_sarsa import GradientAlg


class DiffSemiGradientSarsa(GradientAlg):
  def __init__(self, env, alpha, beta, w_dim, eps):
    super().__init__(env, alpha, w_dim, eps)
    self.b = beta

  def pol_eva(self, qhat, nab_qhat, n_steps):
    a, b, w, self.qhat = self.a, self.b, self.w, qhat
    r_mean = 0
    s = self.env.reset()
    a = self.eps_gre(s)
    for _ in range(n_steps):
      if n_steps > 0 and n_steps % 100 == 0:
        print(n_steps, "steps")
      s_p, r, _, _ = self.env.step(a)
      a_p = self.eps_gre(s_p)
      delt = r - r_mean + qhat(s_p, a_p, w) - qhat(s, a, w)
      r_mean += b * delt
      w += a * delt * nab_qhat(s, a, w)
      s = s_p
      a = a_p
