from off_pol_nstep_sarsa import OffPolnStepSarsa
import numpy as np

class OffPolnStepTD(OffPolnStepSarsa):
  def __init__(self, env, b=None, step_size=None, gamma=0.9, n=1, eps=0.1, simple=False):
    super().__init__(env, b, step_size, gamma, n, eps)
    self.update_fn = self.simple_update if simple else self.off_pol_update
    self.reset()

  def off_pol_update(self, s, ro, tau, T):
    G = self.nstep_return_is(ro, tau, T)
    self.V[s] += self.step_size * (G - self.V[s])

  def simple_ro(self, ro, t, h, T):
    prod, A, S = 1, self.A, self.S
    idx = min(h, T - 1)
    while idx >= t:
      idxm = idx % (self.n)
      prod *= ro[idxm]
      idx -= 1
    return prod

  def simple_update(self, s, ro, tau, T):
    G = self.n_step_return(tau, T)
    is_r = self.simple_ro(ro, tau, tau + self.n - 1, T)
    self.V[s] += self.step_size * is_r * (G - self.V[s])

  def nstep_return_is(self, ro, tau, T):
    n, S, V, R, g = self.n, self.S, self.V, self.R, self.gamma
    h = min(tau + n, T)
    G = V[S[h % (n + 1)]] if tau + n < T else 0
    t = h - 1
    while t >= tau:
      tm, tp1 = t % n, (t + 1) % (n + 1)
      is_r = ro[tm]
      G = is_r * (R[tp1] + g * G) + (1 - is_r) * self.V[S[tm]]
      t -= 1
    return G

  def pol_eval(self, n_ep_train=100, pi=None):
    pi_learned = pi is None
    n, R, S, V = self.n, self.R, self.S, self.V
    avg = None
    self.pi = self.initialize_pi() if pi_learned else pi
    for ep in range(n_ep_train):
      ro = np.ones(n)
      S[0] = self.env.reset()
      T = np.inf
      t = 0
      while True:
        tm, tp1m = t % (n + 1), (t + 1) % (n + 1)
        if t < T:
          a = self.sample_action(self.b, S[tm])
          ro[t % n] = self.pi[(a, S[tm])] / self.b[(a, S[tm])]
          S[tp1m], R[tp1m], d, _ = self.env.step(a)
          if d:
            T = t + 1
        tau = t - n + 1
        if tau >= 0:
          s = S[tau % (n + 1)]
          self.update_fn(s, ro, tau, T)
          if pi_learned:
            self.update_pi(s)
        if tau == (T - 1):
          break
        t += 1
    return self.get_value_list()

  def reset(self):
    super().reset()
