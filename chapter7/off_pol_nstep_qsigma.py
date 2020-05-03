from off_pol_nstep_sarsa import OffPolnStepSarsa

class OffPolnStepQSigma(OffPolnStepSarsa):
  def __init__(self, env, sigma_f, step_size, gamma, n, b=None):
    super().__init__(env, b, step_size, gamma, n)
    self.sigma_f = lambda s: sigma_f if isinstance(sigma_f, float) else sigma_f
    self.reset()

  def pol_eval(self, n_ep=100, pi=None):
    n, g, R, S, Q, A = self.n, self.gamma, self.R, self.S, self.Q, self.A
    ro, sigma = [np.ones(n + 1) for _ in range(2)]
    pi_learned = pi is None
    avg = None
    self.pi = self.initialize_pi() if pi_learned else pi
    avg_length_l = []
    for ep in range(n_ep):
      ep_len = self.get_nb_timesteps(self.pi, 10)
      avg = ep_len if avg is None else 0.2 * ep_len + 0.8 * avg
      avg_length_l.append(avg)
      print(f"nb_timesteps after {ep} train episodes ~= {avg} timesteps")
      S[0] = self.env.reset()
      A[0] = self.sample_action(self.b, S[0])
      T = np.inf
      t = 0
      while True:
        tm, tp1m = t % (n + 1), (t + 1) % (n + 1)
        if t < T:
          S[tp1m], R[tp1m], d, _ = self.env.step(A[tm])
          if d:
            T = t + 1
          else:
            A[tp1m] = self.sample_action(self.b, S[tp1m])
        tau = t - n + 1
        if tau >= 0:
          tmax = min(t + 1, T)
          tmaxm = tmax % (n + 1)
          G = R[tmaxm] + (tmax == T) * g * (self.exp_Q(tmaxm))
          for k in range(tmax - 1, tau, -1):
            km = k % (n + 1)
            s, a, r = S[km], A[km], R[km]
            G = r + g * (self.exp_Q(km, a) + self.pi[(a, s)] * G)
          s, a = S[tau % (n + 1)], A[tau % (n + 1)]
          Q[(s, a)] += self.step_size * (G - Q[(s, a)])
          if pi_learned:
            self.update_pi(s)
        if tau == (T - 1):
          break
        t += 1
    return avg_length_l

  def reset(self):
    super().reset()
