from  nstep_td import nStepTD

class OffPolnStepTD(nStepTD):
  def __init__(self, env, b, V_init=None, step_size=None, gamma=0.9, n=1):
    super().__init__(env, V_init, step_size, gamma)
    self.b = self.uniform_pol() if b is None else b
    assert(self.is_soft(self.b))

  def is_soft(self, pol):
    for s in self.env.states:
      for a in self.env.moves_d[s]:
        if pol[(a, s)] == 0:
          return False
    return True 

  def uniform_pol(self):
    return {(a, s): 1 / len(self.env.moves_d[s]) for s in self.env.states for a in self.env.moves_d[s]}

  def nstep_return_is(self, ro, tau, T):

  def pol_eval(self, n_ep_train=100, pi=None):
    pi_learned = pi is None
    n, R, S, V = self.n, self.R, self.S, self.V
    ro = np.ones(n)
    moving_avg = None
    self.pi = self.initialize_pi() if pi_learned else pi
    avg_length_l = []
    for ep in range(n_ep_train):
      len_sum = 0
      for _ in range(10):
        len_sum += len(super().pol_eval(1, None))
      avg_test_length = len_sum / 10
      moving_avg = avg_test_length if moving_avg is None else 0.2 * avg_test_length + 0.8 * moving_avg
      avg_length_l.append(moving_avg)
      print(f"nb_timesteps after {ep} train episodes ~= {moving_avg} timesteps")
      S[0] = self.env.reset()
      T = np.inf
      t = 0
      while True:
        tm, tp1m = t % (n + 1), (t + 1) % (n + 1)
        if t < T:
          a = self.sample_action(self.b, S[tm])
          ro[t % n] = self.pi[(a, S[tm])] / self.b[(a, S[tm])]
          S[tp1m], R[tp1m], d, _ = self.env.step(A[tm])
          if d:
            T = t + 1
        tau = t - n + 1
        if tau >= 0:
          taum = tau % (n + 1)
          G = self.nstep_return_is(ro, tau, T)
          V[S[taum]] += self.step_size * (G - V[S[taum]])
          if pi_learned:
            self.update_pi(s)
        if tau == (T - 1):
          break
        t += 1
    return avg_length_l
  
  def reset():
    super().reset()
