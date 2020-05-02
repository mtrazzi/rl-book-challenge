from off_pol_nstep_sarsa import OffPolnStepSarsa
import numpy as np

class OffPolnStepExpSarsa(OffPolnStepSarsa):
  def __init__(self, env, b=None, step_size=None, gamma=0.9, n=1, eps=0.1):
    super().__init__(env, b, step_size, gamma, n, eps)

  def exp_val(self, s):
    return sum(self.pi[(a, s)] * self.Q[(s, a)] for a in self.env.moves_d[s])

  def nstep_return_is(self, ro, tau, T):
    n, S, A, Q, R, g = self.n, self.S, self.A, self.Q, self.R, self.gamma
    h = min(tau + n, T)
    hm = h % (n + 1)
    G = Q[(S[hm], A[hm])] if tau + n < T else R[hm]
    t = h - 1
    while t >= tau:
      is_r = ro[t % n]
      tp1 = (t + 1) % (n + 1)
      G += R[tp1] + g * (G - Q[(S[tp1], A[tp1])]) + g * self.exp_val(S[tp1])
      t -= 1
    return G

  def pol_eval(self, n_ep_train=100, pi=None):
    pi_learned = pi is None
    n, R, S, A, Q = self.n, self.R, self.S, self.A, self.Q
    ro = np.ones(n)
    avg = None
    self.pi = self.initialize_pi() if pi_learned else pi
    avg_length_l = []
    for ep in range(n_ep_train):
      print(ep)
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
          a = self.sample_action(self.b, S[tm])
          ro[t % n] = self.pi[(a, S[tm])] / self.b[(a, S[tm])]
          S[tp1m], R[tp1m], d, _ = self.env.step(a)
          if d:
            T = t + 1
          else:
            A[tp1m] = self.sample_action(self.b, S[tp1m])
        tau = t - n + 1
        if tau >= 0:
          taum = tau % (n + 1)
          G = self.nstep_return_is(ro, tau, T)
          Q[(S[taum], A[taum])] += self.step_size * (G - Q[(S[taum], A[taum])])
          if pi_learned:
            self.update_pi(S[taum])
        if tau == (T - 1):
          break
        t += 1
    return avg_length_l
