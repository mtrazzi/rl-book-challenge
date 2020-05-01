#from  nstep_td import nStepTD
from off_pol_nstep_sarsa import OffPolnStepSarsa
import numpy as np

class OffPolnStepTD(OffPolnStepSarsa):
  def __init__(self, env, b=None, step_size=None, gamma=0.9, n=1, eps=0.1):
    super().__init__(env, b, step_size, gamma, n, eps)

  def nstep_return_is(self, ro, tau, T):
    n, S, V, R, g = self.n, self.S, self.V, self.R, self.gamma
    h = min(tau + n, T)
    G = V[S[h % (n + 1)]] if tau + n < T else 0
    t = h - 1
    while t >= tau:
      is_r = ro[t % n]
      tp1 = (t + 1) % (n + 1)
      G += is_r * (R[tp1] + g * G) + (1 - is_r) * self.V[S[tp1]]
      t -= 1
    return G

  def pol_eval(self, n_ep_train=100, pi=None):
    pi_learned = pi is None
    n, R, S, V = self.n, self.R, self.S, self.V
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
          taum = tau % (n + 1)
          G = self.nstep_return_is(ro, tau, T)
          V[S[taum]] += self.step_size * (G - V[S[taum]])
          if pi_learned:
            self.update_pi(S[taum])
        if tau == (T - 1):
          break
        t += 1
    return avg_length_l
