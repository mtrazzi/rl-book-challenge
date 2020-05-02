from nstep_sarsa import nStepSarsa
import numpy as np

class OffPolnStepSarsa(nStepSarsa):
  def __init__(self, env, b=None, step_size=0.1, gamma=0.9, n=1, eps=0.1):
    super().__init__(env, step_size, gamma, n, eps)
    self.b = self.uniform_pol() if b is None else b
    assert(self.is_soft(self.b))
    assert(0 < self.step_size <= 1)
    assert(n is None or n >= 2)

  def is_soft(self, pol):
    for s in self.env.states:
      for a in self.env.moves_d[s]:
        if pol[(a, s)] == 0:
          return False
    return True 

  def uniform_pol(self):
    return {(a, s): 1 / len(self.env.moves_d[s]) for s in self.env.states for a in self.env.moves_d[s]}

  def get_nb_timesteps(self, pi, n_ep=1, max_steps=10000):
    count = 0
    for ep in range(n_ep): 
      s = self.env.reset()
      while True and count <= max_steps:
        s, _, d, _ = self.env.step(self.sample_action(pi, s))
        count += 1
        if d:
          break
    return count / n_ep
  
  def pol_eval(self, n_ep_train=100, pi=None):
    pi_learned = pi is None
    n, R, S, Q, A = self.n, self.R, self.S, self.Q, self.A
    ro = np.ones(n - 1)
    avg = None
    self.pi = self.initialize_pi() if pi_learned else pi
    avg_length_l = []
    for ep in range(n_ep_train):
      #len_sum = 0
      #for _ in range(10):
      #  len_sum += len(super().pol_eval(1, None))
      #avg_test_length = len_sum / 10
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
          max_idx = min(tau + n - 1, T - 1) % (n - 1)
          is_ratio = ro[(tau + 1) % (n-1):].prod() + ro[:max_idx + 1].prod()
          G = self.n_step_return_q(tau, T)
          taum = tau % (n + 1)
          s, a = S[taum], A[taum]
          Q[(s, a)] += self.step_size * is_ratio * (G - Q[(s, a)])
          if pi_learned:
            self.update_pi(s)
        ro[(t + 1) % (n - 1)] = self.pi[(A[tp1m], S[tp1m])] / self.b[(A[tp1m], S[tp1m])]
        if tau == (T - 1):
          break
        t += 1
    return avg_length_l
