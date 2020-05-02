from nstep_td import nStepTD
import numpy as np
import time

class nStepSarsa(nStepTD):
  def __init__(self, env, step_size=0.1, gamma=0.9, n=1, eps=0.1, exp_sar=False): 
    super().__init__(env, None, step_size, gamma, n)
    self.update_pi = self.update_policy(eps)
    self.exp_sar = exp_sar
    self.reset()

  def eps_gre(self, eps):
    def eps_gre_pol(s):
      if np.random.random() < eps:
        return self.random_move(s)
      q_arr = np.array([self.Q[(s, a)] for a in self.env.moves_d[s]])
      return self.env.moves_d[s][np.random.choice(np.flatnonzero(q_arr == q_arr.max()))]
    return eps_gre_pol
 
  def update_policy(self, eps):
    def update_on_q_values(s):
      best_a = self.eps_gre(0)(s)
      moves = self.env.moves_d[s]
      soft_min = eps / len(moves)
      for a in moves:
        self.pi[(a, s)] = soft_min + (1 - eps) * (a == best_a)
    return update_on_q_values

  def initialize_pi(self):
    self.pi = {}
    for s in self.env.states:
      self.update_pi(s)
    return self.pi
 
  def exp_val(self, s):
    return sum(self.pi[(a, s)] * self.Q[(s, a)] for a in self.env.moves_d[s])

  def n_step_return_q(self, tau, T):
    n = self.n
    max_idx = min(tau + n, T)
    r_vals = self.get_r_values(self.R, tau + 1, max_idx + 1)
    G = np.dot(self.gamma_l[:max_idx-tau], r_vals)
    if tau + n < T:
      tau_p_n = (tau + n) % (n + 1)
      s, a = self.S[tau_p_n], self.A[tau_p_n]
      last_term = self.Q[(s, a)] if not self.exp_sar else self.exp_val(s)
      G = G + self.gamma_l[n] * last_term
    return G

  def pol_eval(self, n_ep=100, pi=None):
    n, R, S, Q, A = self.n, self.R, self.S, self.Q, self.A
    pi_learned = pi is None
    self.pi = self.initialize_pi() if pi_learned else pi
    ep_per_t = [] 
    for ep in range(n_ep):
      S[0] = self.env.reset()
      A[0] = self.sample_action(self.pi, S[0])
      T = np.inf
      t = 0
      while True:
        ep_per_t.append(ep)
        tm, tp1m = t % (n + 1), (t + 1) % (n + 1)
        if t < T:
          S[tp1m], R[tp1m], d, _ = self.env.step(A[tm])
          if d:
            T = t + 1
          else:
            A[tp1m] = self.sample_action(self.pi, S[tp1m])
        tau = t - n + 1
        if tau >= 0:
          G = self.n_step_return_q(tau, T)
          taum = tau % (n + 1)
          s, a = S[taum], A[taum]
          Q[(s, a)] += self.step_size * (G - Q[(s, a)])
          if pi_learned:
            self.update_pi(s)
        if tau == (T - 1):
          break
        t += 1
    return ep_per_t


  def get_v(self):
    return {s: max(self.Q[(s, a)] for a in self.env.moves_d[s]) for s in self.env.states} 

  def reset(self):
    super().reset()
    self.Q = {(s, a): 0 for s in self.env.states for a in self.env.moves_d[s]}
    self.A = [None for _ in range(self.n + 1)]
