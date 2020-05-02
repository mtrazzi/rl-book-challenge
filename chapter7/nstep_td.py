import numpy as np
import copy

class TD:
  def __init__(self, env, V_init=None, step_size=None, gamma=0.9, n=1):
    self.env = env
    self.gamma = gamma
    self.V_init = V_init
    self.step_size = step_size
    self.n = n
    self.reset()

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves_d[s]]
    return self.env.moves_d[s][np.random.choice(np.arange(len(self.env.moves_d[s])), p=pi_dist)]
  
  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)
 
  def get_value_list(self):
    return [val for key,val in self.V.items()]
  
  def reset(self):
    self.V = {s: 0 for s in self.env.states} if self.V_init is None else copy.deepcopy(self.V_init)

class nStepTD(TD):
  def __init__(self, env, V_init=None, step_size=None, gamma=0.9, n=1, ex_7_2=False):
    super().__init__(env, V_init, step_size, gamma, n)
    self.return_f = self.n_step_return if not ex_7_2 else self.td_err_sum
    self.reset()

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

  def pol_eval(self, pi, n_ep):
    n, R, S, V = self.n, self.R, self.S, self.V
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
          G = self.return_f(tau, T)
          V[s] += self.step_size * (G - V[s])
        if tau == (T - 1):
          break
        t += 1

  def n_step_return(self, tau, T):
    n = self.n
    max_idx = min(tau + n, T)
    r_vals = self.get_r_values(self.R, tau + 1, max_idx + 1)
    G = np.dot(self.gamma_l[:max_idx-tau], r_vals)
    if tau + n < T:
      G = G + self.gamma_l[n] * self.V[self.S[(tau + n) % (n + 1)]]
    return G

  def td_error(self, t):
    n = self.n
    s, s_p = self.S[t % (n + 1)], self.S[(t + 1) % (n + 1)]
    r = self.R[(t + 1) % (n + 1)]
    return r + self.gamma * self.V[s_p] - self.V[s]

  def td_err_sum(self, tau, T):
    max_idx = min(tau + self.n, T)
    return sum(self.td_error(j) for j in range(tau, max_idx))

  def reset(self):
    self.gamma_l = [self.gamma ** k for k in range(self.n + 1)]
    self.S = [None for _ in range(self.n + 1)]
    self.R = [None for _ in range(self.n + 1)]
    super().reset() 
