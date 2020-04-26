import numpy as np

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
  def __init__(self, env, V_init=None, step_size=None, gamma=0.9, n=1):
    super().__init__(env, V_init, step_size, gamma)
    self.gamma_l = [self.gamma ** k for k in range(n + 1)]
    self.reset()

  def get_r_values(self, R, i, j):
    orig_mod = mod_idx = i % self.n 
    R_vals = []
    while True:
      R_vals.append(R[mod_idx])
      mod_idx = (mod_idx + 1) % self.n
      if mod_idx == orig_mod:
        return R_vals

  def pol_eval(self, pi, n_ep):
    n, R, S, gamma_l, V = self.n, self.R, self.S, self.gamma_l, self.V
    for _ in range(n_ep):
      S[0] = self.env.reset()
      T = np.inf
      t = 0
      while True:
        if t < T:
          S[(t + 1) % n], R[(t + 1) % n], d, _ = self.env.step(self.sample_action(pi, S[t % n]))
          if d:
            T = t + 1
        tau = t - n + 1
        if tau >= 0:
          idx = min(tau + n, T)
          G = np.dot(gamma_l[:idx-tau], self.get_r_values(R, tau, idx))
          if tau + n < T:
            G = G + self.gamma_l[n] * V[S[(tau + n) % n]]
          V[S[tau % n]] += self.step_size * (G - V[S[tau % n]])
        if tau == (T - 1):
          break
        t += 1

  def simple_td(self, pi, n_ep):
    for _ in range(n_ep):
      s = self.env.reset()
      while True:
        s_p, r, d, _ = self.env.step(self.sample_action(pi, s))
        self.V[s] += self.step_size * (r + self.V[s_p] - self.V[s])
        s = s_p
        if d:
          break

  def reset(self):
    self.S = [None for _ in range(self.n)]
    self.R = [None for _ in range(self.n)]
    super().reset() 
