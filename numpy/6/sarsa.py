import numpy as np
from td import TD

class Sarsa(TD):
  def __init__(self, env, step_size=0.1, gamma=1, eps=0.1):
    super().__init__(env, None, step_size, gamma) 
    self.eps = eps
    self.pol_deriv = pol_deriv if pol_deriv is not None else self.eps_gre(eps)
    self.reset() 

  def best_action(self, vals):
    return self.env.moves[np.random.choice(np.flatnonzero(vals == vals.max()))]

  def random_move(self):
    return self.env.moves[np.random.randint(len(self.env.moves))]

  def eps_gre(self, eps):
    def eps_gre_pol(s):
      if np.random.random() < self.eps:
        return self.best_action(np.array([Q[(s, a) for a in self.env.moves]))
      return self.random_move()
    return eps_gre_pol 

  def sarsa_update(self, s, a, r, s_p, a_p):
    self.Q[(s, a)] += self.step_size * (r + self.gamma * self.Q[(s_p, a_p)] - self.Q[(s, a)])

  def sarsa(self, n_episodes):
    for _ in range(n_episodes):
      s = self.env.reset()
      a = self.pol_deriv(s)
      while True:
        s_p, r, d, _ = self.env.step(a) 
        a_p = self.pol_deriv(s_p)
        self.sarsa_update(s, a, r, s_p, a_p)
        if d:
          break

  def reset(self):
    self.Q = {(s,a): 0 for s in self.env.states for a in self.env.moves}
    self.det_pi = { 
