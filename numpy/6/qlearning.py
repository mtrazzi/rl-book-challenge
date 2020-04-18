from sarsa import Sarsa

class QLearning(Sarsa): 
  def __init__(self, env, step_size=0.1, gamma=1, eps=0.1, pol_deriv=None):
    super().__init__(env, step_size, gamma, eps, pol_deriv) 
    self.greedy_pol = self.eps_gre(eps=0)
    self.reset()
    print(f"gamma={self.gamma}") 
    print(f"eps={eps}") 
    print(f"step_size={self.step_size}") 

  def q_learning(self, n_episodes):
    r_sum_l = []
    for ep_nb in range(n_episodes):
      s = self.env.reset()
      r_sum = 0
      while True:
        a = self.pol_deriv(s)
        s_p, r, d, _ = self.env.step(a) 
        r_sum += r
        a_p = self.greedy_pol(s_p)
        self.sarsa_update(s, a, r, s_p, a_p)
        if d:
          r_sum_l.append(r_sum)
          break
        s = s_p
    return r_sum_l

  def reset(self):
    super().reset()
