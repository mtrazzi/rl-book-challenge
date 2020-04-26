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

  def q_learning_log_actions(self, n_episodes, to_log_s, to_log_a):
    per_l = []
    for ep_nb in range(n_episodes):
      s = self.env.reset()
      nb_a, nb_s = 0, 0
      while True:
        a = self.pol_deriv(s)
        s_p, r, d, _ = self.env.step(a) 
        nb_s += (s == to_log_s)
        nb_a += (a == to_log_a) * (s == to_log_s)
        a_p = self.greedy_pol(s_p)
        self.sarsa_update(s, a, r, s_p, a_p)
        if d:
          per_l.append(100 * (nb_a / nb_s))
          break
        s = s_p
    return per_l

  def reset(self):
    super().reset()
