from td import TD

class OffPolicyTD(TD):
  def __init__(self, env, V_init=None, step_size=0.1, pi=None, b=None, gamma=0.9):
    super().__init__(env, gamma, V_init, step_size)
    self.step_size = step_size
    self.V_init = V_init
    self.pi = pi
    self.b = b
    self.reset()

  def reset(self):
    super().reset()

  def generate_episode(self):
    return self.generate_traj(self.b, log_act=True)

  def find_value_function(self, n_episodes):
    for episode in range(1, n_episodes + 1):
      traj = self.generate_episode()
      G = 0
      W = 1
      for (i, (s, a, r)) in enumerate(traj[::-1][:-1]):
        s_p, _, _ = traj[i + 1]
        G = self.gamma * G + r
        is_ratio = self.pi[(a, s)] / self.b[(a, s)]
        td_err = self.td_error(s, r, s_p)
        self.V[s] += td_err * is_ratio 
