from td import TD

class OffPolicyTD(TD):
  def __init__(self, env, V_init=None, step_size=0.1, pi=None, b=None, gamma=0.9):
    super().__init__(env, V_init, step_size, gamma)
    self.step_size = step_size
    self.V_init = V_init
    self.pi = pi
    self.b = b
    self.reset()

  def reset(self):
    super().reset()

  def generate_episode(self):
    return self.generate_traj(self.b, log_act=True)

  def off_policy_td_update(self, s, a, r, s_p):
    is_ratio = self.pi[(a, s)] / self.b[(a, s)] if self.pi[(a, s)] > 0 else 0
    td_err = self.td_error(s, r, s_p)
    self.V[s] += self.step_size * td_err * is_ratio

  def find_value_function(self, n_episodes):
    for episode in range(1, n_episodes + 1):
      traj = self.generate_episode()
      for i in range(len(traj) - 1):
        s, a, r = traj[i]
        s_p, _, _ = traj[i + 1]
        self.off_policy_td_update(s, a, r, s_p)
