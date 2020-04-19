import numpy as np
import copy

class TD:
  def __init__(self, env, V_init=None, step_size=None, gamma=0.9):
    self.env = env
    self.gamma = gamma
    self.V_init = V_init
    self.step_size = step_size
    self.reset()

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)), p=pi_dist)]
  
  def generate_traj(self, pi, log_act=False):
    s = self.env.reset()
    traj = []
    while True:
      a = self.sample_action(pi, s)
      s_p, r, done, _ = self.env.step(a)
      traj.append((s, r) if not log_act else (s, a, r))
      s = s_p
      if done:
        return traj + [(s_p, 0) if not log_act else (s_p, a, 0)]

  def td_update(self, s, r, s_p):
    self.V[s] += self.step_size * self.td_error(s, r, s_p)

  def td_error(self, s, r, s_p):
    return r + self.gamma * self.V[s_p] - self.V[s]
  
  def mc_error(self, s, G):
    return G - self.V[s]

  def get_value_list(self):
    return [val for key,val in self.V.items()]

  def reset(self):
    self.V = {s: 0 for s in self.env.states} if self.V_init is None else copy.deepcopy(self.V_init)

class OneStepTD(TD):
  def __init__(self, env, V_init=None, step_size=0.1, gamma=0.9):
    super().__init__(env, V_init, step_size, gamma)
    self.reset()
  
  def tabular_td_0(self, pi, n_episodes=1):
    for _ in range(n_episodes):
      traj = self.generate_traj(pi)
      for i in range(len(traj) - 1):
        (s, r), (s_p, _) = traj[i], traj[i + 1]
        self.td_update(s, r, s_p)

  def td_0_batch(self, pi, n_episodes=1):
    self.experience += [self.generate_traj(pi) for _ in range(n_episodes)]
    td_error_sum = {s: 0 for s in self.V}
    for traj in self.experience:
      for i in range(len(traj) - 1):
        (s, r), (s_p, _) = traj[i], traj[i + 1]
        td_error_sum[s] += self.td_error(s, r, s_p)
    for s in self.env.states[:-1]:
      self.V[s] += self.step_size * td_error_sum[s]

  def constant_step_size_mc(self, pi, n_episodes=1):
    for _ in range(n_episodes): 
      traj = self.generate_traj(pi)
      G = 0
      for (s, r) in traj[::-1]:
        G = r + self.gamma * G
        self.V[s] += self.step_size * self.mc_error(s, G)

  def constant_step_size_mc_batch(self, pi, n_episodes=1):
    self.experience += [self.generate_traj(pi) for _ in range(n_episodes)]
    def generate_G_traj(traj):
      G = 0
      G_traj = []
      for (_, r) in traj[::-1]:
        G = r + self.gamma * G
        G_traj = [G] + G_traj
      return G_traj
    n_past_traj = len(self.G_trajs)
    for i in range(n_episodes):
      self.G_trajs[n_past_traj + i] = generate_G_traj(self.experience[n_past_traj + i])
    mc_error_sum = {s: 0 for s in self.V}
    for (traj_idx, traj) in enumerate(self.experience):
      for i in range(len(traj) - 1):
        (s, r), (s_p, _) = traj[i], traj[i + 1]
        mc_error_sum[s] += self.mc_error(s, self.G_trajs[traj_idx][i])
    for s in self.env.states:
      self.V[s] += self.step_size * mc_error_sum[s]
 
  def reset(self): 
    super().reset()
    self.log = []
    self.experience = []
    self.G_trajs = {}
