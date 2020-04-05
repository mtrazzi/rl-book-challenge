import numpy as np


class MonteCarlo:
  def __init__(self, env, pi=None, det_pi=None, gamma=0.9):
    self.env = env
    self.pi = pi
    self.det_pi = det_pi
    self.gamma = gamma
    self.reset()

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def print_values(self):
    print(self.V)

  def sample_action(self, s, det=True):
    if not det:
      pi_dist = [self.pi[(a, s)] for a in self.env.moves]
      return np.random.choice(self.env.moves, p=pi_dist)
    else:
      return self.det_pi[s]

  def generate_trajectory(self, start_state=None, det=True):
    trajs = []
    s = self.env.reset() if start_state is None else start_state
    if start_state is not None:
      self.env.force_state(start_state)
    while True:
      a = self.sample_action(s, det)
      s_p, r, done, _ = self.env.step(a)
      trajs.append((s, a, r))
      s = s_p
      if done:
        break
    return trajs

  def update_pi(self, s, a):
    """Sets pi(a|s) = 1 and pi(a'|s) = 0 for a' != a."""
    for a_p in self.env.moves:
      self.pi[(a_p, s)] = (a == a_p)

  def update_det_pi(self, s, a):
    self.det_pi[s] = a

  def estimate_V_from_Q(self):
    for s in self.env.states:
      self.V[s] = max(self.Q[(s, a)] for a in self.env.moves)

  def reset(self):
    self.V = {s: 0 for s in self.env.states}
    self.Q = {(s, a): 0 for s in self.env.states for a in self.env.moves}


class MonteCarloFirstVisit(MonteCarlo):
  def __init__(self, env, pi=None, det_pi=None, gamma=0.9):
    super().__init__(env, pi, det_pi, gamma)
    self.returns = {s: [] for s in env.states}
    self.return_counts = {key: 0 for key in self.returns.keys()}

  def first_visit_mc_prediction(self, n_episodes, start_state=None):
    for _ in range(n_episodes):
      trajs = self.generate_trajectory(start_state=start_state, det=False)
      G = 0
      states = [s for (s, _, _) in trajs]
      for (i, (s, a, r)) in enumerate(trajs[::-1]):
        G = self.gamma * G + r
        if s not in states[:-(i + 1)]:  # logging only first visits
          self.returns[s].append(G)
          self.return_counts[s] += 1
          self.V[s] += (1 / self.return_counts[s]) * (G - self.V[s])


class MonteCarloES(MonteCarlo):
  def __init__(self, env, pi=None, det_pi=None, gamma=0.9):
    """Monte Carlo Exploring Starts (page 99)."""
    super().__init__(env, pi, det_pi, gamma)
    self.returns = {(s, a): [] for s in env.states for a in env.moves}
    self.return_counts = {key: 0 for key in self.returns.keys()}

  def exploring_starts(self):
    """Returns a state-action pair such that all have non-zero probability."""
    def random_choice(l): return l[np.random.randint(len(l))]
    return map(random_choice, (self.env.states, self.env.moves))

  def generate_trajectory_exploring_starts(self, det=True):
    s, a = self.exploring_starts()
    self.env.force_state(s)
    s_p, r, done, _ = self.env.step(a)
    first_step = [(s, a, r)]
    if done:
      return first_step
    return first_step + self.generate_trajectory(start_state=s_p, det=det)

  def estimate_optimal_policy(self, n_episodes):
    for _ in range(n_episodes):
      trajs = self.generate_trajectory_exploring_starts(det=True)
      G = 0
      pairs = [(s, a) for (s, a, _) in trajs]
      for (i, (s, a, r)) in enumerate(trajs[::-1]):
        G = self.gamma * G + r
        if (s, a) not in pairs[:-(i + 1)]:  # logging only first visits
          self.returns[(s, a)].append(G)
          self.return_counts[(s, a)] += 1
          self.Q[(s, a)] += ((1 / self.return_counts[(s, a)]) *
                             (G - self.Q[(s, a)]))
          val = np.array([self.Q[(s, a)] for a in self.env.moves])
          a_max_idx = np.random.choice(np.flatnonzero(val == val.max()))
          self.update_det_pi(s, self.env.moves[a_max_idx])


class OnPolicyFirstVisitMonteCarlo(MonteCarlo):
  def __init__(self, env, pi=None, det_pi=None, gamma=0.9, epsilon=0.1):
    super().__init__(env, pi, None, gamma)
    self.returns = {(s, a): [] for s in env.states for a in env.moves}
    self.return_counts = {key: 0 for key in self.returns.keys()}
    self.epsilon = epsilon

  def update_pi_soft(self, s, a_max):
    n_act = len(self.env.moves)
    for a in self.env.moves:
      self.pi[(a, s)] = self.epsilon / n_act + (1 - self.epsilon) * (a == a_max)

  def estimate_optimal_policy(self, n_episodes):
    for _ in range(n_episodes):
      trajs = self.generate_trajectory(det=False)
      G = 0
      pairs = [(s, a) for (s, a, _) in trajs]
      for (i, (s, a, r)) in enumerate(trajs[::-1]):
        G = self.gamma * G + r
        if (s, a) not in pairs[:-(i + 1)]:  # logging only first visits
          self.returns[(s, a)].append(G)
          self.return_counts[(s, a)] += 1
          self.Q[(s, a)] += ((1 / self.return_counts[(s, a)]) *
                             (G - self.Q[(s, a)]))
          val = np.array([self.Q[(s, a)] for a in self.env.moves])
          a_max_idx = np.random.choice(np.flatnonzero(val == val.max()))
          self.update_pi_soft(s, self.env.moves[a_max_idx])


class OffPolicyMC(MonteCarlo):
  def __init__(self, env, pi, weighted=True, b=None, gamma=0.9):
    super().__init__(env, pi, None, gamma)
    self.C = {(s, a): 0 for s in env.states for a in env.moves}
    self.b = pi if b is None else b
    self.pi = b  # because self.pi used in generate_trajectory
    self.target = pi
    # are we using weighted important sampling or ordinary important sampling
    self.weighted = weighted
    self.visit_counts = {key: 0 for key in self.C.keys()}

  def target_estimate(self, s):
    return sum(self.target[(a, s)] * self.Q[(s, a)] for a in self.env.moves)

  def reset(self):
    super().reset()
    self.C = {(s, a): 0 for s in self.env.states for a in self.env.moves}


class OffPolicyMCPrediction(OffPolicyMC):
  def __init__(self, env, pi, weighted=True, b=None, gamma=1):
    super().__init__(env, pi, weighted, b, gamma)
    self.estimates = []

  def Q_step(self, s, a, W, G):
    if self.weighted:
      return (W / self.C[(s, a)]) * (G - self.Q[(s, a)])
    else:
      return (1 / self.visit_counts[(s, a)]) * (W * G - self.Q[(s, a)])

  def policy_evaluation(self, n_episodes, start_state=None, step_list=None):
    step_list = [] if step_list is None else step_list
    for episode in range(1, n_episodes + 1):
      trajs = self.generate_trajectory(start_state=start_state, det=False)
      G = 0
      W = 1
      for (i, (s, a, r)) in enumerate(trajs[::-1]):
        G = self.gamma * G + r
        self.visit_counts[(s, a)] += 1
        if self.weighted:
          self.C[(s, a)] += W
        self.Q[(s, a)] += self.Q_step(s, a, W, G)
        W *= self.target[(a, s)] / self.b[(a, s)]
        if W == 0:
          break
      if episode in step_list:
        self.estimates.append(self.target_estimate(start_state))

  def estimate_state(self, step_list, start_state=None, seed=0):
    """Returns a list of state estimates at steps `step_list` for MSE."""
    self.seed(seed)
    self.policy_evaluation(max(step_list), start_state=start_state,
                           step_list=step_list)
    estimates_arr = np.array(self.estimates)
    self.estimates = []
    return estimates_arr
