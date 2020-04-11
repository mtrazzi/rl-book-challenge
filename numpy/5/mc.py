import numpy as np
import time

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

  def sample_action(self, s, det=True, eps=None):
    if eps is not None and np.random.random() < eps:
      return self.env.moves[np.random.randint(len(self.env.moves))]
    elif not det:
      pi_dist = []
      for a in self.env.moves:
        pi_dist.append(self.pi[(a,s)])
      pi_dist = [self.pi[(a, s)] for a in self.env.moves]
      return self.env.moves[np.random.choice(np.arange(len(self.env.moves)), p=pi_dist)]
    else:
      return self.det_pi[s]

  def generate_trajectory(self, start_state=None, det=True, max_steps=np.inf, eps=None): 
    trajs = []
    s = self.env.reset() if start_state is None else start_state
    if start_state is not None:
      self.env.force_state(start_state)
    n_steps = 0
    while True and n_steps < max_steps:
      a = self.sample_action(s, det, eps)
      s_p, r, done, _ = self.env.step(a)
      n_steps += 1
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
    self.b = pi if b is None else b
    self.pi = b  # because self.pi used in generate_trajectory
    self.target = pi
    self.weighted = weighted # weighted or ordinary important sampling
    self.reset()

  def target_estimate(self, s):
    return sum(self.target[(a, s)] * self.Q[(s, a)] for a in self.env.moves)

  def reset(self):
    super().reset()
    self.C = {(s, a): 0 for s in self.env.states for a in self.env.moves}
    self.visit_counts = {key: 0 for key in self.C.keys()}


class OffPolicyMCPrediction(OffPolicyMC):
  def __init__(self, env, pi, weighted=True, b=None, gamma=1):
    super().__init__(env, pi, weighted, b, gamma)
    self.reset()

  def reset(self):
    super().reset()
    self.estimates = []
    # returns scaled by importance sampling ratio
    self.is_returns = {(s, a): [] for s in self.env.states for a in self.env.moves}

  def ordinary_is(self, n_episodes, start_state=None, step_list=None):
    """Ordinary Importance Sampling when start_state happens once per episode."""
    step_list = [] if step_list is None else step_list
    q_steps = []
    for episode in range(n_episodes + 1):
      trajs = self.generate_trajectory(start_state=start_state, det=False)
      G = 0
      W = 1
      for (i, (s, a, r)) in enumerate(trajs[::-1]):
        G = self.gamma * G + r
        self.is_returns[(s, a)].append(W * G)
        W *= self.target[(a, s)] / self.b[(a, s)]
        if W == 0:
          break
      if episode in step_list:
        for a in self.env.moves:
          self.Q[(start_state, a)] = np.sum(self.is_returns[(s, a)]) / episode
        self.estimates.append(self.target_estimate(start_state))

  def weighted_is(self, n_episodes, start_state=None, step_list=None):
    """Weighted Importance Sampling when start_state happens once per episode."""
    step_list = [] if step_list is None else step_list
    q_steps = []
    for episode in range(n_episodes + 1):
      trajs = self.generate_trajectory(start_state=start_state, det=False)
      G = 0
      W = 1
      for (i, (s, a, r)) in enumerate(trajs[::-1]):
        G = self.gamma * G + r
        self.C[(s, a)] += W
        self.Q[(s, a)] += (W / self.C[(s, a)]) * (G - self.Q[(s, a)])
        W *= self.target[(a, s)] / self.b[(a, s)]
        if W == 0:
          break
      if episode in step_list:
        self.estimates.append(self.target_estimate(start_state))
  
  def importance_sampling(self, n_episodes, start_state=None, step_list=None):
    algorithm = self.weighted_is if self.weighted else self.ordinary_is
    algorithm(n_episodes, start_state, step_list)

  def estimate_state(self, step_list, start_state=None, seed=0):
    """Returns a list of state estimates at steps `step_list` for MSE."""
    self.seed(seed)
    self.importance_sampling(max(step_list), start_state=start_state,
                             step_list=step_list)
    estimates_arr = np.array(self.estimates)
    self.estimates = []
    return estimates_arr

class OffPolicyMCControl(OffPolicyMC):
  def __init__(self, env, pi, b, gamma=1):
    super().__init__(env, pi, True, b, gamma)
    self.reset()
    self.init_det_pi()


  def init_det_pi(self):
    self.det_target = {}
    for s in self.env.states:
      for a in self.env.moves:
        # initializing Q to allow some randomness in deterministic policy
        self.Q[(s,a)] = -int(1e5)
      self.update_det_target(s)

  def update_det_target(self, s):
    best_move = self.env.moves[np.argmax([self.Q[(s, a)] for a in self.env.moves])]
    self.det_target[s] = best_move

  def det_target_estimate(self, s):
    return self.Q[(s,self.det_target[s])]
 
  def optimal_policy(self, n_episodes, start_state=None, step_list=None):
    step_list = [] if step_list is None else step_list
    for episode in range(1, n_episodes + 1):
      start = time.time() 
      trajs = self.generate_trajectory(start_state=start_state, det=False)
      print(f"generating trajectory took: {time.time() - start}s")
      if episode > 0 and episode % 10 == 0:
        print(f"episode #{episode}")
      G = 0
      W = 1
      for (i, (s, a, r)) in enumerate(trajs[::-1]):
        G = self.gamma * G + r
        self.C[(s, a)] += W
        self.Q[(s, a)] += (W / self.C[(s, a)]) * (G - self.Q[(s, a)])
        self.update_det_target(s)
        if not np.all(a == self.det_target[s]):
          break
        W *= 1 / self.b[(a, s)]
      if episode in step_list:
        self.estimates.append(self.det_target_estimate(start_state))
  
  def truncated_weighted_avg_est(self, n_episodes, start_state=None, step_list=None):
    step_list = [] if step_list is None else step_list
    for episode in range(1, n_episodes + 1):
      start = time.time() 
      trajs = self.generate_trajectory(start_state=start_state, det=False)
      print(f"generating trajectory took: {time.time() - start}s")
      if episode > 0 and episode % 10 == 0:
        print(f"episode #{episode}")
      G = 0
      W = 1
      gamma_fact = 1
      for (i, (s, a, r)) in enumerate(trajs[::-1]):
        G = G + r
        if i == 1 and self.gamma != 1:
          gamma_fact *= (1 - self.gamma) / self.gamma
        else:
          gamma_fact *= self.gamma
        actual_w = W * gamma_fact
        self.C[(s, a)] += actual_w
        self.Q[(s, a)] += (actual_w  / self.C[(s, a)]) * (G - self.Q[(s, a)])
        self.update_det_target(s)
        W *= 1 / self.b[(a, s)]
        if not np.all(a == self.det_target[s]):
          break
      if episode in step_list:
        self.estimates.append(self.det_target_estimate(start_state))
  
  def reset(self):
    super().reset()
    self.estimates = []

  def __str__(self):
    res = ''
    for s in self.env.states:
      res += f"{str(s)}: {self.det_target[s]}\n"
    return res
