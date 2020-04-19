import numpy as np

class DynamicProgrammingAfterstate:
  def __init__(self, env, pi=None, det_pi=None, theta=1e-4, gamma=0.9):
    self.theta = theta
    self.env = env
    self.V = {s: 0 for s in env.states}
    # vect for vectorized computation
    self.V_vect = np.array([self.V[s] for s in self.env.states]).astype(float)
    self.gamma = gamma
    self.pi = self.initialize_deterministic_pi(det_pi) if pi is None else pi
    self.compute_pi_vects()
    # expected reward of s, a
    self.er = {(s, a): np.dot(env.r, env.pr[(s, a)]) for s in env.states
               for a in env.moves_d[s]}
    print(theta)
    print(gamma)

  def initialize_deterministic_pi(self, det_pi_dict=None):
    """Initializes a deterministic policy pi."""
    if det_pi_dict is None or not det_pi_dict:
      det_pi_dict = {s: self.env.moves_d[s][np.random.randint(len(self.env.moves_d[s]))]
                     for s in self.env.states}
    return {(a, s): int(a == det_pi_dict[s]) for s in self.env.states
             for a in self.env.moves_d[s]}

  def compute_pi_vects(self):
    """Initializing vectors for pi(.|s) for faster policy evaluation."""
    self.pi_vect = {s: [self.pi[(a, s)] for a in self.env.moves_d[s]]
                    for s in self.env.states}

  def expected_value(self, s, a, arr):
    return self.er[(s, a)] + self.gamma * np.dot(arr, self.env.psp[(s, a)])

  def policy_evaluation(self):
    """Updates V according to current pi."""
    self.compute_pi_vects()  # for faster policy evaluation
    while True:
      delta = 0
      for s in self.env.states:
        v = self.V[s]
        expected_values = [self.expected_value(s, a, self.V_vect)
                           for a in self.env.moves_d[s]]
        bellman_right_side = np.dot(self.pi_vect[s], expected_values)
        self.V[s] = self.V_vect[self.env.states.index(s)] = bellman_right_side
        delta = max(delta, abs(v-self.V[s]))
      if delta < self.theta:
        break

  def deterministic_pi(self, s):
    return self.env.moves_d[s][np.argmax([self.pi[(a, s)] for a in self.env.moves_d[s]])]

  def update_pi(self, s, a):
    """Sets pi(a|s) = 1 and pi(a'|s) = 0 for a' != a."""
    for a_p in self.env.moves_d[s]:
      self.pi[(a_p, s)] = (a == a_p)

  def policy_improvement(self):
    """Improves pi according to current V. Returns True if policy is stable."""
    policy_stable = True
    for s in self.env.states:
      a_old = self.deterministic_pi(s)
      ev = np.array([self.expected_value(s, a, self.V_vect)
                    for a in self.env.moves_d[s]])
      a_new = self.env.moves_d[s][np.random.choice(np.flatnonzero(ev == ev.max()))]
      self.update_pi(s, a_new)
      policy_stable = policy_stable and (a_old == a_new)
    return policy_stable

  def policy_iteration(self):
    while True:
      self.policy_evaluation()
      if self.policy_improvement():
        return self.V, self.pi

