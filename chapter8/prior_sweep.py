from dyna_q import DynaQ
from queue import PriorityQueue
import time

class Pair:
  def __init__(self, P, s, a):
    self.P = P
    self.s = s
    self.a = a

  def get_s_a(self):
    return (self.s, self.a)

  def __lt__(self, other):
    return self.P < other.P

class PrioritizedSweeping(DynaQ):
  def __init__(self, env, alpha, gamma, theta, policy=None, eps=0.1):
    super().__init__(env, alpha, gamma, eps)
    self.theta = theta
    self.policy = lambda s, Q: self.eps_gre(s) if policy is None else policy
    self.reset()

  def compute_p(self, s, a, r, s_p):
    Q_max = max(self.Q[(s_p, a_p)] for a_p in self.env.moves_d[s])
    return abs(r + self.g * Q_max - self.Q[(s, a)])

  def PQueue_update(self, s, a, r, s_p):
    P = self.compute_p(s, a, r, s_p)
    if P > self.theta:
      self.PQueue.put(Pair(P, s, a))

  def updates_until_optimal(self, n_steps_opt, max_plan_upd, tol=0.1):
    n_updates = 0
    counter = 0
    while True:
      counter += 1
      self.env.reset()
      n_steps = 0
      while True:
        s = self.env.state
        #print(self.env)
        #time.sleep(0.01)
        a = self.policy(s, self.Q)
        s_p, r, d, _ = self.env.step(a)
        n_steps += 1
        self.model.add_transition(s, a, r, s_p)
        self.predecessor[s_p].add((s, a))
        self.PQueue_update(s, a, r, s_p)
        for _ in range(max_plan_upd):
          if self.PQueue.empty():
            break
          s, a = self.PQueue.get().get_s_a()
          s_p, r = self.model.sample_s_r(s, a)
          self.q_learning_update(s, a, r, s_p)
          n_updates += 1
          for (_s, _a) in self.predecessor[s]:
            _, _r = self.model.sample_s_r(_s, _a)
            self.PQueue_update(_s, _a, _r, s)
        if d:
          if counter >= 3:
            time.sleep(1)
            print("n_steps=n_steps")
          break
      if self.test_n_steps(n_steps_opt, tol):
        return n_updates

  def test_n_steps(self, max_steps, tol=0.1): 
    s = self.env.reset()
    n_steps = 0
    d = False
    while n_steps < ((1 + tol) * max_steps) and not d:
      s, _, d, _ = self.env.step(self.gre(s))
      n_steps += 1
    print(f"{n_steps} < {((1 + tol) * max_steps)}")
    print(d)
    return d

  def reset(self):
    super().reset()
    self.PQueue = PriorityQueue()
    self.predecessor = {s: set() for s in self.env.states}
