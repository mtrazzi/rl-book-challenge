import numpy as np

ACCEPT, REJECT = 1, 0
QUE_SIZE = 4
P_FREE = 0.06
R_REJ = 0


class AccessControlQueuingTask:
  def __init__(self, n_serv=10):
    self.get_moves()
    self.n_serv = n_serv
    self.queue = np.random.randn(QUE_SIZE)
    self.reset()

  def get_priority(self):
    return 2 ** len(np.flatnonzero(self.queue < self.queue[0]))

  def get_state(self):
    return self.get_priority() * self.n_free_serv

  def get_moves(self):
    self.moves = [REJECT, ACCEPT]

  def update_queue(self):
    self.queue[0:3] = self.queue[1:]
    self.queue[-1] = np.random.randn()

  def update_servers(self):
    for i in range(self.n_serv - self.n_free_serv):
      self.n_free_serv += (np.random.random() < P_FREE)

  def step(self, a):
    self.update_servers()
    if a == REJECT or self.n_free_serv == 0:
      self.update_queue()
      state = self.get_state()
      return state, R_REJ, False, {}
    else:
      prio = self.get_priority()
      self.update_queue()
      self.n_free_serv -= 1
      state = self.get_state()
      return state, prio, False, {}

  def seed(self, seed):
    np.random.seed(seed)

  def reset(self):
    self.n_free_serv = self.n_serv
    self.state = self.get_state()
