import numpy as np
from corridor import L

R_FT, L_FT = np.array([1, 0]), np.array([0, 1])


def sample_action(env, pi, s):
  moves = env.moves
  pi_dist = [pi[(a, s)] for a in moves]
  return env.moves[np.random.choice(np.arange(len(moves)), p=pi_dist)]


def gen_traj(env, pi, s_0=None):
  if s_0 is None:
    s = env.reset()
  else:
    s = s_0
    env.force_state(s_0)
  traj, d = [], False
  while not d:
    a = sample_action(env, pi, s)
    s_p, r, d, _ = env.step(sample_action(env, pi, s))
    traj.append((s, a, r))
    s = s_p
  return traj


def h_linear(feat, s, a, the):
  return np.dot(feat(s, a), the)


def feat_corr(s, a):
  return L_FT if a == L else R_FT


def softmax(feat, h, s, a, env, the):
  return np.exp(h(feat, s, a, the)) / np.sum(np.exp(h(feat, s, b, the))
                                             for b in env.moves)


def pi_gen_corr(env, the):
  return {(a, s): softmax(feat_corr, h_linear, s, a, env, the)
          for a in env.moves for s in env.states}


def logpi_wrap_corr(env, feat):
  def logpi(a, s, pi):
    ft_as = feat(s, a)
    vec_sum = np.zeros_like(ft_as, dtype='float64')
    for b in env.moves:
      vec_sum += pi[(b, s)] * feat(s, b)
    return ft_as - vec_sum
  return logpi
