import numpy as np


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
