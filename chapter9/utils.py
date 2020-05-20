import numpy as np

R_TERM = 0

def sample_action(env, pi, s):
  moves = env.moves_d[s]
  pi_dist = [pi[(a, s)] for a in moves]
  return env.moves_d[s][np.random.choice(np.arange(len(moves)), p=pi_dist)]

def gen_traj(env, pi, s_0=None, inc_term=False): 
  if s_0 is None:
    s = env.reset()
  else:
    s = s_0
    env.force_state(s_0)
  traj, d = [], False
  while not d:
    s_p, r, d, _ = env.step(sample_action(env, pi, s))
    traj.append((s, r))
    s = s_p
  if inc_term:
    traj.append((s, R_TERM))
  return traj

def gen_traj_ret(env, pi, gamma, s_0=None):
  traj = gen_traj(env, pi, s_0)
  ret_traj, G = [], 0
  for (t, (s, r)) in enumerate(traj[::-1]):
    G = r + gamma * G
    ret_traj.append((s, G))
  return ret_traj[::-1]

def est(env, pi, s_0, gamma, n_ep):
  v = 0
  print(s_0)
  for n in range(1, n_ep + 1):
    _, G = gen_traj_ret(env, pi, gamma, s_0)[0]
    v += (G - v) / n
  return v
