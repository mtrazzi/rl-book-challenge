import numpy as np

def sample(l): 
  return l[np.random.randint(len(l))]

def print_q_values(alg):
  env, Q = alg.env, alg.Q
  for s in env.states:
    print(str(s))
    for a in env.moves_d[s]:
      print(f"->{a}: {Q[(s, a)]}")

def to_arr(V):
  (min_x, max_x) = (min_y, max_y) = np.inf, -np.inf
  for pos in V.keys():
    x, y = pos.x, pos.y
    min_x, min_y = min(x, min_x), min(y, min_y)
    max_x, max_y = max(x, max_x), max(y, max_y)
  arr = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
  for pos, val in V.items():
    arr[pos.x, pos.y] = val
  return arr

