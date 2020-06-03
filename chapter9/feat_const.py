import numpy as np

def poly_feat(s, n): 
  if isinstance(s, float):
    return np.array([s ** i for i in range(n + 1)])
  # code below untested
  k = s.shape[0]
  n_feat = (n + 1) ** k
  x_s = np.ones(n_feat)
  pow_l = np.zeros(k)
  s_pow = [[s[i] ** j for j in range(n + 1)] for i in range(k)]
  idx = 0
  while idx < n_feat:
    for (i, cij) in enumerate(pow_l):
      x_s[idx] *= s_pow[i][int(cij)]
    for col in range(k - 1, -1, -1):
      pow_l[col] = (pow_l[col] + 1) % (n + 1)
      if pow_l[col] > 0:
        break
    idx += 1
  return x_s

def four_feat(s, n):
  if isinstance(s, float):
    return np.array([np.cos(i * np.pi * s) for i in range(n + 1)])
  # code below untested
  k = s.shape[0]
  n_feat = (n + 1) ** k
  x_s = np.ones(n_feat)
  pow_l = np.zeros(k)
  s_pow = [[s[i] ** j for j in range(n + 1)] for i in range(k)]
  idx = 0
  while idx < n_feat:
    for (i, cij) in enumerate(pow_l):
      x_s[idx] = np.cos(np.pi * np.dot(pow_l, s))
    for col in range(k - 1, -1, -1):
      pow_l[col] = (pow_l[col] + 1) % (n + 1)
      if pow_l[col] > 0:
        break
    idx += 1
  return x_s
