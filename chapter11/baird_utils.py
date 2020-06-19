import numpy as np
from baird import SOLID

N_ST = 7
MAT = np.eye(N_ST + 1)
ONE_HOT = [MAT[i] for i in range(N_ST + 1)]
TWO_HOT = [2 * row for row in ONE_HOT]
MOST_ST = [2 * ONE_HOT[i] + ONE_HOT[N_ST] for i in range(N_ST + 1)]
LAST_ST = ONE_HOT[N_ST] + 2 * ONE_HOT[N_ST]


def feat_baird(s, w):
  return LAST_ST if s == N_ST else MOST_ST[s]


def vhat_baird(s, w):
  return 2 * w[s - 1] + w[N_ST] if s < N_ST else w[s - 1] + 2 * w[N_ST]


def nab_vhat_baird(s, w):
  return feat_baird(s, w)


def pi_baird(a, s):
  return a == SOLID


def b_baird(a, s):
  return (1 / 7) if a == SOLID else (6 / 7)
