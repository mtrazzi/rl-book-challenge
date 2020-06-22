from baird import SOLID
import numpy as np

N_ST = 7
MAT = np.eye(N_ST + 1)
ONE_HOT = [MAT[i] for i in range(N_ST + 1)]
TWO_HOT = [2 * row for row in ONE_HOT]
MOST_ST = [2 * ONE_HOT[i] + ONE_HOT[N_ST] for i in range(N_ST + 1)]
LAST_ST = ONE_HOT[N_ST - 1] + 2 * ONE_HOT[N_ST]
ZER = np.zeros(N_ST + 1)


def feat_baird(s):
  return LAST_ST if s == N_ST else MOST_ST[s - 1]


def vhat_baird(s, w):
  return np.dot(w, feat_baird(s))


def nab_vhat_baird(s, w):
  return feat_baird(s)


def qhat_baird(s, a, w):
  return np.dot(w, nab_qhat_baird(s, a, w))


def nab_qhat_baird(s, a, w):
  to_conc = [nab_vhat_baird(s, w), ZER]
  return np.concatenate(to_conc if a == 0 else to_conc[::-1])


def pi_baird(a, s):
  return a == SOLID


def b_baird(a, s):
  return (1 / 7) if a == SOLID else (6 / 7)
