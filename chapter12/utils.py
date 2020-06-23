import numpy as np


def enc_st_agg(s, w, tot_st):
  return int(s // (tot_st // len(w)))


def vhat_st_agg(s, w, tot_st):
  if s == tot_st:
    return 0
  return w[enc_st_agg(s, w, tot_st)]


def nab_vhat_st_agg(s, w, tot_st):
  if s == tot_st:
    return 0
  return np.array([i == enc_st_agg(s, w, tot_st) for i in range(len(w))])
