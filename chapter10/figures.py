import argparse
import matplotlib.pyplot as plt
import numpy as np
from tiles_sutton import IHT, tiles
from mountain_car import MountainCar, X_MIN, X_MAX, V_MIN, V_MAX
from semi_grad_sarsa import EpisodicSemiGradientTD0

TILE_SIZE = 4096
N_TILES = 8

FIG_10_2_ALP_L = [alpha / 8 for alpha in [0.1, 0.2, 0.5]]
FIG_10_2_N_EP = 500
FIG_10_2_G = 1

def get_idxs(iht, x, xdot, a):
  return tiles(iht, N_TILES, [N_TILES * x / (X_MAX - X_MIN), N_TILES * xdot / (V_MAX - V_MIN)], [a])

def fig_10_2():
  env = MountainCar()
  w_dim = TILE_SIZE * N_TILES
  iht = IHT(TILE_SIZE)
  def qhat(s, a, w):
    return np.sum(w[get_idx(iht, s[0], s[1], a)
  #def nab_qhat(s, a, w):
  #  return np.arange(len(w)) == 
  for alp in FIG_10_2_ALP_L:
    alg = EpisodicSemiGradientTD0(env, alph, w_dim, eps=0)
    alg.pol_eva(

PLOT_FUNCTION = {
  '10.2': fig_10_2,
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=list(PLOT_FUNCTION.keys()) + ['all'])
  args = parser.parse_args()
  if args.figure == 'all':
    for key, f in PLOT_FUNCTION.items():
      print(f"[{key}]")
      f()
  else:
    print(f"[{args.figure}]")
    PLOT_FUNCTION[args.figure]()
