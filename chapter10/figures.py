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
  return tiles(iht, N_TILES, [N_TILES * x / (X_MAX - X_MIN),
               N_TILES * xdot / (V_MAX - V_MIN)], [a])


def fig_10_2():
    env = MountainCar()
    w_dim = TILE_SIZE * N_TILES
    iht = IHT(TILE_SIZE)
    def idxs(s, a): return get_idxs(iht, s[0], s[1], a)
    def qhat(s, a, w): return np.sum(w[idxs(s, a)])

    def nab_qhat(s, a, w):
      res = np.zeros(len(w))
      for idx in idxs(s, a):
        res[idx] = True
      return res

    for alp in FIG_10_2_ALP_L:
      print(alp)
      alg = EpisodicSemiGradientTD0(env, alp, w_dim, eps=0)
      n_steps = alg.pol_eva(qhat, nab_qhat, FIG_10_2_N_EP, FIG_10_2_G)
      plt.plot(n_steps, label=f'alpha={alp}')
    plt.legend()
    plt.show()


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


if __name__ == '__main__':
  main()
