import argparse
import matplotlib.pyplot as plt
import numpy as np
from tiles_sutton import IHT, tiles
from mountain_car import MountainCar, X_MIN, X_MAX, V_MIN, V_MAX
from semi_grad_sarsa import EpisodicSemiGradientTD0

N_TILES = 4096
N_TILINGS = 8

FIG_10_2_ALP_L = [alpha / 8 for alpha in [0.1, 0.2, 0.5]]
FIG_10_2_N_EP = 100
FIG_10_2_G = 1
FIG_10_2_N_RUNS = 1


def get_idxs(iht, x, xdot, a):
  return tiles(iht, N_TILINGS, [N_TILINGS * x / (X_MAX - X_MIN),
               N_TILINGS * xdot / (V_MAX - V_MIN)], [a])


def print_qhat_mc(alg, ax, n_pts=50, act=None):
  qhat, w = alg.qhat, alg.w
  X = np.linspace(X_MIN, X_MAX, n_pts)
  V = np.linspace(V_MIN, V_MAX, n_pts)
  #ax.set_title(title, fontsize=10)
  if act is None:
    def cost(x, v): return -max([qhat([x, v], a, w) for a in alg.env.moves])
  else:
    def cost(x, v): return -qhat([x, v], act, w)
  qvals = np.array([[cost(x, v) for x in X] for v in V])
  (X, Y), Z = np.meshgrid(X, V), qvals
  ax.set_xlabel('Position', fontsize=8)
  ax.set_ylabel('Velocity', fontsize=8)
  ax.set_xticks([X.min(), X.max()])
  ax.set_yticks([V.min(), V.max()])
  ax.set_zticks([qvals.min(), qvals.max()])
  COLOR_DICT = {-1: 'r', 0: 'k', 1: 'b'}
  ax.plot_surface(X, Y, Z, color=COLOR_DICT[act] if act is not None else None, antialiased=False)

def fig_10_2():
  env = MountainCar(mnt=False)
  w_dim = N_TILES * N_TILINGS
  iht = IHT(N_TILES)
  def idxs(s, a): return get_idxs(iht, s[0], s[1], a)
  def qhat(s, a, w): return np.sum(w[idxs(s, a)])

  def nab_qhat(s, a, w):
    res = np.zeros(len(w))
    for idx in idxs(s, a):
      res[idx] = True
    return res

  def smooth_steps(arr, to_avg=5):
    nb_rew = len(arr)
    new_arr = np.zeros(nb_rew - to_avg + 1)
    for i in range(nb_rew - to_avg + 1):
      new_arr[i] = np.mean([arr[i + j] for j in range(to_avg)])
    return new_arr

  for alp in FIG_10_2_ALP_L:
    tot_n_steps = np.zeros(FIG_10_2_N_EP)
    for _ in range(FIG_10_2_N_RUNS):
      print(alp)
      alg = EpisodicSemiGradientTD0(env, alp, w_dim, eps=0)
      alg.seed(0)
      for _ in range(10):
        tot_n_steps += np.array(alg.pol_eva(qhat, nab_qhat, FIG_10_2_N_EP,
                                            FIG_10_2_G, max_steps=100))
        fig = plt.figure()
        ax = fig.add_subplot('111', projection='3d')
        for a in alg.env.moves:
          print_qhat_mc(alg, ax, act=a)
        fig.set_size_inches(20, 14)
        plt.show()

    # plt.plot(smooth_steps(tot_n_steps, to_avg=10), label=f'alpha={alp}')
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
