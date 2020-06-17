import argparse
import matplotlib.pyplot as plt
import numpy as np
from tiles_sutton import IHT, tiles
from mountain_car import MountainCar, X_MIN, X_MAX, V_MIN, V_MAX
from semi_grad_sarsa import EpisodicSemiGradientTD0
from nstep_semi_grad_sarsa import nStepSemiGradSarsa

BIG_FONT = 20
MED_FONT = 15

N_TIL = 4096
N_TLGS = 8

FIG_10_1_G = 1
FIG_10_1_STEPS = 428
FIG_10_1_EP_L = [12, 104, 1000, 9000]
FIG_10_1_ALP = 0.5 / 8

FIG_10_2_ALP_L = [alpha / 8 for alpha in [0.1, 0.2, 0.5]]
FIG_10_2_N_EP = 500
FIG_10_2_G = FIG_10_1_G
FIG_10_2_N_RUNS = 100

FIG_10_3_ALP_L = [alpha / 8 for alpha in [0.5, 0.3]]
FIG_10_3_N_L = [1, 8]
FIG_10_3_N_EP = 500
FIG_10_3_G = FIG_10_1_G
FIG_10_3_N_RUNS = 1

FIG_10_4_MAX_N = 4
FIG_10_4_N_EP = 50
FIG_10_4_G = FIG_10_2_G
FIG_10_4_N_RUNS = 1


def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)


def plot_figure(ax, title, xticks, xnames, xlabel, yticks, ynames, ylabel,
                labelpad=15, font=MED_FONT, loc='upper left'):
  ax.set_title(title, fontsize=font)
  ax.set_xticks(xticks)
  ax.set_xticklabels(xnames)
  ax.set_yticks(yticks)
  ax.set_yticklabels(ynames)
  ax.set_xlim([min(xticks), max(xticks)])
  ax.set_ylim([min(yticks), max(yticks)])
  ax.set_xlabel(xlabel, fontsize=font)
  ax.set_ylabel(ylabel, rotation=0, fontsize=font, labelpad=labelpad)
  plt.legend(loc=loc)


def get_idxs(iht, x, xdot, a):
  return tiles(iht, N_TLGS, [N_TLGS * x / (X_MAX - X_MIN),
               N_TLGS * xdot / (V_MAX - V_MIN)], [a])


def print_qhat_mc(alg, fig, fig_id, title, n_pts=50, act=None):
  ax = fig.add_subplot(fig_id, projection='3d')
  ax.set_title(title, fontsize=BIG_FONT)
  qhat, w = alg.qhat, alg.w
  X = np.linspace(X_MIN, X_MAX, n_pts)
  V = np.linspace(V_MIN, V_MAX, n_pts)
  if act is None:
    def cost(x, v): return -max([qhat([x, v], a, w) for a in alg.env.moves])
  else:
    def cost(x, v): return -qhat([x, v], act, w)
  qvals = np.array([[cost(x, v) for x in X] for v in V])
  (X, Y), Z = np.meshgrid(X, V), qvals
  ax.set_xlabel('Position', fontsize=15)
  ax.set_ylabel('Velocity', fontsize=15)
  ax.set_zlabel('Cost', fontsize=15)
  ax.set_xticks([X.min(), X.max()])
  ax.set_yticks([V.min(), V.max()])
  ax.set_zticks([qvals.min(), qvals.max()])
  COLOR_DICT = {-1: 'r', 0: 'g', 1: 'b'}
  ax.plot_surface(X, Y, Z, color=COLOR_DICT[act] if act is not None else None)


def get_qhats(n_til, n_tlgs):
  iht = IHT(N_TIL)
  def idxs(s, a): return get_idxs(iht, s[0], s[1], a)
  def qhat(s, a, w): return np.sum(w[idxs(s, a)])

  def nab_qhat(s, a, w):
    res = np.zeros(len(w))
    for idx in idxs(s, a):
      res[idx] = True
    return res
  return qhat, nab_qhat


def fig_10_1():
  def plot_and_save(filename, title, alg, n_ep, max_steps=np.inf):
    fig = plt.figure()
    alg.pol_eva(qhat, nab_qhat, n_ep, FIG_10_1_G, max_steps=max_steps)
    print_qhat_mc(alg, fig, '111', title)
    fig.set_size_inches(20, 14)
    save_plot(filename, dpi=100)
    plt.show()

  qhat, nab_qhat = get_qhats(N_TIL, N_TLGS)
  env = MountainCar()
  alg = EpisodicSemiGradientTD0(env, FIG_10_1_ALP, N_TIL * N_TLGS, eps=0)
  alg.seed(0)
  plot_and_save(f'fig10.1_{FIG_10_1_STEPS}_steps', f'Step {FIG_10_1_STEPS}',
                alg, 1, FIG_10_1_STEPS)

  tot_ep = 1
  for ep in FIG_10_1_EP_L:
    alg.pol_eva(qhat, nab_qhat, ep - tot_ep, FIG_10_2_G)
    plot_and_save(f'fig10.1_{ep}_episodes', f'Episode {ep}', alg, ep - tot_ep)
    tot_ep += (ep - tot_ep)


def fig_10_2():
  fig, ax = plt.subplots()
  qhat, nab_qhat = get_qhats(N_TIL, N_TLGS)

  for alp in FIG_10_2_ALP_L:
    tot_n_steps = np.zeros(FIG_10_2_N_EP)
    for seed in range(FIG_10_2_N_RUNS):
      print(alp)
      alg = EpisodicSemiGradientTD0(MountainCar(), alp, N_TIL * N_TLGS, eps=0)
      alg.seed(seed)
      tot_n_steps += np.array(alg.pol_eva(qhat, nab_qhat, FIG_10_2_N_EP,
                                          FIG_10_2_G))
    plt.plot(tot_n_steps, label=f'alpha={alp}')
  plt.yscale('log')
  xticks, yticks = [0, 500], [100, 200, 4000, 1000]
  plot_figure(ax, 'Figure 10.2', xticks, xticks, 'Episode', yticks, yticks,
              'Steps\nper episode\n(log scale)')
  fig.set_size_inches(20, 14)
  plt.legend()
  save_plot('fig10.2', dpi=100)
  plt.show()


def fig_10_3():
  fig, ax = plt.subplots()
  qhat, nab_qhat = get_qhats(N_TIL, N_TLGS)

  for (n, alp) in zip(FIG_10_3_N_L, FIG_10_3_ALP_L):
    tot_n_steps = np.zeros(FIG_10_3_N_EP)
    for seed in range(FIG_10_3_N_RUNS):
      print(alp)
      alg = nStepSemiGradSarsa(MountainCar(), alp, N_TIL * N_TLGS, 0, n)
      alg.seed(seed)
      tot_n_steps += np.array(alg.pol_eva(None, qhat, nab_qhat, FIG_10_3_N_EP,
                                          FIG_10_3_G, max_steps=1000))
    plt.plot(tot_n_steps, label=f'n={n}')
  plt.yscale('log')
  xticks, yticks = [0, 500], [100, 200, 4000, 1000]
  plot_figure(ax, 'Figure 10.3', xticks, xticks, 'Episode', yticks, yticks,
              'Steps\nper episode\n(log scale)')
  fig.set_size_inches(20, 14)
  plt.legend()
  save_plot('fig10.3', dpi=100)
  plt.show()


def fig_10_4():
  fig, ax = plt.subplots()
  qhat, nab_qhat = get_qhats(N_TIL, N_TLGS)

  n_l = [2 ** k for k in range(int(np.log(FIG_10_4_MAX_N) / np.log(2)) + 1)]
  alg = nStepSemiGradSarsa(MountainCar(), 0, N_TIL * N_TLGS, 0, 0)
  for n in n_l:
    alg.n = n
    print(f">> n={n}")
    steps_l = []
    alpha_max = 1 if n <= 16 else 1 / (np.log(n // 8) / np.log(2))
    alpha_l = np.linspace(0, alpha_max, 31)
    for alpha in alpha_l:
      alg.a = alpha
      print(f"alpha={alpha}")
      tot_steps = 0
      for seed in range(FIG_10_4_N_RUNS):
        alg.reset()
        alg.seed(seed)
        for ep in range(FIG_10_4_N_EP):
          tot_steps += alg.pol_eva(None, qhat, nab_qhat, 1, FIG_10_4_G)
      steps_l.append(tot_steps / (FIG_10_4_N_RUNS * FIG_10_4_N_EP))
    plt.plot(alpha_l, steps_l, label=f'n={n}')


PLOT_FUNCTION = {
  '10.1': fig_10_1,
  '10.2': fig_10_2,
  '10.3': fig_10_3,
  '10.4': fig_10_4,
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
