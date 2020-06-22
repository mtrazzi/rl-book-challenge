import argparse

from baird import BairdMDP
from baird_utils import b_baird, nab_vhat_baird, pi_baird, vhat_baird
import matplotlib.pyplot as plt
import numpy as np
from semi_grad_dp import SemiGradDP
from semi_grad_off_pol_td import SemiGradOffPolTD

plt.switch_backend('Qt5Agg')

BIG_FONT = 20
MED_FONT = 15

N_TIL = 4096
N_TLGS = 8

FIG_11_2_G = 0.99
FIG_11_2_ALP = 0.01
FIG_11_2_W_0 = [1, 1, 1, 1, 1, 1, 10, 1]
FIG_11_2_N_STEPS = 1000
FIG_11_2_BATCH = 10
FIG_11_2_N_RUNS_L = [10, 1]


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


def run_alg_on_baird(ax, alg, n_runs, title):
    n_batches = FIG_11_2_N_STEPS // FIG_11_2_BATCH
    batch_ticks = FIG_11_2_BATCH * (np.arange(n_batches) + 1)
    w_log = np.zeros((len(alg.env.states) + 1, n_batches))
    is_DP = isinstance(alg, SemiGradDP)
    w_0 = np.array(FIG_11_2_W_0)
    for seed in range(n_runs):
      if seed > 0 and seed % 10 == 0:
        print(f"[RUN #{seed}]")
      alg.w = w_0
      if not is_DP:
        alg.seed(seed)
      for n_iter in range(n_batches):
        alg.pol_eva(FIG_11_2_BATCH)
        w_log[:, n_iter] = w_log[:, n_iter] + alg.w
    for (j, w_j) in enumerate(w_log):
      ax.plot(batch_ticks, w_j / n_runs, label=f'w_{j + 1}')
    xticks, yticks = [0, 1000], [1, 10, 100, 200, 300]
    ax_title = f'{title}' + (f' ({n_runs} runs)' if not is_DP else '')
    plot_figure(ax, ax_title, xticks, xticks, 'Sweeps' if is_DP else 'Steps',
                yticks, yticks, '', labelpad=30)
    ax.legend()


def fig_11_2():
  fig = plt.figure()
  fig.set_size_inches(20, 14)
  fig.suptitle('Figure 11.2', fontsize=BIG_FONT)
  env = BairdMDP()
  b, pi = [{(a, s): f(a, s) for a in env.moves for s in env.states}
           for f in [b_baird, pi_baird]]
  baird_params = (len(FIG_11_2_W_0), FIG_11_2_ALP, FIG_11_2_G, vhat_baird,
                  nab_vhat_baird)
  alg1 = SemiGradOffPolTD(env, pi, b, *baird_params)
  alg2 = SemiGradDP(env, pi, b, *baird_params)
  for (i, alg) in enumerate([alg1, alg2]):
    run_alg_on_baird(fig.add_subplot(f'12{i+1}'), alg, FIG_11_2_N_RUNS_L[i],
                     'Semi-Gradient DP' if i else 'Semi-gradient Off-Policy TD')
  save_plot('fig11.2', dpi=100)
  plt.show()


def ex_11_3():
  pass


PLOT_FUNCTION = {
  '11.2': fig_11_2,
  'ex11.3': ex_11_3,
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
