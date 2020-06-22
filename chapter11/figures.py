import argparse
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
from baird import BairdMDP
from baird_utils import vhat_baird, nab_vhat_baird, pi_baird, b_baird
from semi_grad_off_pol_td import SemiGradOffPolTD
import numpy as np

BIG_FONT = 20
MED_FONT = 15

N_TIL = 4096
N_TLGS = 8

FIG_11_2_G = 0.99
FIG_11_2_ALP = 0.01
FIG_11_2_W_0 = [1, 1, 1, 1, 1, 1, 10, 1]
FIG_11_2_N_STEPS = 1000
FIG_11_2_BATCH = 10
FIG_11_2_N_RUNS = 10


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


def fig_11_2():
  fig = plt.figure()
  fig.set_size_inches(20, 28)
  fig.suptitle('Figure 11.2')
  ax1, ax2 = fig.add_subplot('121'), fig.add_subplot('122')
  env = BairdMDP()
  b, pi = [{(a, s): f(a, s) for a in env.moves for s in env.states}
           for f in [b_baird, pi_baird]]
  w_0 = np.array(FIG_11_2_W_0)
  alg1 = SemiGradOffPolTD(env, pi, b, w_0.shape[0], FIG_11_2_ALP, FIG_11_2_G,
                          vhat_baird, nab_vhat_baird)
  n_batches = FIG_11_2_N_STEPS // FIG_11_2_BATCH
  batch_ticks = FIG_11_2_BATCH * (np.arange(n_batches) + 1)
  w_log = np.zeros((len(env.states) + 1, n_batches))
  for seed in range(FIG_11_2_N_RUNS):
    if seed > 0 and seed % 10 == 0:
      print(f"[RUN #{seed}]")
    alg1.w = w_0
    alg1.seed(seed)
    for n_iter in range(n_batches):
      alg1.pol_eva(FIG_11_2_BATCH)
      w_log[:, n_iter] = w_log[:, n_iter] + alg1.w
  for (i, w_i) in enumerate(w_log):
    ax1.plot(batch_ticks, w_i / FIG_11_2_N_RUNS, label=f'w_{i + 1}')
  xticks, yticks = [0, 1000], [1, 10, 100, 200, 300]
  plot_figure(ax1, f'Semi-gradient Off-Policy TD ({FIG_11_2_N_RUNS} runs)',
              xticks, xticks, 'Steps', yticks, yticks, '', labelpad=30)
  ax1.legend()
  save_plot('fig11.2', dpi=100)
  plt.show()


PLOT_FUNCTION = {
  '11.2': fig_11_2,
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
