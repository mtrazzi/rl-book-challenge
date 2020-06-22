import argparse

from baird import BairdMDP
from baird_utils import (b_baird, nab_vhat_baird, pi_baird, vhat_baird,
                         nab_qhat_baird, qhat_baird, feat_baird)
import matplotlib.pyplot as plt
import numpy as np
from semi_grad_dp import SemiGradDP
from semi_grad_qlearning import SemiGradQLearning
from semi_grad_off_pol_td import SemiGradOffPolTD
from tdc import ExpectedTDC, TDC

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

EX_11_3_W_0 = FIG_11_2_W_0 + FIG_11_2_W_0
EX_11_3_N_STEPS = FIG_11_2_N_STEPS * 10
EX_11_3_BATCH = FIG_11_2_BATCH * 10

FIG_11_5_ALP = 5e-3
FIG_11_5_BET = 5e-2
FIG_11_5_N_RUNS_L = [10, 1]


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


def run_alg_on_baird(ax, alg, n_runs, title, n_steps, batch_size, xticks,
                     yticks, w_init, log_ve_pbe=False):
    n_batches = n_steps // batch_size
    batch_ticks = batch_size * (np.arange(n_batches) + 1)
    w_log = np.zeros((len(w_init), n_batches))
    is_DP = isinstance(alg, SemiGradDP) or isinstance(alg, ExpectedTDC)
    w_0 = np.array(w_init)
    if log_ve_pbe:
      ve, pbe, vpi = np.zeros(n_batches), np.zeros(n_batches), lambda x: 0
    for seed in range(n_runs):
      if seed > 0 and seed % 10 == 0:
        print(f"[RUN #{seed}]")
      alg.w = w_0
      if not is_DP:
        alg.seed(seed)
      for n_iter in range(n_batches):
        alg.pol_eva(batch_size)
        w_log[:, n_iter] = w_log[:, n_iter] + alg.w
        if log_ve_pbe:
          ve[n_iter], pbe[n_iter] = alg.ve(vpi), alg.pbe()
    for (j, w_j) in enumerate(w_log):
      ax.plot(batch_ticks, w_j / n_runs, label=f'w_{j + 1}')
    if log_ve_pbe:
      plt.plot(batch_ticks, np.sqrt(ve) / n_runs, label='sqrt(VE)')
      plt.plot(batch_ticks, np.sqrt(pbe) / n_runs, label='sqrt(PBE)')
      plt.plot(batch_ticks, np.zeros_like(batch_ticks), '--')
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
                     'Semi-Gradient DP' if i else 'Semi-gradient Off-Policy TD',
                     FIG_11_2_N_STEPS, FIG_11_2_BATCH, [0, 1000],
                     [1, 10, 100, 200, 300], w_init=FIG_11_2_W_0)
  save_plot('fig11.2', dpi=100)
  plt.show()


def ex_11_3():
  fig = plt.figure()
  fig.set_size_inches(20, 14)
  fig.suptitle('Exercise 11.3', fontsize=BIG_FONT)
  env = BairdMDP()
  b, pi = [{(a, s): f(a, s) for a in env.moves for s in env.states}
           for f in [b_baird, pi_baird]]
  baird_params = (len(FIG_11_2_W_0) * 2, FIG_11_2_ALP, FIG_11_2_G, qhat_baird,
                  nab_qhat_baird)
  alg = SemiGradQLearning(env, pi, b, *baird_params)
  run_alg_on_baird(fig.add_subplot(f'111'), alg, FIG_11_2_N_RUNS_L[0],
                   'Semi-Gradient Q-learning', EX_11_3_N_STEPS, EX_11_3_BATCH,
                   [0, 1000], [1, 10, 100, 200, 300], w_init=EX_11_3_W_0)
  save_plot('ex11.3', dpi=100)
  plt.show()


def fig_11_5():
  fig = plt.figure()
  fig.set_size_inches(20, 14)
  fig.suptitle('Figure 11.2', fontsize=BIG_FONT)
  env = BairdMDP()
  b, pi = [{(a, s): f(a, s) for a in env.moves for s in env.states}
           for f in [b_baird, pi_baird]]
  args = (env, pi, b, len(FIG_11_2_W_0), FIG_11_5_ALP, FIG_11_5_BET,
          FIG_11_2_G, vhat_baird, feat_baird)
  for (i, alg) in enumerate([TDC(*args), ExpectedTDC(*args)]):
    run_alg_on_baird(fig.add_subplot(f'12{i+1}'), alg, FIG_11_5_N_RUNS_L[i],
                     'TDC' if i == 0 else 'Expected TDC',
                     FIG_11_2_N_STEPS, FIG_11_2_BATCH,
                     [0, 1000], [-2.5, 0, 2, 5, 10], w_init=FIG_11_2_W_0,
                     log_ve_pbe=True)
  save_plot('fig11.5', dpi=100)
  plt.show()


PLOT_FUNCTION = {
  '11.2': fig_11_2,
  'ex11.3': ex_11_3,
  '11.5': fig_11_5,
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
