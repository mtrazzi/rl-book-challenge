import argparse
import numpy as np
import matplotlib.pyplot as plt
from randomwalk import RandomWalk
from utils import vhat_st_agg, nab_vhat_st_agg
from off_lam_ret import OffLamRet

plt.switch_backend('Qt5Agg')

BIG_FONT = 20
MED_FONT = 15

FIG_12_3_LAM_L = [0, .4, .8, .9, .95, .975, .99, 1][:1]
FIG_12_3_N_EP = 10
FIG_12_3_N_ST = 19
FIG_12_3_N_RUNS = 1
FIG_12_3_G = 1


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


def run_random_walks(ax, alg, lam_l, n_ep, n_runs):
  pi = {(a, s): 1.0 for s in alg.env.states for a in alg.env.moves_d[s]}
  true_vals = np.linspace(-1, 1, alg.env.n_states + 2)[1:-1]
  for (k, lam) in enumerate(lam_l):
    alg.lam = lam
    print(f"[LAMBDA={lam}]")
    err_l = []
    alpha_max = 1 if (lam <= 0.95) else 1 / (2 * (k - 3))
    alpha_l = np.linspace(0, alpha_max, 5)
    for alpha in alpha_l:
      alg.a = alpha
      print(f"[ALPHA={alpha}]")
      err_sum = 0
      for seed in range(n_runs):
        alg.reset()
        alg.seed(seed)
        for ep in range(n_ep):
          alg.pol_eva(pi, n_ep=1)
          v_arr = np.array(alg.get_value_list()[:-1])
          err_sum += np.sqrt(np.sum((v_arr-true_vals) ** 2) / alg.env.n_states)
      err_l.append(err_sum / (n_runs * n_ep))
    plt.plot(alpha_l, err_l, label=f'lam={lam}')


def fig_12_3():
  fig, ax = plt.subplots()
  fig.set_size_inches(20, 14)
  def vhat(s, w): return vhat_st_agg(s, w, FIG_12_3_N_ST)
  def nab_vhat(s, w): return nab_vhat_st_agg(s, w, FIG_12_3_N_ST)
  alg = OffLamRet(RandomWalk(), None, FIG_12_3_N_ST, None, vhat, nab_vhat,
                  FIG_12_3_G)
  run_random_walks(ax, alg, FIG_12_3_LAM_L, FIG_12_3_N_EP, FIG_12_3_N_RUNS)
  # xticks, yticks = np.linspace(0, 1, 6), np.linspace(0.25, 0.55, 6)
  # plot_figure(ax, 'Figure 12.3', xticks, xticks, 'alpha', yticks, yticks,
  #             (f'Average\nRMS error\n({FIG_12_3_N_ST} states\nFirst' +
  #              f'{FIG_12_3_N_EP}episodes'))
  save_plot('fig12.3', dpi=100)
  plt.show()


PLOT_FUNCTION = {
  '12.3': fig_12_3,
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


if __name__ == "__main__":
  main()