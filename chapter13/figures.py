import argparse
import numpy as np
import matplotlib.pyplot as plt
from corridor import Corridor, L, R
from reinforce import Reinforce

plt.switch_backend('Qt5Agg')

BIG_FONT = 20
MED_FONT = 15
SMA_FONT = 13

FIG_13_1_ALP_L = [2 ** (-k) for k in range(12, 15)]
FIG_13_1_N_EP = 1000
FIG_13_1_N_RUNS = 10
FIG_13_1_G = 1
R_FT, L_FT = np.array([1, 0]), np.array([0, 1])
FIG_13_1_THE_DIM = R_FT.shape[0]


def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)


def plot_figure(ax, title, xticks, xnames, xlabel, yticks, ynames, ylabel,
                labelpad=15, font=SMA_FONT, loc='upper left'):
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


def run(ax, alg, alp_l, n_ep, n_runs):
  for alp in alp_l:
    alg.a = alp
    print(f"[ALPHA={alp}]")
    tot_rew = np.zeros(n_ep)
    for seed in range(n_runs):
      print(f"[RUN #{seed}]")
      alg.reset()
      alg.seed(seed)
      tot_rew += np.array(alg.train(n_ep))
    plt.plot(tot_rew / n_runs, label=f'alpha={alp}')
  plt.plot(np.zeros(n_ep), '--', label='v*(s_0)')


def benchmark(alg, title, fn):
  fig, ax = plt.subplots()
  fig.suptitle(title, fontsize=BIG_FONT)
  fig.set_size_inches(20, 14)

  xticks, yticks = np.linspace(0, 1000, 6), np.linspace(-90, -10, 9)
  def short_str(x): return str(x)[:3]
  xnames, ynames = map(short_str, xticks), map(short_str, yticks)
  run(ax, alg, FIG_13_1_ALP_L, FIG_13_1_N_EP, FIG_13_1_N_RUNS)
  plot_figure(ax, '', xticks, xnames, 'Episode', yticks, ynames,
              (f'Total\nReward\non episode\n(Averaged over\n' +
               f'{FIG_13_1_N_RUNS} runs)'), font=MED_FONT, labelpad=40,
              loc='upper right')
  save_plot(fn, dpi=100)
  plt.show()


def logpi_wrap_corr(env, feat):
  def logpi(a, s, pi):
    ft_as = feat(s, a)
    vec_sum = np.zeros_like(ft_as, dtype='float64')
    for b in env.moves:
      vec_sum += pi[(b, s)] * feat(s, b)
    return ft_as - vec_sum
  return logpi


def feat_corr(s, a):
  return L_FT if a == L else R_FT


def pi_gen_corr(env, the):
  return {(a, s): the[0] if a == R else (1 - the[0])
          for a in env.moves for s in env.states}


def fig_13_1():
  env = Corridor()

  alg = Reinforce(env, None, FIG_13_1_G, FIG_13_1_THE_DIM, pi_gen_corr,
                  logpi_wrap_corr(env, feat_corr), the_0=np.array([0.5, 0.5]))
  benchmark(alg, 'Figure 13.1', 'fig13.1')


PLOT_FUNCTION = {
  '13.1': fig_13_1,
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
