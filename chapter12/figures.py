import argparse
import numpy as np
import matplotlib.pyplot as plt
from randomwalk import RandomWalk
from utils import vhat_st_agg, nab_vhat_st_agg
from off_lam_ret import OffLamRet
# from on_lam_ret import OnLamRet
from semi_grad_td_lam import SemiGradTDLam
from true_online_td import TrueOnlineTD
from mountain_car import MountainCar, X_MAX, X_MIN, V_MAX, V_MIN
from tiles_sutton import IHT, tiles
from sarsa_lam import SarsaLam, SarsaLamAcc, SarsaLamClr
from true_online_sarsa import TrueOnlineSarsa

#plt.switch_backend('Qt5Agg')

BIG_FONT = 20
MED_FONT = 15
SMA_FONT = 13

FIG_12_3_LAM_L = [0, .4, .8, .9, .95, .975, .99, 1]
FIG_12_3_N_EP = 10
FIG_12_3_N_ST = 19
FIG_12_3_N_RUNS = 1
FIG_12_3_G = 1

FIG_12_6_LAM_L = FIG_12_3_LAM_L
FIG_12_6_N_EP = FIG_12_3_N_EP
FIG_12_6_N_ST = FIG_12_3_N_ST
FIG_12_6_N_RUNS = FIG_12_3_N_RUNS
FIG_12_6_G = FIG_12_3_G

N_TIL = 4096
N_TLGS = 8

FIG_12_10_G = 1
FIG_12_10_EPS = 0
FIG_12_10_LAM_L = [0, .68, .84, .92, .96, .98, .99]
FIG_12_10_ALP_MIN, FIG_12_10_ALP_MAX = 0.4, 1.5
FIG_12_10_N_PTS = 10
FIG_12_10_N_RUNS = 20
FIG_12_10_N_EP = 50
FIG_12_10_MAX_STEPS = 1000

FIG_12_11_G = FIG_12_10_G
FIG_12_11_EPS = FIG_12_10_EPS
FIG_12_11_N_PTS = FIG_12_10_N_PTS
FIG_12_11_N_RUNS = 1#FIG_12_10_N_RUNS
FIG_12_11_N_EP = 10
FIG_12_11_MAX_STEPS = 5000
FIG_12_11_LAM = 0
FIG_12_11_ALP_BND = {
  SarsaLamClr: [.2, 2],
  SarsaLam: [.2, 2],
  TrueOnlineSarsa: [.2, 2],
  SarsaLamAcc: [.2, .5],
}
FIG_12_11_ALG_STR = {
  SarsaLamClr: "Sarsa(Lambda) w/ replacing/clearing traces",
  SarsaLam: "Sarsa(Lambda) w/ replacing traces",
  TrueOnlineSarsa: "True Online Sarsa(Lambda)",
  SarsaLamAcc: "Sarsa(Lambda) w/ accumulating traces",
}


def get_idxs(iht, x, xdot, a):
  return tiles(iht, N_TLGS, [N_TLGS * x / (X_MAX - X_MIN),
               N_TLGS * xdot / (V_MAX - V_MIN)], [a])


def get_fn_mc(n_til, n_tlgs):
  iht = IHT(N_TIL)
  def idxs(s, a): return get_idxs(iht, s[0], s[1], a)
  def qhat(s, a, w): return np.sum(w[idxs(s, a)])
  return idxs, qhat


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


def run_random_walks(ax, alg, lam_l, n_ep, n_runs, sub=3):
  pi = {(a, s): 1.0 for s in alg.env.states for a in alg.env.moves_d[s]}
  true_vals = np.linspace(-1, 1, alg.env.n_states + 2)[1:-1]
  for (k, lam) in enumerate(lam_l):
    alg.lam = lam
    print(f"[LAMBDA={lam}]")
    err_l = []
    alpha_max = 1 if (lam <= 0.95) else 1 / (2 * (k - sub))
    alpha_l = np.linspace(0, alpha_max, 31)
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


def benchmark(alg_class, title, fn, sub=3):
  fig, ax = plt.subplots()
  fig.suptitle(title, fontsize=BIG_FONT)
  fig.set_size_inches(20, 14)
  def vhat(s, w): return vhat_st_agg(s, w, FIG_12_3_N_ST)
  def nab_vhat(s, w): return nab_vhat_st_agg(s, w, FIG_12_3_N_ST)
  alg = alg_class(RandomWalk(), None, FIG_12_3_N_ST, None, vhat, nab_vhat,
                  FIG_12_3_G)
  xticks, yticks = np.linspace(0, 1, 6), np.linspace(0.25, 0.55, 7)
  def short_str(x): return str(x)[:3]
  xnames, ynames = map(short_str, xticks), map(short_str, yticks)
  run_random_walks(ax, alg, FIG_12_3_LAM_L, FIG_12_3_N_EP, FIG_12_3_N_RUNS, sub)
  plot_figure(ax, '', xticks, xnames, 'alpha', yticks, ynames,
              (f'Average\nRMS error\n({FIG_12_3_N_ST} states,\n ' +
               f'{FIG_12_3_N_EP} episodes)'), font=MED_FONT, labelpad=35,
              loc='upper right')
  save_plot(fn, dpi=100)
  plt.show()


def fig_12_3():
  benchmark(OffLamRet, 'Figure 12.3', 'fig12.3')


def fig_12_6():
  benchmark(SemiGradTDLam, 'Figure 12.6', 'fig12.6')


def fig_12_8():
  # benchmark(OnLamRet, 'Figure 12.8', 'fig12.8')
  benchmark(TrueOnlineTD, 'Figure 12.8', 'fig12.8', sub=4)


def fig_12_10():
  fig, ax = plt.subplots()
  for lam in FIG_12_10_LAM_L:
    print(f"[LAM={lam}]")
    steps_l = []
    alpha_l = np.linspace(FIG_12_10_ALP_MIN, FIG_12_10_ALP_MAX, FIG_12_10_N_PTS)
    for alpha in alpha_l:
      F, qhat = get_fn_mc(N_TIL, N_TLGS)
      alg = SarsaLam(MountainCar(), alpha / N_TLGS, N_TIL * N_TLGS, lam, F,
                     qhat, FIG_12_10_EPS, FIG_12_10_G)
      print(f"[ALPHA={alg.a}]")
      tot_steps = 0
      for seed in range(FIG_12_10_N_RUNS):
        print(f"[RUN #{seed}]")
        alg.reset()
        alg.seed(seed)
        for ep in range(FIG_12_10_N_EP):
          print(f"[EP #{ep}]")
          tot_steps += alg.pol_eva(None, 1, max_steps=FIG_12_10_MAX_STEPS)[0]
      steps_l.append(tot_steps / (FIG_12_10_N_RUNS * FIG_12_10_N_EP))
    plt.plot(alpha_l, steps_l, label=f'lam={lam}')
  xticks, yticks = np.linspace(0.5, 1.5, 5), np.linspace(180, 300, 7)
  left_title = (f'Mountain Car\nSteps per\nepisode\n(averaged \nover ' +
                f'first\n{FIG_12_10_N_EP} episodes\n{FIG_12_10_N_RUNS} runs)')
  plot_figure(ax, 'Figure 12.10', list(xticks) + [1.6], xticks,
              f'alpha * number of tilings ({N_TLGS})',
              [160] + list(yticks), yticks, left_title, labelpad=35)
  fig.set_size_inches(20, 14)
  plt.legend()
  save_plot('fig12.10', dpi=100)
  plt.show()

def fig_12_11():
  fig, ax = plt.subplots()
  F, qhat = get_fn_mc(N_TIL, N_TLGS)
  for alg_name in FIG_12_11_ALG_STR.keys():
    steps_l = []
    alpha_l = np.linspace(*FIG_12_11_ALP_BND[alg_name], FIG_12_11_N_PTS)
    for alpha in alpha_l:
      alg = alg_name(MountainCar(), alpha / N_TLGS, N_TIL * N_TLGS,
                     FIG_12_11_LAM, F, qhat, FIG_12_11_EPS, FIG_12_11_G)
      print(f"[ALPHA={alg.a}]")
      tot_steps = 0
      for seed in range(FIG_12_11_N_RUNS):
        print(f"[RUN #{seed}]")
        alg.reset()
        alg.seed(seed)
        for ep in range(FIG_12_11_N_EP):
          print(f"[EP #{ep}]")
          tot_steps += alg.pol_eva(None, 1, max_steps=FIG_12_11_MAX_STEPS)[0]
      steps_l.append(tot_steps / (FIG_12_11_N_RUNS * FIG_12_11_N_EP))
    plt.plot(alpha_l, -np.array(steps_l), label=FIG_12_11_ALG_STR[alg_name])
  xticks, yticks = np.linspace(0.5, 1.5, 5), np.linspace(180, 300, 7)
  left_title = (f'Mountain Car\nReward per\nepisode\n(averaged \nover ' +
                f'first\n{FIG_12_11_N_EP} episodes\n{FIG_12_11_N_RUNS} runs)')
  # plot_figure(ax, 'Figure 12.11', list(xticks) + [1.6], xticks,
              # f'alpha * number of tilings ({N_TLGS})',
              # yticks, yticks, left_title, labelpad=35)
  fig.set_size_inches(20, 14)
  plt.legend()
  save_plot('fig12.11', dpi=100)
  plt.show()


PLOT_FUNCTION = {
  '12.3': fig_12_3,
  '12.6': fig_12_6,
  '12.8': fig_12_8,
  '12.10': fig_12_10,
  '12.11': fig_12_11,
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
