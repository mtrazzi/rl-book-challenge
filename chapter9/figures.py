import argparse
import matplotlib.pyplot as plt
from randomwalk import RandomWalk, EMPTY_MOVE
from gradient_methods import GradientMC, SemiGradientTD0
from nstep_semi_grad import nStepSemiGrad
from utils import est
import numpy as np
from feat_const import poly_feat, four_feat

MED_FONT = 13

FIG_9_1_ALP = 2e-5
FIG_9_1_W_DIM = 10
FIG_9_1_N_EP = 10 ** 5
FIG_9_1_N_EP_TR = 10 ** 3
FIG_9_1_G = 1

FIG_9_2_ALP = FIG_9_1_ALP
FIG_9_2_W_DIM_L = FIG_9_1_W_DIM
FIG_9_2_W_DIM_R = 20
FIG_9_2_N_EP_L = FIG_9_1_N_EP
FIG_9_2_N_EP_R = 10
FIG_9_2_N_RUNS_R = 100
FIG_9_2_N_EP_TR = FIG_9_1_N_EP_TR
FIG_9_2_G = FIG_9_1_G
FIG_9_2_MAX_N = 512

FIG_9_5_BAS = [5, 10, 20]
FIG_9_5_ALP_POL = 1e-4
FIG_9_5_ALP_FOU = 5e-5
FIG_9_5_N_EP = int(5e3)
FIG_9_5_G = FIG_9_1_G
FIG_9_5_N_RUNS = 3

FIG_9_10_ALP_ST_AGG = 1e-4
FIG_9_10_TIL_L = [1, 50]
FIG_9_10_ALP_TIL_L = [FIG_9_10_ALP_ST_AGG / n_til for n_til in FIG_9_10_TIL_L]
FIG_9_10_TOT_ST = 1000
FIG_9_10_ST_AGG = 200
FIG_9_10_N_EP = int(5e3)
FIG_9_10_G = FIG_9_1_G
FIG_9_10_N_RUNS = 1

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

def enc_st_agg(s, w, tot_st=1000):
  return s // (tot_st // len(w))

def vhat_st_agg(s, w, tot_st=1000):
  if s == tot_st:
    return 0
  return w[enc_st_agg(s, w, tot_st)]

def nab_vhat_st_agg(s, w, tot_st=1000):
  if s == tot_st:
    return 0
  return np.array([i == enc_st_agg(s, w, tot_st) for i in range(len(w))])

def get_true_vals(env, pi):
  #if input("load true values? (Y/n)") != "n":
  print("loading true vals")
  true_vals = np.load('true_vals.arr', allow_pickle=True)
  #else:
  #  true_vals = np.array([est(env, pi, s, FIG_9_2_G, n_ep=FIG_9_2_N_EP_TR) for s in env.states])
  #  if input("save true values? (y/N)?") != 'n':
  #    print("saving true vals")
  #    true_vals.dump('true_vals.arr')
  return true_vals

def fig_9_1():
  env = RandomWalk() 
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  true_vals = get_true_vals(env, pi)

  grad_mc = GradientMC(env, FIG_9_1_ALP, FIG_9_1_W_DIM)
  grad_mc.seed(0)
  grad_mc.pol_eva(pi, vhat_st_agg, nab_vhat_st_agg, FIG_9_1_N_EP, FIG_9_1_G)
  est_vals = [vhat_st_agg(s, grad_mc.w) for s in env.states][:-1]

  fig, ax1 = plt.subplots()
  ax1.plot(est_vals, 'b', label='Approximate MC value vhat')
  ax1.plot(true_vals, 'r', label='True value v_pi')
  plot_figure(ax1, 'Figure 9.1', [0, 999], [1, 1000], 'State',
              [-1, 0, 1], [-1, 0, 1], '\n\nValue\nScale')
  ax2 = ax1.twinx()
  ax2.set_yticks([0, 0.0017, 0.0137])
  ax2.set_ylabel('Distribution\nscale', rotation=0, fontsize=MED_FONT)
  ax2.plot(grad_mc.mu[:-1], 'm', label='State distribution mu')
  plt.legend()
  fig.set_size_inches(20, 14)
  save_plot('fig9.1', dpi=100)
  plt.show()

def param_study(ax, alg, pi, vhat, nab_vhat, n_ep, n_runs, true_vals=None,
                max_n=FIG_9_2_MAX_N, gamma=FIG_9_2_G):
  n_l = [2 ** k for k in range(int(np.log(max_n) / np.log(2)) + 1)]
  for n in n_l:
    alg.n = n
    print(f">> n={n}")
    err_l = []
    alpha_max = 1 if n <= 16 else 1 / (np.log(n // 8) / np.log(2))
    alpha_l = np.linspace(0, alpha_max, 31)
    for alpha in alpha_l:
      alg.a = alpha
      print(f"alpha={alpha}")
      err_sum = 0
      for seed in range(n_runs):
        alg.reset()
        alg.seed(seed)
        for ep in range(n_ep):
          alg.pol_eva(pi, vhat, nab_vhat, n_ep=1, gamma=gamma)
          v_arr = np.array(alg.get_value_list(vhat)[:-1]) # removes absorbing state
          err_sum += np.sqrt(np.sum((v_arr-true_vals[:-1]) ** 2) / alg.env.n_states)
      err_l.append(err_sum / (n_runs * n_ep))
    plt.plot(alpha_l, err_l, label=f'n={n}')
  ax.set_xticks(np.linspace(0, 1, 6))
  yticks = np.linspace(0.25, 0.55, 6)
  ax.set_yticks(yticks)
  ax.set_ylim([min(yticks), max(yticks)])
  ax.set_xlabel('Stepsize')
  ax.set_ylabel(f'Average RMS error ({alg.env.n_states} states, first {n_ep} episodes)')

def fig_9_2():
  fig = plt.figure()
  fig.suptitle('Figure 9.2')
  env = RandomWalk()
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  true_vals = get_true_vals(env, pi)

  semi_grad_td = SemiGradientTD0(env, FIG_9_2_ALP, FIG_9_2_W_DIM_L)
  semi_grad_td.seed(0)
  semi_grad_td.pol_eva(pi, vhat_st_agg, nab_vhat_st_agg, FIG_9_2_N_EP_L, FIG_9_2_G)
  est_vals = [vhat_st_agg(s, semi_grad_td.w) for s in env.states][:-1]
  ax1 = fig.add_subplot('121')
  ax1.plot(est_vals, 'b', label='Approximate TD value vhat')
  ax1.plot(true_vals, 'r', label='True value v_pi')
  plot_figure(ax1, '', [0, 999], [1, 1000], 'State', [-1, 0, 1], [-1, 0, 1], '\n\nValue\nScale')

  nstep_semi_grad = nStepSemiGrad(env, None, FIG_9_2_W_DIM_R, FIG_9_2_G, 0)
  ax2 = fig.add_subplot('122')
  param_study(ax2, nstep_semi_grad, pi, vhat_st_agg, nab_vhat_st_agg,
              FIG_9_2_N_EP_R, FIG_9_2_N_RUNS_R, true_vals=true_vals,
              max_n=FIG_9_2_MAX_N, gamma=FIG_9_2_G)
  plt.legend()
  fig.set_size_inches(20, 14)
  save_plot('fig9.2', dpi=100)
  plt.show()

def fig_9_5():
  fig, ax = plt.subplots()
  env = RandomWalk()
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  true_vals = get_true_vals(env, pi)

  for (feat, alp, label) in [(poly_feat, FIG_9_5_ALP_POL, 'polynomial basis'),
                        (four_feat, FIG_9_5_ALP_FOU, 'fourier basis')]:
    for base in FIG_9_5_BAS:
      def vhat(s, w): return np.dot(w, feat(s / 1000, base))
      def nab_vhat(s, w): return feat(s / 1000, base)
      w_dim = base + 1
      grad_mc = GradientMC(env, alp, w_dim)
      err_sum = np.zeros(FIG_9_5_N_EP)
      for seed in range(FIG_9_5_N_RUNS):
        print(f"seed={seed}")
        grad_mc.reset()
        grad_mc.seed(seed)
        err_per_ep = []
        for ep in range(FIG_9_5_N_EP):
          if ep % 100 == 0 and ep > 0:
            print(ep)
          grad_mc.pol_eva(pi, vhat, nab_vhat, n_ep=1, gamma=FIG_9_5_G)
          est_vals = [vhat(s, grad_mc.w) for s in env.states][:-1]
          err_per_ep.append(np.sqrt(np.sum((est_vals-true_vals[:-1]) ** 2) / env.n_states))
        err_sum += err_per_ep
      plt.plot(err_sum / FIG_9_5_N_RUNS, label=f'{label}, n={base}')
  plt.legend()
  plot_figure(ax, 'Figure 9.5', [0, 5000], [0, 5000], "Episodes",
             [0, 0.1, 0.2, 0.3, 0.4], ['0', '0.1', '0.2', '0.3', '0.4'],
             f"Root\nMean\nSquared\nValue\nError\n({FIG_9_5_N_RUNS} runs)",
             labelpad=30, font=MED_FONT, loc='lower left')
  fig.set_size_inches(20, 14)
  save_plot('fig9.5', dpi=100)
  plt.show()

def fig_9_10():
  fig, ax = plt.subplots()
  env = RandomWalk()
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  true_vals = get_true_vals(env, pi)

  def feat_tile(s, offset, st_per_agg):
    if s < offset:
      return 0
    return (s - offset) // st_per_agg + 1

  def feat(s, st_per_agg, n_tiles):
    dx = st_per_agg // n_tiles if n_tiles > 1 else 0
    ft_per_til = FIG_9_10_TOT_ST // st_per_agg + (n_tiles > 1)
    feat_arr = np.zeros(ft_per_til * n_tiles)
    for n in range(n_tiles):
      idx_min = n * ft_per_til
      s_id = feat_tile(s, n * dx, st_per_agg) - (n_tiles == 1)
      feat_arr[idx_min + s_id] = True
    return feat_arr.astype(bool)

  for (idx, n_tiles) in enumerate(FIG_9_10_TIL_L):
    def feat_vec(s): return feat(s, FIG_9_10_ST_AGG, n_tiles)
    def vhat(s, w): return np.sum(w[feat_vec(s)]) if s < FIG_9_10_TOT_ST else 0
    def nab_vhat(s, w): return feat_vec(s) if s < FIG_9_10_TOT_ST else 0
    w_dim = (FIG_9_10_TOT_ST // FIG_9_10_ST_AGG + (n_tiles > 1)) * n_tiles
    grad_mc = GradientMC(env, FIG_9_10_ALP_TIL_L[idx], w_dim)
    print(f"w_dim={w_dim}, alpha={grad_mc.a}, n_tiles={n_tiles}")
    err_sum = np.zeros(FIG_9_10_N_EP)
    for seed in range(FIG_9_10_N_RUNS):
      print(f"seed={seed}")
      grad_mc.reset()
      grad_mc.seed(seed)
      err_per_ep = []
      for ep in range(FIG_9_10_N_EP):
        if ep % 100 == 0 and ep > 0:
          print(ep)
        grad_mc.pol_eva(pi, vhat, nab_vhat, n_ep=1, gamma=FIG_9_10_G)
        est_vals = [vhat(s, grad_mc.w) for s in env.states][:-1]
        err_per_ep.append(np.sqrt(np.sum((est_vals-true_vals[:-1]) ** 2) / env.n_states))
      err_sum += np.array(err_per_ep)
    plt.plot(err_sum / FIG_9_10_N_RUNS, label=f'{n_tiles} tiles')
  plt.legend()
  plot_figure(ax, 'Figure 9.10', [0, 5000], [0, 5000], "Episodes",
             [0, 0.1, 0.2, 0.3, 0.4], ['0', '0.1', '0.2', '0.3', '0.4'],
             f"Root\nMean\nSquared\nValue\nError\n({FIG_9_10_N_RUNS} runs)",
             labelpad=30, font=MED_FONT, loc='lower left')
  fig.set_size_inches(20, 14)
  save_plot('fig9.10', dpi=100)
  plt.show()

PLOT_FUNCTION = {
  '9.1': fig_9_1,
  '9.2': fig_9_2,
  '9.5': fig_9_5,
  '9.10': fig_9_10,
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
