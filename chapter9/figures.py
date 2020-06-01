import argparse
import matplotlib.pyplot as plt
from randomwalk import RandomWalk, EMPTY_MOVE
from gradient_methods import GradientMC, SemiGradientTD0
from nstep_semi_grad import nStepSemiGrad
from utils import est
import numpy as np
from feat_const import poly_feat

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

FIG_9_5_POL_BAS = [5]
FIG_9_5_ALP = 1e-4
FIG_9_5_N_EP = 10 ** 4
FIG_9_5_G = FIG_9_1_G

def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)

def plot_figure(ax, title, xticks, xnames, xlabel, yticks, ynames, ylabel, labelpad=15):
  ax.set_title(title, fontsize=MED_FONT)
  ax.set_xticks(xticks)
  ax.set_xticklabels(xnames)
  ax.set_yticks(yticks)
  ax.set_yticklabels(ynames)
  ax.set_xlim([min(xticks), max(xticks)])
  ax.set_ylim([min(yticks), max(yticks)])
  ax.set_xlabel(xlabel, fontsize=MED_FONT)
  ax.set_ylabel(ylabel, rotation=0, fontsize=MED_FONT, labelpad=labelpad)
  plt.legend(loc='upper left')

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
  if input("load true values? (Y/n)") != "n":
    print("loading true vals")
    true_vals = np.load('true_vals.arr', allow_pickle=True)
  else:
    true_vals = np.array([est(env, pi, s, FIG_9_2_G, n_ep=FIG_9_2_N_EP_TR) for s in env.states])
    if input("save true values? (y/N)?") != 'n':
      print("saving true vals")
      true_vals.dump('true_vals.arr')
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
  plot_figure(ax1, 'Figure 9.1', [0, 999], [1, 1000], 'State', [-1, 0, 1], [-1, 0, 1], '\n\nValue\nScale')
  ax2 = ax1.twinx()
  ax2.set_yticks([0, 0.0017, 0.0137])
  ax2.set_ylabel('Distribution\nscale', rotation=0, fontsize=MED_FONT)
  ax2.plot(grad_mc.mu[:-1], 'm', label='State distribution mu')
  plt.legend()
  fig.set_size_inches(20, 14)
  save_plot('fig9.1', dpi=100)
  plt.show()

def param_study(ax, alg, pi, vhat, nab_vhat, n_ep, n_runs, true_vals=None, max_n=FIG_9_2_MAX_N, gamma=FIG_9_2_G):
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
  param_study(ax2, nstep_semi_grad, pi, vhat_st_agg, nab_vhat_st_agg, FIG_9_2_N_EP_R, FIG_9_2_N_RUNS_R, true_vals=true_vals, max_n=FIG_9_2_MAX_N, gamma=FIG_9_2_G)
  plt.legend()
  fig.set_size_inches(20, 14)
  save_plot('fig9.2', dpi=100)
  plt.show()

def fig_9_5():
  fig, ax = plt.subplots()
  fig.suptitle('Figure 9.5')
  env = RandomWalk()
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  true_vals = get_true_vals(env, pi)

  for base in FIG_9_5_POL_BAS:
    def vhat_pol(s, w): return np.dot(w, poly_feat(s / 1000, base))
    def nab_vhat_pol(s, w): return poly_feat(s / 1000, base)
    w_dim = base + 1
    grad_mc = GradientMC(env, FIG_9_5_ALP, w_dim)
    grad_mc.seed(0)
    grad_mc.pol_eva(pi, vhat_pol, nab_vhat_pol, FIG_9_5_N_EP, FIG_9_5_G)
    est_vals = [vhat_polynomial(s / 1000, grad_mc.w) for s in env.states][:-1]

PLOT_FUNCTION = {
  '9.1': fig_9_1,
  '9.2': fig_9_2,
  '9.5': fig_9_5,
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
