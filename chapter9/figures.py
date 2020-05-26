import argparse
import matplotlib.pyplot as plt
from randomwalk import RandomWalk, EMPTY_MOVE
from gradient_methods import GradientMC, SemiGradientTD0
from nstep_semi_grad import nStepSemiGrad
from utils import est
import numpy as np

MED_FONT = 13

FIG_9_1_ALP = 2e-5
FIG_9_1_W_DIM = 100
FIG_9_1_N_EP = 10 ** 5
FIG_9_1_N_EP_TR = 1000
FIG_9_1_G = 1

FIG_9_2_ALP = FIG_9_1_ALP
FIG_9_2_W_DIM = FIG_9_1_W_DIM
FIG_9_2_N_EP = int(5e2)
FIG_9_2_N_EP_TR = 10
FIG_9_2_G = 1
FIG_9_2_N = 2

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

def enc_st_agg(s, w):
  return s // len(w)

def vhat_st_agg(s, w):
  return w[enc_st_agg(s, w)]

def nab_vhat_st_agg(s, w):
  return np.array([i == enc_st_agg(s, w) for i in range(len(w))])

def fig_9_1():
  env = RandomWalk() 
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  grad_mc = GradientMC(env, FIG_9_1_ALP, FIG_9_1_W_DIM)
  grad_mc.seed(0)
  grad_mc.pol_eva(pi, vhat_st_agg, nab_vhat_st_agg, FIG_9_1_N_EP, FIG_9_1_G)
  est_vals = [vhat_st_agg(s, grad_mc.w) for s in env.states][:-1]
  true_vals = [est(env, pi, s, FIG_9_1_G, n_ep=FIG_9_1_N_EP_TR) for s in env.states]

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

def fig_9_2():
  fig = plt.figure()
  env = RandomWalk()
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}

  semi_grad_td = SemiGradientTD0(env, FIG_9_2_ALP, FIG_9_2_W_DIM)
  semi_grad_td.seed(0)
  #semi_grad_td.pol_eva(pi, vhat_st_agg, nab_vhat_st_agg, FIG_9_2_N_EP, FIG_9_2_G)
  #est_vals = [vhat_st_agg(s, semi_grad_td.w) for s in env.states][:-1]
  #ax1 = fig.add_subplot('211')
  ##true_vals = [est(env, pi, s, FIG_9_2_G, n_ep=FIG_9_2_N_EP_TR) for s in env.states]
  #ax1.plot(est_vals, 'b', label='Approximate MC value vhat')
  ##ax1.plot(true_vals, 'r', label='True value v_pi')
  #plot_figure(ax1, 'Figure 9.2', [0, 999], [1, 1000], 'State', [-1, 0, 1], [-1, 0, 1], '\n\nValue\nScale')

  nstep_semi_grad = nStepSemiGrad(env, FIG_9_2_ALP, FIG_9_2_W_DIM, FIG_9_2_G, FIG_9_2_N)
  nstep_semi_grad.pol_eval(pi, vhat_st_agg, nab_vhat_st_agg, FIG_9_2_N_EP)
  save_plot('fig9.2', dpi=100)
  plt.show()

PLOT_FUNCTION = {
  '9.1': fig_9_1,
  '9.2': fig_9_2,
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
