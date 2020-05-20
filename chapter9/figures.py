import argparse
import matplotlib.pyplot as plt
from randomwalk import RandomWalk, EMPTY_MOVE
from gradient_mc import GradientMC
from utils import est
import numpy as np

EXA_9_1_ALP = 2e-5
EXA_9_1_W_DIM = 100
EXA_9_1_N_EP = 10000
EXA_9_1_N_EP_TR = 100
EXA_9_1_G = 1

def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)

def plot_figure(ax, filename, title, xticks, xnames, xlabel, yticks, ynames, ylabel):
  ax.set_title(title)
  plt.xticks(xticks, xnames)
  plt.yticks(yticks, ynames)
  ax.set_xlim([min(xticks), max(xticks)])
  ax.set_ylim([min(yticks), max(yticks)])
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  plt.legend()
  save_plot(filename, dpi=100)
  plt.show()

def fig_9_1():
  def enc(s, w): return int(s // len(w))
  def vhat(s, w): return w[enc(s, w)]
  def nab_vhat(s, w):
    return np.array([i == enc(s, w) for i in range(len(w))])

  env = RandomWalk() 
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  grad_mc = GradientMC(env, EXA_9_1_ALP, EXA_9_1_W_DIM)
  grad_mc.seed(0)
  grad_mc.pol_eva(pi, vhat, nab_vhat, EXA_9_1_N_EP, EXA_9_1_G)
  est_vals = [vhat(s, grad_mc.w) for s in env.states][:-1]
  true_vals = [est(env, pi, s, EXA_9_1_G, n_ep=EXA_9_1_N_EP_TR) for s in env.states]

  # plotting
  fig, ax = plt.subplots()
  plt.plot(est_vals, label='Approximate MC value vhat')
  plt.plot(true_vals, label='True value v_pi')
  plot_figure(ax, 'fig9.1', 'Figure 9.1', [0, 999], [1, 1000], 'State', [-1, 0, 1], [-1, 0, 1], 'Value Scale')

PLOT_FUNCTION = {
  '9.1': fig_9_1,
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
