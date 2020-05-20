import argparse
import matplotlib.pyplot as plt
from randomwalk import RandomWalk, EMPTY_MOVE
from gradient_mc import GradientMC
import numpy as np

EXA_9_1_ALP = 2e-5
EXA_9_1_W_DIM = 100
EXA_9_1_N_EP = 1000

def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)

def plot_figure(filename, title, to_plot):
  plt.title(title)
  plt.plot(to_plot)
  save_plot(filename, dpi=100)
  plt.show()

def fig_9_1():
  env = RandomWalk() 
  grad_mc = GradientMC(env, EXA_9_1_ALP, EXA_9_1_W_DIM)
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  def enc(s, w): return int(s // len(w))
  def vhat(s, w): return w[enc(s, w)]
  def nab_vhat(s, w):
    return np.array([i == enc(s, w) for i in range(len(w))])
  grad_mc.seed(0)
  grad_mc.pol_eva(pi, vhat, nab_vhat, EXA_9_1_N_EP)
  est_vals = [vhat(s, grad_mc.w) for s in env.states][:-1]
  plot_figure('fig9.1', 'Figure 9.1', est_vals)

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
