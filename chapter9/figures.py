import argparse
import matplotlib.pyplot as plt
from randomwalk import RandomWalk, EMPTY_MOVE
from gradient_mc import GradientMC

EXA_9_1_ALP = 2e-5
EXA_9_1_W_DIM = 100

def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)

def example_9_1():
  env = RandomWalk() 
  grad_mc = GradientMC(env, EXA_9_1_ALP, EXA_9_1_W_DIM)
  pi = {(EMPTY_MOVE, s): 1 for s in env.states}
  traj = grad_mc.gen_traj_ret(pi)
  print(traj)

PLOT_FUNCTION = {
  'example9.1': example_9_1,
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
