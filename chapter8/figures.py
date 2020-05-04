import argparse
import matplotlib.pyplot as plt
from dyna_maze import DynaMaze 
from tabular_q import TabularQ
from utils import to_arr
import seaborn as sns

SEC_8_1_ALP = 0.001
SEC_8_1_N_STEPS = int(1e6)
DYNA_MAZE_GAMMA = 0.95

def section_8_1():
  env = DynaMaze()
  alg = TabularQ(env, SEC_8_1_ALP, DYNA_MAZE_GAMMA)
  alg.random_sample_one_step_planning(SEC_8_1_N_STEPS, decay=True)
  V = alg.get_V()
  print(to_arr(V))
  sns.heatmap(to_arr(V))
  plt.show()

PLOT_FUNCTION = {
  'section8.1': section_8_1,
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=list(PLOT_FUNCTION.keys()))
  args = parser.parse_args()
  PLOT_FUNCTION[args.figure]()

if __name__ == '__main__':
  main()
