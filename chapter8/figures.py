import argparse
import matplotlib.pyplot as plt
from dyna_q import DynaQ
from dyna_maze import DynaMaze 
from models import FullModel
from tabular_q import TabularQ
from utils import to_arr
import seaborn as sns

SEC_8_1_ALP = 0.001
SEC_8_1_N_STEPS = int(1e6)
DYNA_MAZE_GAMMA = 0.95
FIG_8_2_ALP = 0.1
FIG_8_2_N_EP = 50
FIG_8_2_EPS = 0.1

def save_plot(filename):
  plt.savefig('plots/' + filename + '.png')

def section_8_1():
  env = DynaMaze()
  alg = TabularQ(FullModel(env), SEC_8_1_ALP, DYNA_MAZE_GAMMA)
  alg.seed(0)
  alg.rand_sam_one_step_pla(SEC_8_1_N_STEPS, decay=True)
  V = alg.get_V()
  plt.title('Section 8.1 - tabular Q (1-step random sample, dyna maze)')
  sns.heatmap(to_arr(V), cbar_kws={'label': 'max(Q(s, a))'})
  save_plot('section8.1')
  plt.show()

def fig_8_2():
  fig, ax = plt.subplots()
  env = DynaMaze()
  alg = DynaQ(env, SEC_8_1_ALP, DYNA_MAZE_GAMMA, FIG_8_2_EPS)
  alg.seed(0)
  xticks = [2, 10, 20, 30, 40, 50]
  yticks = [14, 200, 400, 600, 800]
  ax.set_xlim([min(xticks), max(xticks)])
  ax.set_ylim([min(yticks), max(yticks)])
  ax.set_xlabel('Episodes')
  ax.set_ylabel('Steps\nper\nepisode', rotation=0)
  plt.plot(alg.tabular_dyna_q(FIG_8_2_N_EP))
  save_plot('fig8.2')
  plt.show()

PLOT_FUNCTION = {
  'section8.1': section_8_1,
  '8.2': fig_8_2,
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
