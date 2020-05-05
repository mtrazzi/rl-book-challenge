import argparse
import matplotlib.pyplot as plt
from dyna_q import DynaQ
from dyna_maze import DynaMaze 
from models import FullModel
from tabular_q import TabularQ
from utils import to_arr
import seaborn as sns
import numpy as np

SEC_8_1_ALP = 0.001
SEC_8_1_N_STEPS = int(1e6)
DYNA_MAZE_GAMMA = 0.95
FIG_8_2_ALP = 0.1
FIG_8_2_N_EP = 50
FIG_8_2_EPS = 0.1
FIG_8_2_PLAN_STEPS = [0, 5, 50]
FIG_8_2_C_DIC = {0: 'b', 5: 'g', 50: 'r'}
FIG_8_2_N_RUNS = 30
BIG_FONT = 15

def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)

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
  ax.set_title('Figure 8.2', fontsize=BIG_FONT)
  ax.set_xlim([min(xticks), max(xticks)])
  ax.set_ylim([0, max(yticks)])
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)
  ax.set_xlabel('Episodes', fontsize=BIG_FONT)
  ax.set_ylabel('Steps\nper\nepisode', rotation=0, labelpad=25, fontsize=BIG_FONT)
  ep_ticks = list(range(2, 51))
  for n_plan_steps in FIG_8_2_PLAN_STEPS:
    arr_sum = np.zeros(FIG_8_2_N_EP)
    for seed in range(FIG_8_2_N_RUNS):
      print(seed)
      alg.reset()
      alg.seed(seed)
      arr_sum += np.array(alg.tabular_dyna_q(FIG_8_2_N_EP, n_plan_steps))
    plt.plot(ep_ticks, (arr_sum / FIG_8_2_N_RUNS)[1:], label=f'{n_plan_steps} planning steps', color=FIG_8_2_C_DIC[n_plan_steps])
  plt.legend()
  fig.set_size_inches(10, 8)
  save_plot('fig8.2', dpi=100)
  plt.show()

def fig_8_3():
  env = DynaMaze()
  alg = DynaQ(env, SEC_8_1_ALP, DYNA_MAZE_GAMMA, FIG_8_2_EPS)
  alg.tabular_dyna_q(2)
  pass

PLOT_FUNCTION = {
  'section8.1': section_8_1,
  '8.2': fig_8_2,
  '8.3': fig_8_3, 
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
