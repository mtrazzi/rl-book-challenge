import argparse
from td import OneStepTD
from driving import DrivingEnv, TRAVEL_TIME
import matplotlib.pyplot as plt

def print_driving_home(states, V_old, V_new):
  fig, ax = plt.subplots()
  def pred(V):
    return [V[idx] + sum(TRAVEL_TIME[:idx]) for idx in range(len(V))]
  fig.suptitle('Figure 6.1')
  ax.set_xticks([0, 1, 2, 3, 4, 5])
  ax.set_xticklabels(states, fontdict={'fontsize': 8})
  ax.set_xlabel('Situation')
  ax.set_ylabel('Predicted total travel time')
  ax.set_title('TD methods')
  plt.plot(pred(V_old), color='#000000', label='actual outcome')
  plt.plot(pred(V_new), color='blue', label='after update')
  plt.legend()
  plt.show()

def fig_6_1():
  env = DrivingEnv()
  V_0 = [30, 35, 15, 10, 3, 0]
  alg = OneStepTD(env, step_size=1, gamma=1)
  # predicted total travel time
  alg.V = {s: V_0[idx] for (idx, s) in enumerate(env.states)}
  pi = {(a, s): 1.0 for s in env.states for a in env.moves}
  alg.tabular_td_0(pi)
  V_new = [val for key,val in alg.V.items()]
  print_driving_home(env.states, V_0, V_new)

PLOT_FUNCTION = {
  '6.1': fig_6_1,
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=PLOT_FUNCTION.keys())
  args = parser.parse_args()

  if args.figure in ['6.1']:
    PLOT_FUNCTION[args.figure]()

if __name__ == "__main__":
  main()
