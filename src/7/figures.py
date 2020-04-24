import argparse
import matplotlib.pyplot as plt
from nstep_td import nStepTD
import numpy as np
from randomwalk import RandomWalk

UND = 1
FIG_7_2_N_EP = 10
FIG_7_2_N_STATES = 19

def true_values(n_states):
  return np.linspace(-1, 1, n_states + 2)[1:-1]

def ex_7_1():
  pass

def fig_7_2():
  n_l = [1]
  alpha_l = np.linspace(0, 1, 40)
  env = RandomWalk(n_states=FIG_7_2_N_STATES)
  pi = {(a, s): 1.0 for s in env.states for a in env.moves_d[s]}
  true_vals = true_values(env.n_states)
  for n in n_l:
    err_l = []
    for alpha in alpha_l:
      alg = nStepTD(env, V_init=None, step_size=alpha, gamma=UND, n=n)
      alg.pol_eval(pi, n_ep=FIG_7_2_N_EP)
      v_arr = np.array(alg.get_value_list()[:-1])
      err_l.append(np.linalg.norm(v_arr-true_vals))
    plt.plot(alpha_l, err_l, label=f'n={n}')
  plt.legend()
  plt.show()

PLOT_FUNCTION = {
  'ex7.1': ex_7_1,
  '7.2': fig_7_2,
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=PLOT_FUNCTION.keys())
  args = parser.parse_args()

  PLOT_FUNCTION[args.figure]()

if __name__ == "__main__":
  main()
