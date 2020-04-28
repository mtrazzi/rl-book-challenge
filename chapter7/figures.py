import argparse
import matplotlib.pyplot as plt
from nstep_td import nStepTD
import numpy as np
from randomwalk import RandomWalk, EMPTY_MOVE

UND = 1
FIG_7_2_N_EP = 10
FIG_7_2_N_STATES = 19
FIG_7_2_N_RUNS = 100
FIG_7_2_MAX_N = 512

def fig_7_2():
  fig, ax = plt.subplots()
  ax.set_title('Figure 7.2')
  n_l = [2 ** k for k in range(int(np.log(512) / np.log(2)) + 1)]
  env = RandomWalk(n_states=FIG_7_2_N_STATES)
  pi = {(a, s): 1.0 for s in env.states for a in env.moves_d[s]}
  true_vals = np.linspace(-1, 1, env.n_states + 2)[1:-1] 
  alg = nStepTD(env, V_init=None, step_size=None, gamma=UND, n=None)
  for n in n_l:
    alg.n = n
    print(f">> n={n}")
    err_l = []
    alpha_max = 1 if n <= 16 else 1 / (np.log(n // 8) / np.log(2))
    alpha_l = np.linspace(0, alpha_max, 31)
    for alpha in alpha_l:
      alg.step_size = alpha
      print(f"alpha={alpha}")
      err_sum = 0
      for seed in range(FIG_7_2_N_RUNS):
        alg.reset()
        alg.seed(seed)
        for ep in range(FIG_7_2_N_EP):
          alg.pol_eval(pi, n_ep=1)
          v_arr = np.array(alg.get_value_list()[:-1])
          err_sum += np.sqrt(np.sum((v_arr-true_vals) ** 2) / FIG_7_2_N_STATES)
      err_l.append(err_sum / (FIG_7_2_N_RUNS * FIG_7_2_N_EP))
    plt.plot(alpha_l, err_l, label=f'n={n}')
  ax.set_xticks(np.linspace(0, 1, 6))
  yticks = np.linspace(0.25, 0.55, 6)
  ax.set_yticks(yticks)
  ax.set_ylim([min(yticks), max(yticks)])
  ax.set_xlabel('Stepsize')
  ax.set_ylabel(f'Average RMS error ({FIG_7_2_N_STATES} states, first {FIG_7_2_N_EP} episodes)')
  plt.legend(fontsize='x-small')
  plt.show()

PLOT_FUNCTION = {
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
