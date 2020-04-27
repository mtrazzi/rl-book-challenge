import argparse
import matplotlib.pyplot as plt
from nstep_td import nStepTD
import numpy as np
from randomwalk import RandomWalk, EMPTY_MOVE

UND = 1
FIG_7_2_N_EP = 10
FIG_7_2_N_STATES = 19
FIG_7_2_N_RUNS = 100

def true_values(n_states):
  return np.linspace(-1, 1, n_states + 2)[1:-1]

def comp_true_vals(n_states, n_ep_per_s=10000):
  env = RandomWalk(n_states)
  vals = []
  for state in range(n_states + 1):
    print(state)
    tot = 0
    for _ in range(n_ep_per_s):
      r_ep, d = 0, False
      env.force_state(state)
      while not d:
        _, r, d, _ = env.step(EMPTY_MOVE)
        r_ep += r
      tot += r_ep
    vals.append(tot / n_ep_per_s)
  return np.array(vals[:-1])

def ex_7_1():
  pass

def fig_7_2():
  fig, ax = plt.subplots()
  ax.set_title('Figure 7.2')
  n_l = [1, 2, 3]
  alpha_l = np.linspace(0, 1, 11)[1:]
  env = RandomWalk(n_states=FIG_7_2_N_STATES)
  pi = {(a, s): 1.0 for s in env.states for a in env.moves_d[s]}
  true_vals = true_values(env.n_states)
  alg = nStepTD(env, V_init=None, step_size=None, gamma=UND, n=None)
  for n in n_l:
    alg.n = n
    print(f">> n={n}")
    err_l = []
    for alpha in alpha_l:
      alg.step_size = alpha
      print(f"alpha={alpha}")
      err_sum = 0
      for seed in range(FIG_7_2_N_RUNS):
        alg.reset()
        alg.seed(seed)
        alg.pol_eval(pi, n_ep=FIG_7_2_N_EP)
        #alg.simple_td(pi, n_ep=FIG_7_2_N_EP)
        v_arr = np.array(alg.get_value_list()[:-1])
        err_sum += np.linalg.norm(v_arr-true_vals)
      err_l.append(err_sum / FIG_7_2_N_RUNS)
    plt.plot(alpha_l, err_l, label=f'n={n}')
  ax.set_xticks(np.linspace(0, 1, 6))
  ax.set_xlabel('Stepsize')
  ax.set_ylabel(f'Average RMS error ({FIG_7_2_N_STATES} states, first {FIG_7_2_N_EP} episodes)')
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
