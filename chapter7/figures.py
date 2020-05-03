import argparse
import matplotlib.pyplot as plt
from nstep_td import nStepTD
import numpy as np
from randomwalk import RandomWalk, NotSoRandomWalk, EMPTY_MOVE
from windy_gridworld import WindyGridworld
from nstep_sarsa import nStepSarsa
from off_pol_nstep_exp_sarsa import OffPolnStepExpSarsa
from off_pol_nstep_qsigma import OffPolnStepQSigma
from off_pol_nstep_sarsa import OffPolnStepSarsa
from off_pol_nstep_td import OffPolnStepTD
from nstep_tree_backup import nStepTreeBackup

UND = 1
FIG_7_2_N_EP = 10
FIG_7_2_N_STATES = 19
FIG_7_2_N_RUNS = 100
FIG_7_2_MAX_N = 512
EX_7_2_N_RUNS = 1
EX_7_3_N_RUNS = 1
EX_7_3_N_STATES = 5
FIG_7_4_STEPSIZE = 0.5
FIG_7_4_N_EP = 170
FIG_7_4_MAX_N = 16
SECTION_7_3_STEPSIZE = 0.01
SECTION_7_3_N_EP_TRAIN = 200
SECTION_7_3_MAX_N = 8
EX_7_7_N_EP_TRAIN = 1000
EX_7_7_STEPSIZE = 0.001
EX_7_7_MAX_N = 8
EX_7_10_N_EP_TRAIN = 10000
EX_7_10_STEPSIZE = 0.0001
EX_7_10_MAX_N = 2
EX_7_10_N_BATCHES = 5
EX_7_10_N_STATES = 5
SECTION_7_5_STEPSIZE = 0.01
SECTION_7_5_N_EP_TRAIN = 200
SECTION_7_5_MAX_N = 8
SECTION_7_6_STEPSIZE = 0.01
SECTION_7_6_N_EP_TRAIN = 100
SECTION_7_6_N = 2
SECTION_7_6_SIGMA_L = [0, 0.25, 0.5, 0.75, 1]

def run_random_walks(ax, ex_7_2=False, show=True, extra_label='', dashed=False, n_runs=FIG_7_2_N_RUNS, n_states=FIG_7_2_N_STATES, left_rew=-1, true_vals=None, V_init=None):
  n_l = [2 ** k for k in range(int(np.log(FIG_7_2_MAX_N) / np.log(2)) + 1)]
  env = RandomWalk(n_states=n_states, r_l=left_rew)
  pi = {(a, s): 1.0 for s in env.states for a in env.moves_d[s]}
  true_vals = np.linspace(-1, 1, env.n_states + 2)[1:-1] if true_vals is None else true_vals 
  alg = nStepTD(env, V_init=V_init, step_size=None, gamma=UND, n=n_l[0], ex_7_2=ex_7_2)
  for n in n_l:
    alg.n = n
    print(f">> n={n}")
    err_l = []
    alpha_max = 1 if (n <= 16 or ex_7_2) else 1 / (np.log(n // 8) / np.log(2))
    alpha_l = np.linspace(0, alpha_max, 31)
    for alpha in alpha_l:
      alg.step_size = alpha
      print(f"alpha={alpha}")
      err_sum = 0
      for seed in range(n_runs):
        alg.reset()
        alg.seed(seed)
        for ep in range(FIG_7_2_N_EP):
          alg.pol_eval(pi, n_ep=1)
          v_arr = np.array(alg.get_value_list()[:-1])
          err_sum += np.sqrt(np.sum((v_arr-true_vals) ** 2) / env.n_states)
      err_l.append(err_sum / (n_runs * FIG_7_2_N_EP))
    plt.plot(alpha_l, err_l, label=f'{extra_label} n={n}', linestyle='dashed' if dashed else None)
  ax.set_xticks(np.linspace(0, 1, 6))
  yticks = np.linspace(0.25, 0.55, 6)
  ax.set_yticks(yticks)
  ax.set_ylim([min(yticks), max(yticks)])
  ax.set_xlabel('Stepsize')
  ax.set_ylabel(f'Average RMS error ({env.n_states} states, first {FIG_7_2_N_EP} episodes)')

def ex_7_2():
  fig, ax = plt.subplots()
  ax.set_title('Exercise 7.2')
  run_random_walks(ax, ex_7_2=True, extra_label='td error sum', dashed=True, n_runs=EX_7_2_N_RUNS)
  run_random_walks(ax, ex_7_2=False, extra_label='n-step td', n_runs=EX_7_2_N_RUNS)
  plt.legend(fontsize='x-small')
  plt.savefig('plots/ex7.2.png')
  plt.show()

def fig_7_2():
  fig, ax = plt.subplots()
  ax.set_title('Figure 7.2')
  run_random_walks(ax)
  plt.legend(fontsize='x-small')
  plt.savefig('plots/fig7.2.png')
  plt.show()

def ex_7_3():
  # testing with 5 states
  fig = plt.figure()
  fig.suptitle('Exercise 7.3')
  ax = fig.add_subplot('121')
  ax.set_title(f'{EX_7_3_N_STATES} states')
  run_random_walks(ax, n_runs=EX_7_3_N_RUNS, n_states=EX_7_3_N_STATES)
  yticks = np.linspace(0.15, 0.55, 8)
  ax.set_yticks(yticks)
  ax.set_ylim([min(yticks), max(yticks)]) 
  
  # testing with a left reward of 0 with better true values
  ax2 = fig.add_subplot('122')
  ax2.set_title(f'r=0 left, {FIG_7_2_N_STATES} states')
  true_vals = np.linspace(0, 1, FIG_7_2_N_STATES + 2)[1:-1]
  V_init = {s: 1/2 if s != FIG_7_2_N_STATES else 0 for s in range(FIG_7_2_N_STATES + 1)}
  yticks = np.linspace(0.1, 0.3, 11)
  run_random_walks(ax2, n_runs=EX_7_3_N_RUNS, n_states=FIG_7_2_N_STATES, left_rew=0, true_vals=true_vals, V_init=V_init)
  ax2.set_yticks(yticks)
  ax2.set_ylim([min(yticks), max(yticks)])
  plt.legend(fontsize='x-small')
  plt.savefig('plots/ex7.3.png')
  plt.show()

def run_alg(alg, title, filename, n_ep, k_min, n_max, x_label='Timesteps', y_label='Episodes', show=True, extra_label='', ax=None, reset=True, dashed=False):
  if ax is None:
    fig, ax = plt.subplots()
    ax.set_title(title)
  n_l = [2 ** k for k in range(k_min, int(np.log(n_max) / np.log(2)) + 1)]
  alg.seed(0)
  for n in n_l:
    alg.n = n
    if reset:
      alg.reset()
    plt.plot(alg.pol_eval(n_ep), label=f'{extra_label}n={n}', linestyle='dashed' if dashed else None)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  if show:
    plt.legend()
    plt.savefig(filename)
    plt.show()

def fig_7_4():
  env = WindyGridworld()
  alg = nStepSarsa(env, step_size=FIG_7_4_STEPSIZE, gamma=UND, n=2)
  run_alg(alg, f'Figure 7.4 - n-step sarsa on windy gridworld (alpha={alg.step_size})', 'plots/fig7.4.png', FIG_7_4_N_EP, 0, FIG_7_4_MAX_N)

def section_7_3():
  env = NotSoRandomWalk()
  alg = OffPolnStepSarsa(env, b=None, step_size=SECTION_7_3_STEPSIZE, gamma=UND, n=2)
  run_alg(alg, f'Section 7.3 - off-policy n-step sarsa on (not so) random walk\n({env.n_states} states, alpha={alg.step_size})', 'plots/section7.3.png', SECTION_7_3_N_EP_TRAIN, 1, SECTION_7_3_MAX_N, 'Train episodes', 'avg episode length for 10 test episodes\n (+ moving average)')

def ex_7_7():
  env = NotSoRandomWalk()
  alg = OffPolnStepExpSarsa(env, b=None, step_size=EX_7_7_STEPSIZE, gamma=UND, n=2)
  run_alg(alg, f'Exercise 7.7 - off policy n-step expected sarsa on (not so) random walk\n({env.n_states} states, alpha={alg.step_size})', 'plots/ex7.7.png', EX_7_7_N_EP_TRAIN, 1, EX_7_7_MAX_N, 'Train episodes', 'avg episode length for 10 test episodes\n(+ moving average)')

def ex_7_10():
  fig, ax = plt.subplots()
  env = NotSoRandomWalk(n_states=EX_7_10_N_STATES, r_l=0)
  alg_simple, alg_off_pol = [OffPolnStepTD(env, b=None, step_size=EX_7_10_STEPSIZE, gamma=UND, n=2, simple=is_simple) for is_simple in [True, False]]
  for batch in range(EX_7_10_N_BATCHES):
    for (alg, dashed, extra_lab) in [(alg_simple, True, '(7.1) & (7.9)'), (alg_off_pol, False, '(7.2) & (7.13)')]:
      print((batch + 1) * EX_7_10_N_EP_TRAIN)
      run_alg(alg, '', '', EX_7_10_N_EP_TRAIN, 1, EX_7_10_MAX_N, 'States', 'Value', show=False, ax=ax, extra_label=f'{extra_lab} {(batch + 1) * EX_7_10_N_EP_TRAIN} ep. ', reset=False, dashed=dashed)
  ax.set_title(f'Exercise 7.10 - Off Pol. n-step TD on \n(not so) random walk ({env.n_states} states, alpha={alg.step_size})')
  fig.set_size_inches(8, 6)
  plt.legend()
  plt.savefig('plots/ex7.10.png', dpi=100)
  plt.show()

def section_7_5():
  env = NotSoRandomWalk(n_states=19)
  alg = nStepTreeBackup(env, step_size=SECTION_7_5_STEPSIZE, gamma=UND, n=1)
  run_alg(alg, f'Section 7.5 - n-step tree backup on (not so) random walk\n({env.n_states} states, alpha={alg.step_size})', 'plots/section7.5.png', SECTION_7_5_N_EP_TRAIN, 1, SECTION_7_5_MAX_N, 'Train episodes', 'avg episode length for 10 test episodes\n (+ moving average)')

def section_7_6():
  env = NotSoRandomWalk()
  fig, ax = plt.subplots()
  for sigma in SECTION_7_6_SIGMA_L:
    alg = OffPolnStepQSigma(env, sigma_f=sigma, step_size=SECTION_7_6_STEPSIZE, gamma=UND, n=SECTION_7_6_N)
    plt.plot(alg.pol_eval(SECTION_7_6_N_EP_TRAIN), label=f'sigma={sigma}')
  ax.set_title(f'Section 7.6 - Off Pol. n-step Q(sigma) on \n(not so) random walk ({env.n_states} states, alpha={alg.step_size}, n={SECTION_7_6_N})')
  ax.set_xlabel('Train episodes')
  ax.set_ylabel('avg episode length for 10 test episodes\n (+ moving average)')
  plt.legend()
  plt.savefig('plots/section7.6.png', dpi=100)
  plt.show()

PLOT_FUNCTION = {
  'ex7.2': ex_7_2,
  '7.2': fig_7_2,
  'ex7.3': ex_7_3,
  '7.4': fig_7_4,
  'section7.3': section_7_3,
  'ex7.7': ex_7_7,
  'ex7.10': ex_7_10,
  'section7.5': section_7_5,
  'section7.6': section_7_6,
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=list(PLOT_FUNCTION.keys()) + ['all'])
  args = parser.parse_args()

  if args.figure == 'all':
    for key, f in PLOT_FUNCTION.items():
      if key not in ['ex7.2', '7.2']:
        f()
  else:
    PLOT_FUNCTION[args.figure]()

if __name__ == "__main__":
  main()
