# In this repo

1. Python replication of all the plots from [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018trimmed.pdf)
2. Solution for all of the exercises
3. Anki flashcards summary of the book

## 1. Replicate all the figures

To reproduce a figure, say figure 2.2, do:

```bash
cd chapter2
python figures.py 2.2
```

### Chapter 2
1. [Figure 2.2: Average performance of epsilon-greedy action-value methods on the 10-armed testbed](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter2/plots/fig2.2.png)
2. [Figure 2.3: Optimistic initial action-value estimates](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter2/plots/fig2.3.png)
3. [Figure 2.4: Average performance of UCB action selection on the 10-armed testbed](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter2/plots/fig2.4.png)
4. [Figure 2.5: Average performance of the gradient bandit algorithm](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter2/plots/fig2.5.png)
5. [Figure 2.6: A parameter study of the various bandit algorithms](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter2/plots/fig2.6.png)

### Chapter 4
1. Figure 4.2: Jack’s car rental problem ([value function](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/fig4.2.png), [policy](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/fig4.2_policy.png))
2. Figure 4.3: The solution to the gambler’s problem ([value function](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/fig4.3.png), [policy](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/fig4.3_policy.png))

### Chapter 5
1. [Figure 5.1: Approximate state-value functions for the blackjack policy](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/fig5.1.png)
2. [Figure 5.2: The optimal policy and state-value function for blackjack found by Monte Carlo ES](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/fig5.2.png)
3. [Figure 5.3: Weighted importance sampling](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/fig5.3.png)
4. [Figure 5.4: Ordinary importance sampling with surprisingly unstable estimates](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/fig5.4.png)
5. Figure 5.5: A couple of right turns for the racetrack task ([1](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/fig5.5_left.png), [2](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/fig5.5_right_1.png), [3](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/fig5.5_right_2.png))

### Chapter 6
1. [Figure 6.1: Changes recommended in the driving home example by Monte Carlo methods (left)
and TD methods (right)](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/fig6.1.png)
2. [Example 6.2: Random walk](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/example6.2.png) ([comparison](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/example6.2_comparison.png))
3. [Figure 6.2: Performance of TD(0) and constant MC under batch training on the random walk task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/fig6.2.png)
4. [Example 6.5: Windy Gridworld](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/example6.5.png)
5. [Example 6.6: Cliff Walking](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/example6.6.png)
6. [Figure 6.3: Interim and asymptotic performance of TD control methods](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/fig6.3.png) ([comparison](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/fig6.3_comparison.png))
7. [Figure 6.5: Comparison of Q-learning and Double Q-learning](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/fig6.5.png)

### Chapter 7
1. [Figure 7.2: Performance of n-step TD methods on 19-state random walk](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter7/plots/fig7.2.png) ([comparison](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter7/plots/fig7.2_comparison.png))
2. [Figure 7.4: Gridworld example of the speedup of policy learning due to the use of n-step
methods](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter7/plots/fig7.4.png)

### Chapter 8
1. [Figure 8.2: Average learning curves for Dyna-Q agents varying in their number of planning steps](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/fig8.2.png)
2. [Figure 8.3: Policies found by planning and nonplanning Dyna-Q agents](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/fig8.3.png)
3. [Figure 8.4: Average performance of Dyna agents on a blocking task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/fig8.4.png)
4. [Figure 8.5: Average performance of Dyna agents on a shortcut task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/fig8.5.png)
5. [Example 8.4: Prioritized sweeping significantly shortens learning time on the Dyna maze task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/example8.4.png)
6. [Figure 8.7: Comparison of efficiency of expected and sample updates](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/fig8.7.png)
7. [Figure 8.8: Relative efficiency of different update distributions](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/fig8.8.png)

### Chapter 9
1. [Figure 9.1: Gradient Monte Carlo algorithm on the 1000-state random walk task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter9/plots/fig9.1.png)
2. [Figure 9.2: Semi-gradient n-steps TD algorithm on the 1000-state random walk task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter9/plots/fig9.2.png)
3. [Figure 9.5: Fourier basis vs polynomials on the 1000-state random walk task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter9/plots/fig9.5.png) ([comparison](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter9/plots/fig9.5_comparison.png))
4. [Figure 9.10: State aggregation vs. Tile coding on 1000-state random walk task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter9/plots/fig9.10.png) ([comparison](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter9/plots/fig9.10.png))

### Chapter 10
1. Figure 10.1: The cost-to-go function for Mountain Car task in one run ([428 steps](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.1_428_steps.png); [12](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.1_12_episodes.png), [104](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.1_104_episodes.png), [1000](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.1_1000_episodes.png), [9000](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.1_9000_episodes.png) episodes)
2. [Figure 10.2: Learning curves for semi-gradient Sarsa on Mountain Car task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.2.png)
3. [Figure 10.3: One-step vs multi-step performance of semi-gradient Sarsa on the Mountain Car task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.3.png)
4. [Figure 10.4: Effect of the alpha and n on early performance of n-step semi-gradient Sarsa](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.4.png)
5. [Figure 10.5: Differential semi-gradient Sarsa on the access-control queuing task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter10/plots/fig10.5.png)

### Chapter 11
1. [Figure 11.2: Demonstration of instability on Baird’s counterexample](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter11/plots/fig11.2.png)
2. [Figure 11.5: The behavior of the TDC algorithm on Baird’s counterexample](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter11/plots/fig11.5.png)
3. [Figure 11.6: The behavior of the ETD algorithm in expectation on Baird’s counterexample](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter11/plots/fig11.6.png)

### Chapter 12
1. [Figure 12.3: Off-line λ-return algorithm on 19-state random walk](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter12/plots/fig12.3.png)
2. [Figure 12.6: TD(λ) algorithm on 19-state random walk](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter12/plots/fig12.6.png)
3. [Figure 12.8: True online TD(λ) algorithm on 19-state random walk](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter12/plots/fig12.8.png)
4. [Figure 12.10: Sarsa(λ) with replacing traces on Mountain Car](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter12/plots/fig12.10.png)
5. [Figure 12.11: Summary comparison of Sarsa(λ) algorithms on Mountain Car](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter12/plots/fig12.11.png)

### Chapter 13
1. [Figure 13.1: REINFORCE on the short-corridor grid world](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter13/plots/fig13.1.png)
2. [Figure 13.2: REINFORCE with baseline on the short-corridor grid-world](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter13/plots/fig13.2.png)

## 2. Solution to all of the exercises ([text answers](https://github.com/mtrazzi/rl-book-challenge/tree/master/exercises.txt))

To reproduce the results of an exercise, say exercise 2.5 do:

```bash
cd chapter2
python figures.py ex2.5
```

### Chapter 2

1. [Exercise2.5: Difficulties that sample-average methods have for nonstationary problems](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter2/plots/ex2.5.png)

1. [Exercise2.11: Figure analogous to Figure 2.6 for the nonstationary
case](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter2/plots/ex2.11.png)

### Chapter 4

1. Exercise 4.7: Modified Jack's car rental problem ([value function](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/ex4.7.png), [policy](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/ex4.7_policy.png))

2. Exercise 4.9: Gambler’s problem with ph = 0.25 ([value function](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/ex4.9_ph_025.png), [policy](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/ex4.9_ph_025_policy.png)) and ph = 0.55 ([value function](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/ex4.9_ph_055.png), [policy](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter4/plots/ex4.9_ph_055_policy.png))

### Chapter 5

1. Exercise 5.14: Modified MC Control on the racetrack ([1](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/ex5.14_right_1.png), [2](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter5/plots/ex5.14_right_2.png))

### Chapter 6

1. [Exercise 6.4: Wider range of values alpha](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/ex6.4.png)
2. [Exercise 6.5: High alpha, 99ffect of initialization](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/ex6.5.png)
3. [Exercise 6.9: Windy Gridworld with King’s Moves](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/ex6.9.png)
4. [Exercise 6.10: Stochastic Wind](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/ex6.10.png)
5. [Exercise 6.13: Double Expected Sarsa vs. Expected Sarsa](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter6/plots/ex6.13.png)

### Chapter 7

1. [Exercise7.2: Sum of TD error vs. n-step TD on 19-states random walk](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter7/plots/ex7.2.png)
2. [Exercise7.3: 19 states vs. 5 states, left-side outcome of -1](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter7/plots/ex7.3.png)
3. [Exercise7.7: Off-policy action-value prediction on a not-so-random walk](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter7/plots/ex7.7.png)
4. [Exercise7.10: Off-policy action-value prediction on a not-so-random walk](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter7/plots/ex7.10.png)

### Chapter 8
1. [Exercise8.1: n-step sarsa on the maze task](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/ex8.1.png)
2. [Exercise8.4: Gridworld experiment to test the exploration bonus](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter8/plots/ex8.4.png)

### Chapter 11

1. [Exercise11.3: One-step semi-gradient Q-learning to Baird’s counterexample](https://raw.githubusercontent.com/mtrazzi/rl-book-challenge/master/chapter11/plots/ex11.3.png)

## 3. [Anki flashcards](https://drive.google.com/open?id=1K2B8FsxHShDDER9EXIHDrirBbXf7M2K4) (cf. [this blog](http://augmentingcognition.com/ltm.html))

## Appendix

### Dependencies

```bash
numpy
matplotlib
seaborn
```

### Credits

All of the code and answers are mine, except for mountain car's [tile coding](https://github.com/mtrazzi/rl-book-challenge/blob/master/chapter10/tiles_sutton.py) (url in the book).

This README is inspired from [ShangtongZhang's repo](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction).

### Design choices

1. All of the chapters are self-contained.
2. The environments use a gym-like API with methods:

```bash
s = env.reset()
s_p, r, d, dict = env.step(a)
```

### How long did it take

The entire thing (plots, exercises, anki cards (including reviewing)) took about 400h of focused work.
