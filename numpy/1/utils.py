import os

from agents import RandomAgent, RLAgent
from board import TicTacToeBoard
import matplotlib.pyplot as plt
import numpy as np


def play_against_agent(agent):
  board = TicTacToeBoard()
  while True:
    board.reset()
    while not board.is_end_state():
      os.system('cls' if os.name == 'nt' else 'clear')
      print(board)
      x, y = [int(elt) for elt in input().split()]
      board.do_move(x, y)
      print(board)
      print(agent.get_possible_move_values(board))
      input("...")
      board.do_move(*agent.best_move(board))


def test_agent(agent=RLAgent(), opponent=RandomAgent(), n_episodes=10000):
  board = TicTacToeBoard()
  n_wins = 0
  for _ in range(n_episodes):
    board.reset()
    while not board.is_end_state():
      board.do_move(*opponent.best_move(board))
      if not board.is_end_state():
        board.do_move(*agent.best_move(board))
    n_wins += board.has_won(agent.sym)
  print(f"{n_wins}/{n_episodes}")
  return n_wins / n_episodes


def weighted_averages(arr, alpha=0.99):
  new_arr = np.zeros_like(arr)
  avg = None
  for i in range(len(arr)):
    avg = alpha * avg + (1 - alpha) * arr[i] if avg is not None else arr[i]
    new_arr[i] = avg
  return new_arr


def benchmark(agent=RLAgent(), opponent=RandomAgent('x'), step=10,
              training_steps=100, n_eval_episodes=1000, alpha=0.99):
  results = []
  for _ in range(training_steps // step):
    results.append(test_agent(agent, opponent, n_eval_episodes))
    agent.train(opponent, step)
  plt.plot(weighted_averages(results, alpha))
  plt.show()
