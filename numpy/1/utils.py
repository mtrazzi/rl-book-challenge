import os


def play_against_agent(board, agent):
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
