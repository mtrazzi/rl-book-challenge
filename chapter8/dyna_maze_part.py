from dyna_maze import DynaMaze, INIT_POS, GOAL_POS_L, GRID_SHAPE, WALLS

class DynaMazePartitioned(DynaMaze):
  def __init__(self, n_part, init_pos=INIT_POS, goal_pos_l=GOAL_POS_L, grid_shape=GRID_SHAPE, walls1=WALLS, walls2=WALLS):
    self.n_part = n_part
    self.row_part = (n_part + 1) // 2
    self.col_part = n_part // 2
    self.row_mul = 2 ** self.row_part
    self.col_mul = 2 ** self.col_part
    init_pos = self.expand(init_pos)
    grid_shape = self.expand(grid_shape)
    walls1, walls2, goal_pos_l  = [self.part_l(l) for l in [walls1, walls2, goal_pos_l]]
    super().__init__(init_pos, goal_pos_l, grid_shape, walls1, walls2)

  def expand(self, pos):
    return pos[0] * self.row_mul, pos[1] * self.col_mul

  def part(self, pos):
    x_0, y_0 = self.expand(pos)
    return [(_x + x_0, _y + y_0) for _x in range(self.row_mul) for _y in range(self.col_mul)]

  def part_l(self, pos_list):
    return sum([self.part(pos) for pos in pos_list], [])
