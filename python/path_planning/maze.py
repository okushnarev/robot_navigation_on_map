from enum import Enum
from typing import NamedTuple


class Cell(Enum):
    EMPTY = 255
    BLOCKED = 0
    PATH = [167, 121, 78]


class MazePosition(NamedTuple):
    row: int
    col: int


class Maze:
    def __init__(self, rows=10, columns=10, map=None, start=MazePosition(0, 0), goal=MazePosition(9, 9)):
        self._rows = rows
        self._columns = columns
        self.start = start
        self.goal = goal

        self._grid = map

    def goal_test(self, m_loc):
        return m_loc == self.goal

    def neighbours(self, m_loc):
        neighbours = []
        if m_loc.row - 1 >= 0 and self._grid[m_loc.row - 1][m_loc.col] != Cell.BLOCKED.value:
            neighbours.append(MazePosition(m_loc.row - 1, m_loc.col))

        if m_loc.row + 1 < self._rows and self._grid[m_loc.row + 1][m_loc.col] != Cell.BLOCKED.value:
            neighbours.append(MazePosition(m_loc.row + 1, m_loc.col))

        if m_loc.col - 1 >= 0 and self._grid[m_loc.row][m_loc.col - 1] != Cell.BLOCKED.value:
            neighbours.append(MazePosition(m_loc.row, m_loc.col - 1))

        if m_loc.col + 1 < self._columns and self._grid[m_loc.row][m_loc.col + 1] != Cell.BLOCKED.value:
            neighbours.append(MazePosition(m_loc.row, m_loc.col + 1))

        return neighbours


def mark_path(img, path, color=Cell.PATH.value):
    for loc in path:
        img[loc.row][loc.col] = color
