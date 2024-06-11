import numpy as np


def rotate(vec_2d: list, angle: float | list) -> list:
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    return rot_matrix @ vec_2d