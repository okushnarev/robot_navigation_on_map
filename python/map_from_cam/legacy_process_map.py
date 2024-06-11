from pathlib import Path

import cv2 as cv
import numpy as np

from python.map_from_cam.utils import mm_to_px, draw_filled_square, undo_patch
from python.path_planning.astar import astar, manhattan_distance
from python.path_planning.maze import MazePosition, Maze, mark_path, Cell

changed_patches = []


def add_point(img, x, y):
    global changed_patches
    side = 5
    patch = draw_filled_square(img, (x, y), side)
    changed_patches.append(patch)

    print(f'Coordinates are: {x, y}')


def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        add_point(_img, x, y)


def solve_maze(_img, gain: float = 1):
    global changed_patches
    assert len(changed_patches) > 1, 'Minimum of two points on map required'

    start_pos = MazePosition(*changed_patches[-2].center)
    goal_pos = MazePosition(*changed_patches[-1].center)
    assert start_pos != goal_pos, 'Goal position cannot be the same as the start position'

    cols, rows = _img.shape[:2]
    maze = Maze(rows, cols, _img, start_pos, goal_pos)
    goal_node = astar(start_pos, maze.goal_test, maze.neighbours, manhattan_distance(goal_pos, gain))

    path = goal_node.to_path() if goal_node is not None else None

    return path


if __name__ == '__main__':
    import_dir = Path('./maps')

    name = '5_2'
    img_gray = cv.imread(import_dir / f'{name}.png', cv.IMREAD_GRAYSCALE)
    img_orig = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    print(f'Shape if the image is: {img_gray.shape}')

    # Enter robot center coordinates (in px)
    rob_x = 555
    rob_y = 205
    rob_diam_px = mm_to_px(450)

    # paint robot's position with white circle
    cv.circle(img_gray, (rob_x, rob_y), rob_diam_px // 2 + 1, 255, -1)

    # Morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (rob_diam_px, rob_diam_px))
    img_gray = cv.erode(img_gray, kernel)

    # Resize image for one pixel to fit 10mm WxH
    scale = mm_to_px(10)
    h_orig, w_orig = img_gray.shape
    h_rsz, w_rsz, rob_x, rob_y = (x // scale for x in (h_orig, w_orig, rob_x, rob_y))
    img_gray = cv.resize(img_gray, (w_rsz, h_rsz))

    print(f'Shape if the image is: {img_gray.shape}')

    # Set up window to find click coordinates
    cv.namedWindow('image')
    cv.setMouseCallback('image', mouse_callback)

    # Cast color space before show image
    _img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    _img_backup = _img.copy()

    add_point(_img, rob_x, rob_y)
    while True:
        cv.imshow('image', _img)

        k = cv.waitKey(33)
        if k == ord('q'):  # Quit
            break

        elif k == ord('z'):  # Undo (like cmd+z)
            undo_patch(_img, changed_patches.pop())

        elif k == ord('p'):  # find Path and show it on Map
            path = solve_maze(img_gray, gain=1.3)
            if path is not None:
                mark_path(_img, path)
            else:
                print('No path found, try another goal point')
                undo_patch(_img, changed_patches.pop())

        elif k == ord('o'):  # show Original image with path
            # resize image
            img_temp = cv.resize(_img, (w_orig, h_orig))

            # extract color
            color = np.array(Cell.PATH.value)
            mask = cv.inRange(img_temp, color, color)

            # thicken the line
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv.dilate(mask, kernel, iterations=3)

            # copy original image and draw path on it
            img_show = img_orig.copy()
            img_show[mask == 255] = color

            # show image to user until 'q' is pressed
            cv.imshow('original image', img_show)
            cv.waitKey(0)
            cv.destroyWindow('original image')

        elif k == ord('r'):  # Redraw
            _img = _img_backup.copy()
            changed_patches = []
            add_point(_img, rob_x, rob_y)
