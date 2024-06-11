from pathlib import Path

import cv2 as cv
import numpy as np

from python.map_from_cam.utils import mm_to_px, draw_filled_square
from python.path_planning.astar import astar, manhattan_distance
from python.path_planning.maze import MazePosition, Maze, mark_path, Cell


class Map:
    def __init__(self, img_path, rob_x, rob_y, heuristic_gain=1.0):
        # Enter robot center coordinates (in px)
        self.rob_x = rob_x
        self.rob_y = rob_y
        self.rob_diam_px = mm_to_px(450)

        self.heuristic_gain = heuristic_gain
        self.changed_patches = []

        self.img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        self.img_orig = cv.cvtColor(self.img_gray, cv.COLOR_GRAY2BGR)
        self.h_orig, self.w_orig = self.img_gray.shape
        self.h_rsz = self.w_rsz = None
        print(f'Shape if the Original image is: {self.img_gray.shape}')

        # Prepare image
        self.prep_image()  # map for a robot
        self.img_map = cv.cvtColor(self.img_gray, cv.COLOR_GRAY2BGR)
        self.img_map_backup = self.img_map.copy()

        # Paint robot's position with white circle
        cv.circle(self.img_gray, (rob_x, rob_y), self.rob_diam_px // 2 + 1, 255, -1)

        # # Add robot's center as start point
        # self.add_point(self._img, self.rob_x, self.rob_y)

        # Set up window with map. Attach callback
        self.setup_window()

        # Path on a map
        self.path = []

    def prep_image(self):
        # Morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.rob_diam_px, self.rob_diam_px))
        self.img_gray = cv.erode(self.img_gray, kernel)

        # Resize image for one pixel to fit 10mm WxH
        scale = mm_to_px(10)
        self.h_rsz, self.w_rsz, self.rob_x, self.rob_y = (x // scale for x in
                                                          (self.h_orig, self.w_orig, self.rob_x, self.rob_y))
        self.img_gray = cv.resize(self.img_gray, (self.w_rsz, self.h_rsz))

        print(f'Shape if the Resized image is: {self.img_gray.shape}')

    def setup_window(self):
        # Set up window to find click coordinates
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.add_point(self.img_map, x, y)

    def add_point(self, img, x, y):
        side = 5
        patch = draw_filled_square(img, (x, y), side)
        self.changed_patches.append(patch)

        print(f'Coordinates are: {x, y}')

    def undo_point(self, img, patch):
        """
        Places the original patch back onto the image at the specified center.

        :param img: The image on which to place the patch.
        :param patch: An instance of the Patch class containing the center and the original patch.
        """
        half_side = patch.container.shape[0] // 2
        top_left = (patch.center[0] - half_side, patch.center[1] - half_side)
        bottom_right = (patch.center[0] + half_side, patch.center[1] + half_side)

        # Place the patch back onto the image
        img[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = patch.container

    def solve_maze(self, img, gain: float = 1):
        assert len(self.changed_patches) > 1, 'Minimum of two points on map required'

        start_pos = MazePosition(*self.changed_patches[-2].center[::-1])
        goal_pos = MazePosition(*self.changed_patches[-1].center[::-1])
        assert start_pos != goal_pos, 'Goal position cannot be the same as the start position'

        rows, cols = img.shape[:2]
        maze = Maze(rows, cols, img, start_pos, goal_pos)
        goal_node = astar(start_pos, maze.goal_test, maze.neighbours, manhattan_distance(goal_pos, gain))

        path = goal_node.to_path() if goal_node is not None else None

        return path

    def run(self):
        while True:
            cv.imshow('image', self.img_map)

            k = cv.waitKey(33)
            if k == ord('q'):  # Quit
                break

            elif k == ord('z'):  # Undo (like cmd+z)
                self.undo_point(self.img_map, self.changed_patches.pop())

            elif k == ord('p'):  # find Path and show it on Map
                self.find_and_show_path()

            elif k == ord('o'):  # show Original image with path
                self.show_original_image()

            elif k == ord('r'):  # Redraw
                self.redraw_map()

            elif k == ord('e'):  # Export
                cv.destroyAllWindows()
                return self.path

    def find_and_show_path(self):
        _path = self.solve_maze(self.img_gray, gain=self.heuristic_gain)
        if _path is not None:
            mark_path(self.img_map, _path)
            self.path.extend(_path)
        else:
            print('No path found, try another goal point')
            self.undo_point(self.img_map, self.changed_patches.pop())

    def redraw_map(self):
        self.img_map = self.img_map_backup.copy()
        self.changed_patches = []
        self.path = []
        self.add_point(self.img_map, self.rob_x, self.rob_y)

    def show_original_image(self):


        # Resize the BGR image to match the dimensions of the grayscale image
        temp_img = cv.resize(self.img_map, (self.w_orig, self.h_orig), interpolation=cv.INTER_NEAREST)

        # Find and draw contours
        temp_img_thresh =cv.inRange(temp_img, (0,0,0), (60,60,60))
        contours, _ = cv.findContours(temp_img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(temp_img, contours, -1, (2, 2, 255), 1)

        # Overlay the resized BGR image onto the converted grayscale image
        # Here we assume the BGR image has some transparency or mask, if not, we can create a mask
        mask = cv.inRange(temp_img, (1, 1, 1), (255, 255, 255))
        overlay_image = np.where(mask[:, :, None] != 0, temp_img, self.img_orig)

        # show image to user until key is pressed
        cv.imshow('original image', overlay_image)
        cv.waitKey(0)
        cv.destroyWindow('original image')

        return overlay_image


if __name__ == '__main__':
    import_dir = Path('./maps')
    name = '5_2'
    import_path = import_dir / f'{name}.png'

    map_1 = Map(img_path=import_path,
                rob_x=555,
                rob_y=205,
                heuristic_gain=1)

    path = map_1.run()
    print(path)
