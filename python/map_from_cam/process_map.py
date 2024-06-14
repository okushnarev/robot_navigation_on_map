from pathlib import Path


import cv2 as cv
import numpy as np

from python.map_from_cam.utils import mm_to_px, draw_filled_square, Waypoint, put_text_top_center
from python.path_planning.astar import astar, manhattan_distance
from python.path_planning.maze import MazePosition, Maze, mark_path, Cell



class Map:
    def __init__(self, img_path, heuristic_gain=1.0):

        self.heuristic_gain = heuristic_gain
        self.changed_patches = []  # This is about original colourful image
        self.waypoints = []  # This is about occupancy grid map

        self.img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        self.img_orig = cv.cvtColor(self.img_gray, cv.COLOR_GRAY2BGR)
        self.img_orig_backup = self.img_orig.copy()
        self.h_orig, self.w_orig = self.img_gray.shape
        self.h_rsz = self.w_rsz = None
        self.scale = 10
        print(f'Shape if the Original image is: {self.img_gray.shape}')

        # Set up window with map. Attach callback
        self.setup_window()

        # Robot shape on map
        self.rob_x = None
        self.rob_y = None
        self.rob_diam_px = None

        # Prepare image
        self.prep_image()  # map for a robot
        self.img_map = cv.cvtColor(self.img_gray, cv.COLOR_GRAY2BGR)
        self.img_map_backup = self.img_map.copy()

        # Path on a map
        self.path = []

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = mm_to_px(val)

    def prep_image(self):
        self.rob_x, self.rob_y = self.find_robot_center()

        self.rob_diam_px = self.find_robot_diameter()

        # Paint robot's position with white circle
        cv.circle(self.img_gray, (self.rob_x, self.rob_y), self.rob_diam_px // 2, 255, -1)

        # Morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.rob_diam_px, self.rob_diam_px))
        self.img_gray = cv.erode(self.img_gray, kernel)

        # Resize image for one pixel to fit scale mm WxH
        self.w_rsz = self.w_orig // self.scale
        self.h_rsz = self.h_orig // self.scale
        self.img_gray = cv.resize(self.img_gray, (self.w_rsz, self.h_rsz))

        print(f'Shape if the Resized image is: {self.img_gray.shape}')

    def find_robot_diameter(self):
        print('Mark robot\'s radius on map')
        img = put_text_top_center(self.img_orig.copy(), 'Mark robot\'s radius')
        while len(self.changed_patches) < 2:
            cv.imshow('image', img)
            if cv.waitKey(33) == ord('q'):
                raise Exception('No robot\'s radius chosen')
        rad_point = self.changed_patches[-1]
        self.undo_point(self.img_orig)
        dx = rad_point.center[0] - self.rob_x
        dy = rad_point.center[1] - self.rob_y
        return int(np.hypot(dx, dy)) * 2

    def find_robot_center(self):
        print('Mark robot\'s center on map')
        img = put_text_top_center(self.img_orig.copy(), 'Mark robot\'s center')
        while len(self.changed_patches) < 1:
            cv.imshow('image', img)
            if cv.waitKey(33) == ord('q'):
                raise Exception('No robot\'s center chosen')
        return self.changed_patches[0].center

    def setup_window(self):
        # Set up window to find click coordinates
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.add_point(x, y)

    def add_point(self, x, y, side=15):
        patch = draw_filled_square(self.img_orig, (x, y), side, color=[242, 43, 83][::-1])
        self.changed_patches.append(patch)
        self.waypoints.append(Waypoint(x // self.scale, y // self._scale))

        print(f'Coordinates are: {x, y}')

    def undo_point(self, img):
        """
        Places the original patch back onto the image at the specified center.

        :param img: The image on which to place the patch.
        """
        self.waypoints.pop()
        patch = self.changed_patches.pop()

        half_side = patch.container.shape[0] // 2
        top_left = (patch.center[0] - half_side, patch.center[1] - half_side)
        bottom_right = (patch.center[0] + half_side, patch.center[1] + half_side)

        # Place the patch back onto the image
        img[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = patch.container

    def solve_maze(self, img, gain: float = 1):
        assert len(self.waypoints) > 1, 'Minimum of two points on map required'

        start_pos = MazePosition(self.waypoints[-2].y, self.waypoints[-2].x)
        goal_pos = MazePosition(self.waypoints[-1].y, self.waypoints[-1].x)
        assert start_pos != goal_pos, 'Goal position cannot be the same as the start position'

        rows, cols = img.shape[:2]
        maze = Maze(rows, cols, img, start_pos, goal_pos)
        goal_node = astar(start_pos, maze.goal_test, maze.neighbours, manhattan_distance(goal_pos, gain))

        path = goal_node.to_path() if goal_node is not None else None

        return path

    def find_path(self):
        return self.solve_maze(self.img_gray, gain=self.heuristic_gain)

    def show_path(self, path):
        if path is not None:
            mark_path(self.img_map, path)
            self.path.extend(path)
        else:
            print('No path found, try another goal point')
            self.undo_point(self.img_orig)

    def redraw_map(self):
        # Restore images
        self.img_map = self.img_map_backup.copy()
        self.img_orig = self.img_orig_backup.copy()

        # Restore points
        self.changed_patches = []
        self.path = []

        # Mark robot's center
        self.add_point(self.rob_x, self.rob_y)

    def show_original_image(self, win_name='original image'):

        # Resize the BGR image to match the dimensions of the grayscale image
        temp_img = cv.resize(self.img_map, (self.w_orig, self.h_orig), interpolation=cv.INTER_NEAREST)

        # Find and draw contours
        temp_img_thresh = 255 - cv.inRange(temp_img, (0, 0, 0), (60, 60, 60))
        contours, _ = cv.findContours(temp_img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(self.img_orig, contours, -1, (2, 200, 2), 2)

        # Extract path from map
        path_color = np.array(Cell.PATH.value)
        mask = cv.inRange(temp_img, (1,1,1), (254,254,254))
        overlay_image = np.where(mask[:, :, None] != 0, temp_img, self.img_orig)

        # show image to user until key is pressed
        cv.imshow(win_name, overlay_image)

        return overlay_image

    def run(self):
        while True:
            self.show_original_image('image')

            k = cv.waitKey(33)
            if k == ord('q'):  # Quit
                break

            elif k == ord('z'):  # Undo (like cmd+z)
                self.undo_point(self.img_orig)

            elif k == ord('p'):  # find Path and show it on Map
                path = self.find_path()
                self.show_path(path)

            elif k == ord('r'):  # Redraw
                self.redraw_map()

            elif k == ord('e'):  # Export
                cv.destroyAllWindows()
                return self.path

        cv.destroyAllWindows()


if __name__ == '__main__':
    import_dir = Path('./maps')
    name = '7_2'
    import_path = import_dir / f'{name}.png'

    map_1 = Map(img_path=import_path,
                heuristic_gain=1)

    path = map_1.run()
    print(path)
