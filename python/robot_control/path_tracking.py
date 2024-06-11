from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python.map_from_cam.process_map import Map
from python.path_planning.maze import MazePosition, mark_path
from python.robot_control.utils import rotate


class PurePursuit:
    def __init__(self, path, look_ahead_distance, constant_speed):
        self.path = path
        self.look_ahead_distance = look_ahead_distance
        self.constant_speed = constant_speed
        self.current_index = 0

    def get_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def find_look_ahead_point(self, current_position):
        while self.current_index < len(self.path) - 1:
            segment_start = self.path[self.current_index]
            if list(segment_start) == current_position:
                self.current_index += 1
                continue
            segment_end = self.path[self.current_index + 1]
            segment_vector = np.array(segment_end) - np.array(segment_start)
            robot_vector = np.array(current_position) - np.array(segment_start)
            segment_length = self.get_distance(segment_start, segment_end)
            projection_length = np.dot(robot_vector, segment_vector) / segment_length
            print(projection_length)
            if projection_length < 0:
                look_ahead_point = segment_start
            elif projection_length > segment_length:
                self.current_index += 1
                continue
            else:
                look_ahead_point = segment_start + projection_length * segment_vector / segment_length

            if self.get_distance(current_position, look_ahead_point) >= self.look_ahead_distance:
                return look_ahead_point
            else:
                self.current_index += 1

        return self.path[-1]

    def calculate_control(self, current_position):
        look_ahead_point = self.find_look_ahead_point(current_position)
        dx = look_ahead_point[0] - current_position[0]
        dy = look_ahead_point[1] - current_position[1]

        distance_to_point = np.hypot(dx, dy)

        if distance_to_point < 1e-6:
            Vx, Vy = 0.0, 0.0
        else:
            Vx = dx / distance_to_point * self.constant_speed
            Vy = dy / distance_to_point * self.constant_speed

        return Vx, Vy


# Example usage
if __name__ == "__main__":

    import_dir = Path('../map_from_cam/maps')
    name = '5_2'
    import_path = import_dir / f'{name}.png'

    map_1 = Map(img_path=import_path,
                rob_x=555,
                rob_y=205,
                heuristic_gain=1)

    path_nodes = map_1.run()

    # Define the path as a list of points (x, y)
    cell_to_m = 10 / 1000
    path = np.array([(node.row * cell_to_m, node.col * cell_to_m) for node in path_nodes])
    shift = path[0].copy()
    path -= shift
    look_ahead_distance = 0.05  # Define look-ahead distance (in m)

    # Initialize the Pure Pursuit controller
    pp = PurePursuit(path, look_ahead_distance, 0.1)

    # Initial robot state
    current_position = list(path[0])
    pose_history = []

    while pp.current_index < len(path) - 1:
        # Update pose history
        pose_history.append(current_position.copy())

        # Calculate control commands
        Vx, Vy = pp.calculate_control(current_position)

        # Update the robot's state (this would be replaced with actual robot control code)
        current_position[0] += Vx * 0.1  # Update position with a timestep of 0.1 seconds
        current_position[1] += Vy * 0.1

        # Debug info
        """
        print(f'Idx: {pp.current_index}')
        print(f"Position: {current_position}, Vx: {Vx}, Vy: {Vy}")
        """

    print("Path following complete.")

    x, y = zip(*path)
    plt.plot(y, x, label='Desired')

    x, y = zip(*pose_history)
    plt.plot(y, x, label='Real')

    plt.legend()
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()

    pose_history = np.array(pose_history)
    pose_history += shift
    history_nodes = [MazePosition(int(p[0] / cell_to_m), int(p[1] / cell_to_m)) for p in pose_history]
    mark_path(map_1.img_map, history_nodes, [219, 39, 153])
    map_1.show_original_image()
