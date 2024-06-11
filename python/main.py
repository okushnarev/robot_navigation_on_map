import time
from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from python.map_from_cam.process_map import Map
from python.path_planning.maze import MazePosition, mark_path
from python.robot_control.path_tracking import PurePursuit
from python.robot_control.robotino_api import Robotino4
from python.robot_control.utils import rotate
from python.utils import init_matplotlib, to_grid

if __name__ == '__main__':

    import_dir = Path('./map_from_cam/maps')

    export_dir = Path('./runs')
    export_dir.mkdir(exist_ok=True, parents=True)

    # Scan all runs and find current run id
    prev_runs = (int(d.name) for d in export_dir.iterdir())
    try:
        run_id = max(prev_runs) + 1
    except ValueError:
        run_id = 1

    export_dir /= str(run_id)
    export_dir.mkdir(exist_ok=True, parents=True)

    map_name = '5_2'
    map_path = import_dir / f'{map_name}.png'

    map_1 = Map(img_path=map_path,
                rob_x=555,
                rob_y=205,
                heuristic_gain=1)

    path_nodes = map_1.run()

    # Define the path as a list of points (x, y)
    cell_to_m = 10 / 1000
    path = np.array([(node.row * cell_to_m, node.col * cell_to_m) for node in path_nodes])
    # Convert to local coordinates
    shift_path = path[0].copy()
    path -= shift_path

    # Path tracking params for controller
    look_ahead_distance = 0.05  # Define look-ahead distance (in m)
    rob_speed = 0.05  # Define robot speed (in m/s)

    # Initialize the Pure Pursuit controller
    pp = PurePursuit(path=path,
                     look_ahead_distance=look_ahead_distance,
                     constant_speed=rob_speed)

    # Initial robot state
    current_position = list(path[0])
    pose_history = []

    # Initialize robot controller
    robot = Robotino4()

    # Calc shift to find odometry in local coordinates
    odometry = robot.get_odometry()
    shift_odometry = np.array([odometry[0], odometry[1]])
    rot_odometry = odometry[2]


    Vx = 0
    Vy = 0

    # Timer
    period_s = 0.25
    current_time = time.time()
    while pp.current_index < len(path) - 1:
        if time.time() - current_time < period_s:
            print(f'Time diff: {time.time()-current_time}')
            # Update timer
            current_time = time.time()
            # Update pose history
            pose_history.append(current_position.copy())

            # Calculate control commands
            Vx, Vy = pp.calculate_control(current_position)

            robot.set_omnidrive(vx=Vx, vy=Vy)

            odometry = np.array(robot.get_odometry()[:2])
            odometry -= shift_odometry
            odometry = rotate(odometry, -rot_odometry)

            # Update the robot's state
            current_position[0] = odometry[0]
            current_position[1] = odometry[1]

            # Debug info
            print(f'Idx: {pp.current_index}/{len(path)-1}')
            print(f"Position: {current_position}, Vx: {Vx}, Vy: {Vy}")

    dist = 1
    while dist > 0.001:
        dx = path[-1][0] - current_position[0]
        dy = path[-1][1] - current_position[1]
        dist = np.hypot(dx, dy)
        print(dist)

        robot.set_omnidrive(vx=Vx, vy=Vy)
        odometry = np.array(robot.get_odometry()[:2])
        odometry -= shift_odometry
        odometry = rotate(odometry, -rot_odometry)

        # Update the robot's state
        current_position[0] = odometry[0]
        current_position[1] = odometry[1]

    print("Path following complete.")

    # Path plots
    init_matplotlib()

    x, y = zip(*path)
    plt.plot(y, x, label='Desired')

    x, y = zip(*pose_history)
    plt.plot(y, x, label='Real')

    plt.tight_layout()
    plt.xlabel('Y, m')
    plt.ylabel('X, m')
    plt.legend()
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.savefig(export_dir / 'plots.pdf')

    # Map with paths
    pose_history = np.array(pose_history) + shift_path  # Convert to global coordinates
    _conv = to_grid(1/cell_to_m)
    history_nodes = [MazePosition(_conv(p[0]), _conv(p[1])) for p in pose_history]
    mark_path(map_1.img_map, history_nodes, [219, 39, 153])
    img = map_1.show_original_image()
    cv.imwrite(export_dir / 'map.png', img)
