import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import numpy as np
import heapq
import math
import random
import yaml
import scipy.interpolate as si
import sys
import threading
import time
import signal
import subprocess

from rclpy.qos import qos_profile_sensor_data

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

with open("src/turtlebot3-ros2-autonomous-frontier-based-exploration/autonomous_exploration/config/params.yaml", 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

param_lookahead_distance = params["lookahead_distance"]
param_speed = params["speed"]
param_expansion_size = params["expansion_size"]
param_target_error = params["target_error"]
param_min_distance_to_obstacles = params["min_distance_to_obstacles"]

# region DataTransformation
# Calculate euler yaw angle from x, y, z, w


def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion (x, y, z, w).
    """
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - \
        math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + \
        math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - \
        math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + \
        math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    return (qx, qy, qz, qw)

# endregion DataTransformation

# region MapPreparation


def prepare_map_and_get_groups(map_data, row, column):
    # Expand the detected walls/borders
    data = costmap(map_data.data, map_data.info.width,
                   map_data.info.height, map_data.info.resolution)

    data[row][column] = 0  # Robot Current Position
    data[data > 5] = 1  # 0 is navigable, 100 is a definite obstacle

    data = mark_frontiers(data)  # Find boundary points

    groups = assign_groups(data)  # Group boundary points

    # Sort the groups from smallest to largest. Take the 5 largest groups
    groups = sort_groups(groups)

    # -0.05 is unknown. Mark it as not navigable. 0 = navigable, 1 = not navigable.
    data[data < 0] = 1

    return data, groups


def costmap(data, width, height, resolution):
    # Reshape 1D data array into 2D array, based on height and width of map
    data = np.array(data).reshape(height, width)
    # Get the positions of walls in the map (x-value and y-value)
    wall = np.where(data == 100)
    # Loop over the range defined by the expansion size to expand the walls
    for i in range(-param_expansion_size, param_expansion_size+1):
        for j in range(-param_expansion_size, param_expansion_size+1):
            # Skip the center position (the original wall position)
            if i == 0 and j == 0:
                continue
            # Calculate the new positions to expand to
            x = wall[0]+i
            y = wall[1]+j
            # Ensure the new positions are within the bounds of the map
            x = np.clip(x, 0, height-1)
            y = np.clip(y, 0, width-1)
            # Set the new positions to 100 (indicating expanded walls)
            data[x, y] = 100
    # Scale the data array by the resolution value
    data = data*resolution
    return data


def mark_frontiers(matrix):
    # Search for frontier areas between already explored and navigable (value 0) and not yet explored (value < 0) areas on the map. Mark the pixels in this area with the value 2.
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0.0:
                if i > 0 and matrix[i-1][j] < 0:
                    matrix[i][j] = 2
                elif i < len(matrix)-1 and matrix[i+1][j] < 0:
                    matrix[i][j] = 2
                elif j > 0 and matrix[i][j-1] < 0:
                    matrix[i][j] = 2
                elif j < len(matrix[i])-1 and matrix[i][j+1] < 0:
                    matrix[i][j] = 2
    return matrix

# depth-first-search, search for groups (Adjacent pixels with the same value)


def dfs(matrix, i, j, group, groups):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
        return group
    if matrix[i][j] != 2:
        return group
    if group in groups:
        groups[group].append((i, j))
    else:
        groups[group] = [(i, j)]
    matrix[i][j] = 0
    dfs(matrix, i + 1, j, group, groups)
    dfs(matrix, i - 1, j, group, groups)
    dfs(matrix, i, j + 1, group, groups)
    dfs(matrix, i, j - 1, group, groups)
    dfs(matrix, i + 1, j + 1, group, groups)  # lower right cross
    dfs(matrix, i - 1, j - 1, group, groups)  # upper left cross
    dfs(matrix, i - 1, j + 1, group, groups)  # upper right cross
    dfs(matrix, i + 1, j - 1, group, groups)  # lower left cross
    return group + 1


def assign_groups(matrix):
    group = 1
    groups = {}  # Dictionary for group keys and coordinates
    # Iterate through each position in the map
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # Check if current cell is marked as frontier
            if matrix[i][j] == 2:
                group = dfs(matrix, i, j, group, groups)
    return groups


def sort_groups(groups):
    sorted_groups = sorted(
        groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = [g for g in sorted_groups[:5] if len(g[1]) > 2]
    return top_five_groups

# endregion MapPreparation

# region CalculateWhereToGo


def calculate_centroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid


def euclidean_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)


def get_path_length(path):
    path_length = 0.0
    for i in range(1, len(path.poses)):
        position1 = path.poses[i-1].pose.position
        position2 = path.poses[i].pose.position
        path_length += euclidean_distance(position1, position2)
    return path_length


def get_nav_path(nav, init_pose_points, goal_pose_points):
    init_pose = PoseStamped()
    init_pose.header.frame_id = 'map'
    init_pose.header.stamp = nav.get_clock().now().to_msg()
    init_pose.pose.position.x = init_pose_points[0]
    init_pose.pose.position.y = init_pose_points[1]
    init_pose.pose.position.z = init_pose_points[2]

    init_pose.pose.orientation.x = init_pose_points[3]
    init_pose.pose.orientation.y = init_pose_points[4]
    init_pose.pose.orientation.z = init_pose_points[5]
    init_pose.pose.orientation.w = init_pose_points[6]

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = nav.get_clock().now().to_msg()
    goal_pose.pose.position.x = goal_pose_points[0]
    goal_pose.pose.position.y = goal_pose_points[1]
    goal_pose.pose.position.z = 0.0
    goal_pose.pose.orientation.w = 1.0

    path = nav.getPath(init_pose, goal_pose)
    retries = 0
    while nav.getResult() == TaskResult.FAILED and retries < 5:
        path = nav.getPath(init_pose, goal_pose)
        time.sleep(0.1)
        print("Try again...")
        retries += 1

    return path, goal_pose


def find_closest_group(self, matrix, groups, resolution, originX, originY, odomX, odomY, odomZ, odom_or_x, odom_or_y, odom_or_z, odom_or_w):
    target_path = None
    target_point = None
    distances = []
    paths = []
    score = []
    target_points = []
    max_score_index = -1  # max score index
    for i in range(len(groups)):
        middle = calculate_centroid([p[0] for p in groups[i][1]], [
                                    p[1] for p in groups[i][1]])

        publish_middle_x = middle[1]*resolution+originX
        publish_middle_y = middle[0]*resolution+originY
        # For debugging
        # publish_centroid_point(self, (publish_middle_x, publish_middle_y))

        # Calculate path to centroid/middle of the group
        path, goal_pose = get_nav_path(self.nav, (odomX, odomY, odomZ, odom_or_x, odom_or_y, odom_or_z, odom_or_w),
                                       (publish_middle_x, publish_middle_y))
        if path != None:
            total_distance = get_path_length(path)
            distances.append(total_distance)

            target_points.append(goal_pose)
            paths.append(path)
    # score calculated paths to centroids
    for i in range(len(distances)):
        if distances[i] == 0:
            score.append(10000)
        else:
            points_in_group = len(groups[i][1])
            score.append(distances[i])
    # select path with the best score
    for i in range(len(distances)):
        if distances[i] > param_target_error*3:
            if max_score_index == -1 or score[i] < score[max_score_index]:
                max_score_index = i
    if max_score_index != -1:
        target_path = paths[max_score_index]
        target_point = target_points[max_score_index]
    else:
        print("Choose random target point")
        index = random.randint(0, len(groups)-1)
        target_group = groups[index][1]
        target_point = target_group[random.randint(0, len(target_group)-1)]
        target_point_x = target_point[1]*resolution+originX
        target_point_y = target_point[0]*resolution+originY
        target_path, target_point = get_nav_path(
            self.nav, (odomX, odomY, odomZ, odom_or_x, odom_or_y, odom_or_z, odom_or_w), (target_point_x, target_point_y))
    return target_path, max_score_index, target_point

# endregion CalculateWhereToGo

# region Navigation

# Set initial pose


def set_initial_pose(self, odom_x, odom_y, odom_z, odom_or_x, odom_or_y, odom_or_z, odom_or_w):
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = self.get_clock().now().to_msg()
    initial_pose.pose.position.x = odom_x
    initial_pose.pose.position.y = odom_y
    initial_pose.pose.position.z = odom_z

    initial_pose.pose.orientation.w = odom_or_x
    initial_pose.pose.orientation.w = odom_or_y
    initial_pose.pose.orientation.w = odom_or_z
    initial_pose.pose.orientation.w = odom_or_w

    self.nav.setInitialPose(initial_pose)

    return initial_pose


# Calculate the steering angle required to follow the path

def pure_pursuit(current_x, current_y, current_heading, path, index, lookahead_distance=param_lookahead_distance):
    closest_point = None
    v = param_speed
    for i in range(index, len(path.poses)):
        x = path.poses[i].pose.position.x
        y = path.poses[i].pose.position.y
        distance = math.hypot(current_x - x, current_y - y)
        if distance > lookahead_distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        target_heading = math.atan2(
            closest_point[1] - current_y, closest_point[0] - current_x)
        desired_steering_angle = target_heading - current_heading
    else:
        target_heading = math.atan2(
            path.poses[-1].pose.position.y - current_y, path.poses[-1].pose.position.x - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path.poses)-1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        v = 0.0
    return v, desired_steering_angle, index, target_heading


def prepare_obstacle_handling(self):
    publish_cmd_vel(self, 0.0, 0.0)
    # Wait until roboter stops moving
    while (self.odom_lin_vel > 0):
        print("Wait until roboter stops moving...")
        time.sleep(0.25)
        continue


def move_backwards(self, distance, speed):
    prepare_obstacle_handling(self)

    initial_x = self.odom_x
    initial_y = self.odom_y

    # Get the current time
    total_seconds = time.time()
    start_time = total_seconds
    distance_moved = 0

    publish_cmd_vel(self, -speed, 0.0)

    while (total_seconds < start_time + 20) and (distance_moved < distance):

        total_seconds = time.time()

        dx = self.odom_x - initial_x
        dy = self.odom_y - initial_y
        distance_moved = (dx**2 + dy**2)**0.5

        time.sleep(0.01)

    prepare_obstacle_handling(self)


def rotate(self, target_orientation_deg):
    prepare_obstacle_handling(self)

    # Zielorientierung in rad relativ zur aktuellen Orientierung
    target_orientation_rad = math.radians(target_orientation_deg)
    target_yaw = self.odom_yaw + target_orientation_rad

    angular_speed = 0.75
    while True:
        error = target_yaw - self.odom_yaw
        # Fehler in den Bereich [-pi, pi] normalisieren
        error = math.atan2(math.sin(error), math.cos(error))

        if abs(error) < 0.01:  # Toleranzschwelle für das Erreichen des Zielwinkels
            break

        direction = 1 if error > 0 else -1
        z = direction * angular_speed
        publish_cmd_vel(self, 0.0, z)

        # Spin einmal, um die aktuelle Orientierung zu aktualisieren
        # rclpy.spin_once(self)

    prepare_obstacle_handling(self)


def handle_obstacles(self):
    obstacle_detected = False
    # If the robot detects an obstacle in its path, it reverses and changes its orientation until it is out of the obstacle's area.
    # To ensure that a new path can be planned after the robot has moved out of the obstacle area,
    # the "required distance to wall" depends on the "expansion-size".
    if self.scan_forward_distance < param_min_distance_to_obstacles:
        print("Obstacle in front detected, turn around.")
        move_backwards(self, 0.1, 0.05)

        rotate(self, 90)
        print("Turned around.")
        obstacle_detected = True
    return obstacle_detected

# endregion Navigation

# region RosDebuggingTopics


def publish_cmd_vel(self, v, z):
    twist = Twist()
    twist.linear.x = v
    twist.angular.z = z

    self.publisher.publish(twist)


def publish_groups(self, data, groups, width, height, resolution, originX, originY):
    map_msg = OccupancyGrid()
    map_msg.header = Header()
    map_msg.header.stamp = self.get_clock().now().to_msg()
    map_msg.header.frame_id = "map"

    map_msg.info.resolution = resolution
    map_msg.info.width = width
    map_msg.info.height = height
    map_msg.info.origin.position.x = originX
    map_msg.info.origin.position.y = originY
    map_msg.info.origin.position.z = 0.0
    map_msg.info.origin.orientation.w = 1.0

    send_data = data * 0
    for i in range(len(groups)):
        values = groups[i][1]
        color = 100 - 20 * i
        for a in range(len(values)):
            send_data[values[a][0]][values[a][1]] = color

    map_msg.data = send_data.astype(int).flatten().tolist()
    self.publisher_map.publish(map_msg)


def publish_path(self, path):
    pub_path = Path()
    pub_path.header = Header()
    pub_path.header.stamp = self.get_clock().now().to_msg()
    pub_path.header.frame_id = "map"
    pub_path.poses = path.poses

    self.publisher_path.publish(pub_path)


def publish_target_point(self, target_point):
    point = PointStamped()
    point.header.stamp = self.get_clock().now().to_msg()
    point.header.frame_id = "map"
    point.point.x = target_point.pose.position.x
    point.point.y = target_point.pose.position.y
    point.point.z = 0.0
    self.publisher_point.publish(point)


def publish_centroid_point(self, middle):
    point = PointStamped()
    point.header.stamp = self.get_clock().now().to_msg()
    point.header.frame_id = "map"
    point.point.x = middle[0]
    point.point.y = middle[1]
    point.point.z = 0.0
    self.publisher_centroid.publish(point)


def publish_orientation(self, orientation_yaw):
    pose = PoseStamped()
    pose.header.stamp = self.get_clock().now().to_msg()
    pose.header.frame_id = 'map'
    quaternion = euler_to_quaternion(0, 0, orientation_yaw)
    pose.pose.position.x = self.odom_x
    pose.pose.position.y = self.odom_y
    pose.pose.position.z = self.odom_z
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]
    self.orientation_publisher_.publish(pose)

# endregion RosDebuggingTopics


class navigationControl(Node):
    def __init__(self):
        super().__init__('Navigation')
        self.nav = BasicNavigator()


class explorationControl(Node):
    def __init__(self):
        super().__init__('Exploration')
        self.subscription = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)
        self.subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.publisher_point = self.create_publisher(
            PointStamped, '/point_topic', 10)
        self.publisher_map = self.create_publisher(
            OccupancyGrid, '/group_map_topic', 10)
        self.publisher_path = self.create_publisher(Path, '/path_topic', 10)
        self.publisher_centroid = self.create_publisher(
            PointStamped, '/centroid_topic', 10)
        self.orientation_publisher_ = self.create_publisher(
            PoseStamped, 'turtlebot_orientation', 10)

        self.nav = None

        self.go_to_initial_pose = False

        print("Initialization done. Start Thread")
        # Needs thread because of While True and rclpy.spin in main
        t = threading.Thread(target=self.run_exploration)
        t.daemon = True  # End thread if main thread is stopped
        t.start()  # Runs the exploration function as a thread.

    def set_nav_node(self, nav_node):
        self.nav = nav_node.nav

    def run_exploration(self):
        # Init
        running_state = 0
        retries = 0
        while True:
            # Wait for data
            if running_state == 0:
                # Wait until data from subscribed topics is available
                if not hasattr(self, 'map_msg') or not hasattr(self, 'odom_msg') or not hasattr(self, 'scan_msg'):
                    print("Wait for data...")
                    time.sleep(0.5)
                    continue
                else:
                    # set initial pose to use nav2 package
                    initial_pose = set_initial_pose(
                        self, self.odom_x, self.odom_y, self.odom_z, self.odom_orientation_x, self.odom_orientation_y, self.odom_orientation_z, self.odom_orientation_w)

                    running_state = 1
            # Prepare map
            elif running_state == 1:
                # Data received, start exploration

                # Wait until roboter stops moving
                while (self.odom_lin_vel != 0):
                    print("Wait until roboter stops moving...")
                    time.sleep(0.25)
                    continue

                # current position on map
                row = int((self.odom_y - self.map_originY)/self.map_resolution)
                column = int((self.odom_x - self.map_originX) /
                             self.map_resolution)

                data, groups = prepare_map_and_get_groups(
                    self.map_msg, row, column)

                publish_groups(self, data, groups, self.map_width, self.map_height,
                               self.map_resolution, self.map_originX, self.map_originY)

                if len(groups) > 0:
                    running_state = 2
                else:
                    # no groups left, end exploration
                    print("No groups with more than 2 points found.")
                    self.go_to_initial_pose = True
                    running_state = 5
            # Choose target, plan path
            elif running_state == 2:
                if retries <= 5:
                    # choose group and calculate path
                    # Find the nearest group
                    print("Search for next target...")
                    path, index, target_point = find_closest_group(
                        self, data, groups, self.map_resolution, self.map_originX, self.map_originY, self.odom_x, self.odom_y, self.odom_z, self.odom_orientation_x, self.odom_orientation_y, self.odom_orientation_z, self.odom_orientation_w)
                    if path != None:
                        running_state = 3
                    else:
                        print("No path found.")
                        retries += 1
                        running_state = 1
                        time.sleep(0.5)
                else:
                    print("Too much retries...")
                    running_state = 6
            # Plan path
            elif running_state == 3:
                # Path calculated, smooth it with bspline planner
                print("Navigate to path...")
                publish_target_point(self, target_point)

                path = self.nav.smoothPath(path)

                publish_path(self, path)

                # Debugging proposes
                time.sleep(1)

                self.i = 0
                running_state = 4
            # Navigate to target
            elif running_state == 4:
                v, w, self.i, target_heading = pure_pursuit(
                    self.odom_x, self.odom_y, self.odom_yaw, path, self.i)

                obstacle_detected = handle_obstacles(self)
                distance_to_target_x = abs(
                    self.odom_x - path.poses[-1].pose.position.x)
                distance_to_target_y = abs(
                    self.odom_y - path.poses[-1].pose.position.y)
                if obstacle_detected:
                    v = 0.0
                    w = 0.0

                    running_state = 1
                    time.sleep(1)
                # if robot near target
                elif (distance_to_target_x < param_target_error and distance_to_target_y < param_target_error):
                    print("Target reached")
                    v = 0.0
                    w = 0.0
                    retries = 0
                    if self.go_to_initial_pose == True:
                        print("Initial pose reached.")
                        self.go_to_initial_pose = False
                        running_state = 5
                    else:
                        running_state = 1

                publish_cmd_vel(self, v, w)
            # Exit
            elif running_state == 5:
                if self.go_to_initial_pose == True:
                    print("Go to initial pose")
                    path, target_point = get_nav_path(self.nav, (self.odom_x, self.odom_y, self.odom_z, self.odom_orientation_x, self.odom_orientation_y, self.odom_orientation_z, self.odom_orientation_w),
                                                      (initial_pose.pose.position.x, initial_pose.pose.position.y))
                    if path != None:
                        path = self.nav.smoothPath(path)

                        publish_path(self, path)

                        time.sleep(2)

                        self.i = 0

                        publish_target_point(
                            self, target_point)
                        running_state = 4
                    else:
                        running_state = 6
                else:
                    empty_path = Path()
                    publish_path(self, empty_path)

                    print("⠀⠀   ⠀⠀⣠⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣄⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⢀⡔⢺⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⡗⠢⡀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⢷⣶⡟⠉⢻⣷⡀⠀⠀⠀⠀⠀⠀⠀⣾⡿⠉⢻⣶⡾⠁⠀⠀⠀⠀\n⣠⣤⣴⠖⠒⠹⣿⣷⣀⣸⣿⣧⠀⠀⠀⠀⠀⠀⣸⣿⣇⣀⣾⣿⠏⠒⠲⣶⣤⣄\n⢹⣿⣿⡄⠀⣀⡝⠁⠘⣿⣿⣿⡆⠀⠀⠀⠀⢠⣿⣿⣿⠏⠈⢻⣀⠀⢠⣿⣿⡏\n⠀⢃⠀⠘⣿⣿⣿⣄⣠⡟⠉⢿⣿⡀⠀⠀⠀⣾⡿⠉⢹⣄⣀⣿⣿⣿⠃⠀⡸⠀\n⠀⠈⣦⣴⣾⠉⠁⠈⣿⣷⠀⠈⣿⣷⠀⠀⣼⣿⠃⠀⣾⣿⠃⠈⠉⣹⣦⣴⠁⠀\n⠀⠀⠸⣿⣿⡦⠤⠐⠋⠁⠀⠀⠸⣿⣇⢰⣿⡏⠀⠀⠈⠙⠂⠤⢴⣿⣿⠇⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
                    print("Exploration finished")
                    sys.exit()
            elif running_state == 6:
                print("Exploration failed")
                sys.exit()
            else:
                print("Unknown running state: ", running_state)

    # This function is executed when new scan data is available
    def scan_callback(self, msg):
        self.scan_msg = msg
        scan = msg.ranges

        # Set all nan values to "values in range"
        np.nan_to_num(scan, nan=param_min_distance_to_obstacles+0.1)

        number_of_values = len(scan)
        increment = round(number_of_values / 16)

        forward_distance = scan[0:increment*3] + scan[increment*12:]
        self.scan_forward_distance = min(forward_distance)
        self.scan_left_forward_distance = min(scan[increment:increment*4])
        self.scan_right_forward_distance = min(scan[increment*12:increment*15])

    # This function is executed when new map data is available
    def map_callback(self, msg):
        self.map_msg = msg
        self.map_resolution = self.map_msg.info.resolution
        self.map_originX = self.map_msg.info.origin.position.x
        self.map_originY = self.map_msg.info.origin.position.y
        self.map_width = self.map_msg.info.width
        self.map_height = self.map_msg.info.height
        self.map_data = self.map_msg.data

    # This function is executed when new odom data is available
    def odom_callback(self, msg):
        self.odom_msg = msg
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_z = msg.pose.pose.position.z

        self.odom_orientation_x = msg.pose.pose.orientation.x
        self.odom_orientation_y = msg.pose.pose.orientation.y
        self.odom_orientation_w = msg.pose.pose.orientation.w
        self.odom_orientation_z = msg.pose.pose.orientation.z

        self.odom_yaw = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                              msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        self.odom_lin_vel = msg.twist.twist.linear.x


def signal_handler(signal, frame):
    print("\n \n######\n\nExiting started... Turtlebot will be stopped as soon as possible, be patient!\n\n#####\n \n")

    command = ["ros2", "topic", "pub", "--once", "/cmd_vel", "geometry_msgs/msg/Twist",
               "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"]

    result = subprocess.run(command, capture_output=True, text=True)

    print("Exited.")
    sys.exit(0)


def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)

    rclpy.init(args=args)
    navigation_control = navigationControl()
    exploration_control = explorationControl()
    exploration_control.set_nav_node(nav_node=navigation_control)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(navigation_control)
    executor.add_node(exploration_control)

    # rclpy.spin(exploration_control)
    executor.spin()

    # exploration_control.destroy_node()
    # exploration_control.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
