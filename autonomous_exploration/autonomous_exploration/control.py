import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid , Odometry, Path
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import numpy as np
import heapq , math , random , yaml
import scipy.interpolate as si
import sys , threading , time
import signal
import subprocess

from rclpy.qos import qos_profile_sensor_data


with open("src/turtlebot3-ros2-autonomous-frontier-based-exploration/autonomous_exploration/config/params.yaml", 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

param_lookahead_distance = params["lookahead_distance"]
param_speed = params["speed"]
param_expansion_size = params["expansion_size"]
param_target_error = params["target_error"]
param_robot_r = params["robot_r"]

pathGlobal = 0

#region DataTransformation
# Calculate euler yaw angle from x, y, z, w
def euler_from_quaternion(x,y,z,w):
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
#endregion DataTransformation

# Calculate the steering angle required to follow the path
def pure_pursuit(current_x, current_y, current_heading, path, index):
    closest_point = None
    v = param_speed
    for i in range(index,len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if param_lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        desired_steering_angle = target_heading - current_heading
    else:
        target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path)-1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        v = 0.0
    return v, desired_steering_angle, index

#region MapPreparation

def costmap(data,width,height,resolution):
     # Reshape 1D data array into 2D array, based on height and width of map
    data = np.array(data).reshape(height,width)
    # Get the positions of walls in the map (x-value and y-value)
    wall = np.where(data == 100)
    # Loop over the range defined by the expansion size to expand the walls
    for i in range(-param_expansion_size,param_expansion_size+1):
        for j in range(-param_expansion_size,param_expansion_size+1):
            # Skip the center position (the original wall position)
            if i  == 0 and j == 0:
                continue
            # Calculate the new positions to expand to
            x = wall[0]+i
            y = wall[1]+j
            # Ensure the new positions are within the bounds of the map 
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            # Set the new positions to 100 (indicating expanded walls)
            data[x,y] = 100
    # Scale the data array by the resolution value
    data = data*resolution
    return data

def markFrontiers(matrix):
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
    dfs(matrix, i + 1, j + 1, group, groups) # lower right cross
    dfs(matrix, i - 1, j - 1, group, groups) # upper left cross
    dfs(matrix, i - 1, j + 1, group, groups) # upper right cross
    dfs(matrix, i + 1, j - 1, group, groups) # lower left cross
    return group + 1

def assign_groups(matrix):
    group = 1
    groups = {} # Dictionary for group keys and coordinates
    # Iterate through each position in the map
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
             # Check if current cell is marked as frontier
            if matrix[i][j] == 2: 
                group = dfs(matrix, i, j, group, groups)
    return matrix, groups

def sortGroups(groups):
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = [g for g in sorted_groups[:5] if len(g[1]) > 2]    
    return top_five_groups

#endregion MapPreparation

#region CalculateWhereToGo
def calculate_centroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

# A*-Algorithm for path calculation
def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data = data + [start]
            data = data[::-1]
            return data
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    # If no path to goal was found, return closest path to goal
    if goal not in came_from:
        closest_node = None
        closest_dist = float('inf')
        for node in close_set:
            dist = heuristic(node, goal)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist
        if closest_node is not None:
            data = []
            while closest_node in came_from:
                data.append(closest_node)
                closest_node = came_from[closest_node]
            data = data + [start]
            data = data[::-1]
            return data
    return False

def pathLength(path):
    for i in range(len(path)):
        path[i] = (path[i][0],path[i][1])
        points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:,0], differences[:,1])
    total_distance = np.sum(distances)
    return total_distance

def findClosestGroup(matrix, groups, current, resolution, originX, originY):
    targetP = None
    distances = []
    paths = []
    score = []
    max_score_index = -1 #max score index
    for i in range(len(groups)):
        middle = calculate_centroid([p[0] for p in groups[i][1]],[p[1] for p in groups[i][1]])
        # Calculate path to centroid/middle of the group
        path = astar(matrix, current, middle)
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        total_distance = pathLength(path)
        distances.append(total_distance)
        paths.append(path)
    # score calculated paths to centroids
    for i in range(len(distances)):
        if distances[i] == 0:
            score.append(0)
        else:
            points_in_group = len(groups[i][1])
            score.append(points_in_group/distances[i])
    # select path with the best score
    for i in range(len(distances)):
        if distances[i] > param_target_error*3:
            if max_score_index == -1 or score[i] > score[max_score_index]:
                max_score_index = i
    if max_score_index != -1:
        targetP = paths[max_score_index]
        index = max_score_index
    else: # If #groups are closer than target_error*2, it chooses a random point as a target. This allows the robot to get out of some situations.
        print("Choose random target")
        index = random.randint(0,len(groups)-1)
        target = groups[index][1]
        target = target[random.randint(0,len(target)-1)]
        path = astar(matrix, current, target)
        targetP = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
    return targetP, index

#  B-Spline-Interpolation, smooth path
def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]
        N = 2
        t = range(len(x))
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)
        path = [(rx[i],ry[i]) for i in range(len(rx))]
    except:
        path = array
    return path

#endregion CalculateWhereToGo

#region RosDebuggingTopics

def publishGroups(self, data, groups, width, height, resolution, originX, originY, index):
        # print("Index group: ", index)
        # print(groups)

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
            if i == index:
                color = 100
            else:
                color = 20
            for a in range(len(values)):
                send_data[values[a][0]][values[a][1]] = color

        map_msg.data = send_data.astype(int).flatten().tolist()
        self.publisher_map.publish(map_msg)

def PublishPath(self, path):
    pub_path = Path()
    pub_path.header = Header()
    pub_path.header.stamp = self.get_clock().now().to_msg()
    pub_path.header.frame_id = "map"

    for i in range(len(path)):
        pose = PoseStamped()
        pose.pose.position.x = path[i][0]
        pose.pose.position.y = path[i][1]
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        pub_path.poses.append(pose)
    
    self.publisher_path.publish(pub_path)

#endregion RosDebuggingTopics

def exploration(self, data,width,height,resolution,column,row,originX,originY):
        global pathGlobal
        data = costmap(data,width,height,resolution) #Expand the detected walls/borders
        data[row][column] = 0 #Robot Current Position
        data[data > 5] = 1 # 0 is navigable, 100 is a definite obstacle
        data = markFrontiers(data) #Find boundary points
        data, groups = assign_groups(data) #Group boundary points
        print("Len groups: ", len(groups))
        groups = sortGroups(groups) #Sort the groups from smallest to largest. Take the 5 largest groups
        if len(groups) == 0: #If there is no group, the exploration is completed
            print("No groups left")
            path = -1
        else: #Find the nearest group if there is one
            data[data < 0] = 1 #-0.05 is unknown. Mark it as not navigable. 0 = navigable, 1 = not navigable.
            path, index = findClosestGroup(data,groups,(row,column),resolution,originX,originY) #Find the nearest group
            if path != None: #Correct with BSpline if there is a path
                print("Target point before: ", path[-1][0], path[-1][1])
                path = bspline_planning(path,len(path)*5)
                print("Target point after bspline: ", path[-1][0], path[-1][1])
            else: # path == None
                path = -1

            publishGroups(self, data, groups, width, height, resolution, originX, originY, index)

        pathGlobal = path
        return

def localControl(scan):
    v = None
    w = None
    len_scans = len(scan)
    range_min = round(len_scans/6)
    min_forward_left_distance = min(scan[0:range_min])
    if min_forward_left_distance < param_robot_r:
        print("Robot near obstacle, move right")
        v = 0.05
        w = -math.pi/4 # move right
    if v == None:
        min_forward_right_distance = min(scan[0:range_min])
        if min_forward_right_distance < param_robot_r:
            print("Robot near obstacle, move left")
            v = 0.05
            w = math.pi/4 #move left
    return v,w

class explorationControl(Node):
    def __init__(self):
        super().__init__('Exploration')
        self.subscription = self.create_subscription(OccupancyGrid,'map',self.map_callback,10)
        self.subscription = self.create_subscription(Odometry,'odom',self.odom_callback,10)
        self.subscription = self.create_subscription(LaserScan,'scan',self.scan_callback,qos_profile_sensor_data)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.publisher_point = self.create_publisher(PointStamped, '/point_topic', 10)
        self.publisher_map = self.create_publisher(OccupancyGrid, '/group_map_topic', 10)
        self.publisher_path = self.create_publisher(Path, '/path_topic', 10)

        print("Initialization done. Start Thread")
        self.explore = True
        # Needs thread because of While True (rclpy.spin)
        t = threading.Thread(target=self.exp)
        t.daemon = True # End thread if main thread is stopped
        t.start() #Runs the exploration function as a thread.
        
    def exp(self):
        twist = Twist()
        while True: #Wait for sensor data.
            if not hasattr(self,'map_data') or not hasattr(self,'odom_data') or not hasattr(self,'scan_data'):
                print("Wait for data")
                time.sleep(0.1)
                continue
            if self.explore == True:
                # is pathGlobal of type int (!= none) and 0?
                if isinstance(pathGlobal, int) and pathGlobal == 0:
                    column = int((self.x - self.originX)/self.resolution)
                    row = int((self.y- self.originY)/self.resolution)
                    exploration(self, self.data,self.width,self.height,self.resolution,column,row,self.originX,self.originY)
                    self.path = pathGlobal
                else:
                    self.path = pathGlobal
                # is path of type int and not -1?
                if isinstance(self.path, int) and self.path == -1:
                    print("Exploration completed")
                    sys.exit()
                self.c = int((self.path[-1][0] - self.originX)/self.resolution) 
                self.r = int((self.path[-1][1] - self.originY)/self.resolution) 
                self.explore = False
                self.i = 0
                print("New target set")
                t = pathLength(self.path)/param_speed
                print("Pathlength: ", pathLength(self.path), " T: ", t)
                t = t - 0.2 #Subtract 0.2 seconds from the calculated time according to the formula #x = v * t. After time t, the discovery function is run.
                self.t = threading.Timer(t,self.target_callback) #Runs the intercept function when the target is close.
                self.t.start()
            
            #Route Following Block Start
            else:
                v , w = localControl(self.scan)
                if v == None:
                    v, w,self.i = pure_pursuit(self.x,self.y,self.yaw,self.path,self.i)
                
                PublishPath(self, self.path)

                # Publish point 
                point = PointStamped()
                point.header.stamp = self.get_clock().now().to_msg()
                point.header.frame_id = "map"  # Beispiel-Rahmen-ID
                point.point.x = self.path[-1][0]
                point.point.y = self.path[-1][1]
                point.point.z = 0.0
                self.publisher_point.publish(point) 

                if(abs(self.x - self.path[-1][0]) < param_target_error and abs(self.y - self.path[-1][1]) < param_target_error):
                    v = 0.0
                    w = 0.0
                    twist.linear.x = v
                    twist.angular.z = w
                    print("Speed 0")
                    self.publisher.publish(twist)
                    self.explore = True
                    print("Target achieved, wait for thread")
                    self.t.join() #Wait until #Thread is finished.
                    print("Thread finished")
                twist.linear.x = v
                twist.angular.z = w
                self.publisher.publish(twist)
                time.sleep(0.1)
            #Route Following Block End

    def target_callback(self):
        exploration(self, self.data,self.width,self.height,self.resolution,self.c,self.r,self.originX,self.originY)
        
    # This function is executed when new scan data is available
    def scan_callback(self,msg):
        self.scan_data = msg
        self.scan = msg.ranges

    # This function is executed when new map data is available
    def map_callback(self,msg):
        self.map_data = msg
        self.resolution = self.map_data.info.resolution
        self.originX = self.map_data.info.origin.position.x
        self.originY = self.map_data.info.origin.position.y
        self.width = self.map_data.info.width
        self.height = self.map_data.info.height
        self.data = self.map_data.data

    # This function is executed when new odom data is available
    def odom_callback(self,msg):
        self.odom_data = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)

def signal_handler(signal, frame):
    print("\n \n######\n\nExiting started... Turtlebot will be stopped as soon as possible, be patient!\n\n#####\n \n")

    command = ["ros2", "topic", "pub", "--once", "/cmd_vel", "geometry_msgs/msg/Twist", "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"]

    result = subprocess.run(command, capture_output=True, text=True)

    print("Exited.")
    sys.exit(0)

def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)

    rclpy.init(args=args)
    exploration_control = explorationControl()
    rclpy.spin(exploration_control)
    exploration_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
