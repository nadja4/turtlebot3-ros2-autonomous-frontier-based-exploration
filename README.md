# ROS2 TurtleBot3 Frontier-based exploration for physical autonomous robot
This repository is a fork of abdulkadrtr's [repository](https://github.com/nadja4/turtlebot3-ros2-autonomous-frontier-based-exploration), which implements a frontier based exploration for a simulated turtlebot3.

## Installation (tested on Ubuntu 22.04 - ROS 2 Humble)

Install Turtlebot3 and ROS2 Humble Packages as descripted in the Quick-Start-Guide (link below). 

When installing TurtleBot3 Packages make sure you build them from source (*not* with sudo apt ...)

[Quick-Start-Guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)

Don't forget to install colcon:
```
sudo apt install python3-colcon-common-extensions
```
Install Python libraries:
```
sudo apt install python3-pip
pip3 install pandas
```
Create a ROS2 workspace:
```
mkdir -p ~/turtlebot3_frontier_based_ws/src
cd ~/turtlebot3_frontier_based_ws/src
```
Clone the repository:
```
git clone <link-to-repo>
```
Compile packages and get dependencies:
```
cd ~/turtlebot3_frontier_based_ws/src
sudo apt update && rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y
```
Build packages
```
cd ~/turtlebot3_frontier_based_ws/

source /opt/ros/humble/setup.bash

colcon build
```
Include the following lines in ~/.bashrc:
```
source /opt/ros/humble/local_setup.bash
source ~/turtlebot3_frontier_based_ws/install/local_setup.bash

export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=30
```

# How does it work?

1 - To get started with autonomous exploration, first launch a bringup 

by running the following command:

```
source /opt/ros/humble/setup.bash

ros2 launch autonomous_exploration autonomous_exploration.launch.py
```

2 - Then, launch the turtlebot bringup on the turtlebot (e.g. via SSH)

using the following command:

```
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=30

source /opt/ros/humble/setup.bash

ros2 launch turtlebot3_bringup robot.launch.py
```

3 - Once the turtlebot3 is running, run the autonomous_exploration 

package using the following command:

```
source /opt/ros/humble/setup.bash

source ~/turtlebot3_frontier_based_ws/install/local_setup.bash

ros2 run autonomous_exploration control
```
This will start the robot's autonomous exploration.

You can choose the logging information with
```
ros2 run autonomous_exploration control --ros-args --log-level <log-level> (e.g. warn)
```

## Requirements

- [ROS2 - Humble](https://docs.ros.org/en/humble/Installation.html)
- [Slam Toolbox](https://github.com/SteveMacenski/slam_toolbox/blob/ros2/launch/online_async_launch.py)
- [Turtlebot3 Package](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)

## Troubleshooting

# Error: /usr/lib/python3/dist-packages/setuptools/command/easy_install.py:158: EasyInstallDeprecationWarning: easy_install command is deprecated. Use build and pip and other standards-based tools. 

Solution:
```
pip install setuptools==58.2.0
```
also see: https://answers.ros.org/question/396439/setuptoolsdeprecationwarning-setuppy-install-is-deprecated-use-build-and-pip-and-other-standards-based-tools/  

# Error: Navigation2 does not load the map
[component_container_isolated-1] [ERROR] [1719472866.953168558] [amcl]: Original error: According to the loaded plugin descriptions the class nav2_amcl::MotionModel with base class type nav2_amcl::MotionModel does not exist. Declared types are  nav2_amcl::DifferentialMotionModel nav2_amcl::OmniMotionModel 

Solution:
Install packages manually as mentioned above

# Error: No connection to TurtleBot3 
Solution: 
If you use a virtual environment, make sure you have activated a network bridge. 
