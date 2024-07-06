import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    slam_toolbox_launchfile_dir = os.path.join(
        get_package_share_directory('slam_toolbox'), 'launch')

    rviz_config_dir = os.path.join(
        get_package_share_directory('autonomous_exploration'),
        'rviz',
        'exploration_default_view.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [slam_toolbox_launchfile_dir, '/online_async_launch.py']),
            launch_arguments={
                # 'map': map_dir,
                'use_sim_time': use_sim_time,
                # 'params_file': param_dir
            }.items(),
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'),
    ])
