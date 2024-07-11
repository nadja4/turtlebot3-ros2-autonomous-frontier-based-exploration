import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    slam_toolbox_launchfile_dir = os.path.join(
        get_package_share_directory('slam_toolbox'), 'launch')

    navigation2_launchfile_dir = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch')

    rviz_config_dir = os.path.join(
        get_package_share_directory('autonomous_exploration'),
        'rviz',
        'exploration_default_view.rviz')

    params_file_dir = os.path.join(get_package_share_directory(
        'autonomous_exploration'), 'config', 'nav2_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [navigation2_launchfile_dir, '/navigation_launch.py']),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': params_file_dir,
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [slam_toolbox_launchfile_dir, '/online_async_launch.py']),
            launch_arguments={
                'use_sim_time': use_sim_time,
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
