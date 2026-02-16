import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('arena_viz')
    urdf_path = os.path.join(pkg_share, 'models', 'DDRena', 'DDRena.urdf')
    world_path = os.path.join(pkg_share, 'worlds', 'arena.world')

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        SetEnvironmentVariable(name='GAZEBO_MODEL_PATH', value=os.path.join(pkg_share, 'models')),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
            launch_arguments={'world': world_path}.items()
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-f', 'world']
        )
    ])
