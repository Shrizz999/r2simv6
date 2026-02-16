import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from pathlib import Path

def generate_launch_description():
    # 1. Get Package Directories
    pkg_r2 = get_package_share_directory('r2krishna')
    pkg_arena = get_package_share_directory('arena_viz')
    pkg_slam = get_package_share_directory('slam_toolbox')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    
    # 2. DEFINE CUSTOM PATHS
    # Point directly to the 'models' folder for DDRena
    arena_models_path = os.path.join(pkg_arena, 'models')
    
    # Point to the 'share' directory so package://r2krishna works
    r2_share = str(Path(pkg_r2).parent.resolve())
    arena_share = str(Path(pkg_arena).parent.resolve())
    
    # 3. SAFE MERGE
    # Get existing path or empty string
    current_gz_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    
    # Create the new path string
    new_path_list = [
        arena_models_path,
        r2_share,    # Allows package://r2krishna resolution
        arena_share, # Allows package://arena_viz resolution
        pkg_arena,
        pkg_r2,
        current_gz_path
    ]
    
    # Join with ':' and filter out empty strings
    merged_resource_path = ':'.join(filter(None, new_path_list))

    # 4. Set the Environment Variable
    gz_resource_env = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=merged_resource_path
    )

    # 5. File Configurations
    world_file = os.path.join(pkg_arena, 'worlds', 'arena.world')
    urdf_file = os.path.join(pkg_r2, 'urdf', 'r2krishna.urdf')
    rviz_config = os.path.join(pkg_r2, 'config', 'view_bot.rviz') 

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        LogInfo(msg=['[SAFE START] GZ_SIM_RESOURCE_PATH: ', merged_resource_path]),
        
        gz_resource_env,

        # 1. Robot State Publisher
        Node(
            package='robot_state_publisher', 
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
        ),

        # 2. Gazebo Simulation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')]),
            launch_arguments={'gz_args': f'-r {world_file}'}.items()
        ),

        # 3. Spawn Robot
        Node(
            package='ros_gz_sim', 
            executable='create',
            arguments=[
                '-topic', 'robot_description', 
                '-name', 'r2krishna', 
                '-x', '1.0', 
                '-y', '1.0', 
                '-z', '0.6',
                '-Y', '1.5708'  # <--- ADDED: -Y is for Yaw (90 degrees in radians)
            ],
            output='screen'
        ),

        # 4. ROS-Gazebo Bridge (Full Suite)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                # Commands
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/cmd_vel_front@geometry_msgs/msg/Twist@gz.msgs.Twist',
                
                # Odometry & TF
                '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
                
                # Vision & Depth
                '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
                '/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
                '/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
                
                # IMU & System
                '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
                '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
                '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
                
                # Ultrasonic
                '/ultrasonic@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'
            ],
            output='screen'
        ),

        # 5. Static Transforms
        Node(package='tf2_ros', executable='static_transform_publisher', arguments=['0', '0', '0.10', '0', '0', '0', 'base_link', 'r2krishna/base_footprint/lidar'], parameters=[{'use_sim_time': True}]),
        Node(package='tf2_ros', executable='static_transform_publisher', arguments=['0', '0', '0', '0', '0', '0', 'camera_link', 'r2krishna/base_footprint/camera'], parameters=[{'use_sim_time': True}]),
        Node(package='tf2_ros', executable='static_transform_publisher', arguments=['0.32', '0', '0.05', '0', '0', '0', 'base_link', 'ultrasonic_link'], parameters=[{'use_sim_time': True}]),
        Node(package='tf2_ros', executable='static_transform_publisher', arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'imu_link'], parameters=[{'use_sim_time': True}]),

        # 6. SLAM Toolbox
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(pkg_slam, 'launch', 'online_async_launch.py')]),
            launch_arguments={'use_sim_time': 'true'}.items()
        ),

        # 7. RViz
        TimerAction(
            period=5.0, 
            actions=[Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config], parameters=[{'use_sim_time': True}])]
        )
    ])
