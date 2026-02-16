#!/usr/bin/env python3
import os
import sys
import rclpy
from gazebo_msgs.srv import SpawnEntity
from ament_index_python.packages import get_package_share_directory

def main():
    rclpy.init()
    node = rclpy.create_node('spawn_red_block')
    
    client = node.create_client(SpawnEntity, '/spawn_entity')
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for spawning service...')

    # POINTING TO YOUR SPECIFIC DIRECTORY
    # We use the absolute path to be safe since we just created it manually
    sdf_file_path = '/home/shrizz/ros2_ws/src/r2krishna/models/red_block/model.sdf' 

    if not os.path.exists(sdf_file_path):
        node.get_logger().error(f'Could not find model at {sdf_file_path}')
        return

    with open(sdf_file_path, 'r') as f:
        model_xml = f.read()

    request = SpawnEntity.Request()
    request.name = 'red_target_block'
    request.xml = model_xml
    
    # --- UPDATE THESE COORDINATES FOR BLOCK 4003 ---
    request.initial_pose.position.x = 3.0  # <--- CHANGE THIS
    request.initial_pose.position.y = 0.0  # <--- CHANGE THIS
    request.initial_pose.position.z = 0.175 # Half of 350mm to sit on floor
    
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    
    if future.result() is not None:
        node.get_logger().info('Spawned Red Block successfully.')
    else:
        node.get_logger().error('Failed to spawn.')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
