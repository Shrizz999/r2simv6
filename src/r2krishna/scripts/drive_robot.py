#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class RoverController(Node):
    def __init__(self):
        super().__init__('rover_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Rover Controller Node Started.')
        self.get_logger().info('Modes: 4W (Cruise/Turn) | 6W (Climb Full Torque)')

    def move_robot(self, linear_x, angular_z, duration, mode="4W"):
        msg = Twist()
        
        # Logic for Modes
        if mode == "6W":
            self.get_logger().info(f'ACTIVATE 6-WHEEL MODE: Climbing at {linear_x} m/s')
            # Increase linear velocity target to engage maximum plugin torque
            msg.linear.x = float(linear_x * 1.5) 
            msg.angular.z = float(angular_z)
            publish_rate = 0.05  # High frequency (20Hz) for consistent torque
        else:
            self.get_logger().info(f'ACTIVATE 4-WHEEL MODE: Standard Operation')
            msg.linear.x = float(linear_x)
            msg.angular.z = float(angular_z)
            publish_rate = 0.1   # Standard frequency (10Hz)

        end_time = time.time() + duration
        while time.time() < end_time:
            self.publisher_.publish(msg)
            time.sleep(publish_rate)
        
        # Immediate stop after duration
        self.stop()

    def stop(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.publisher_.publish(stop_msg)

    def execute_mission(self):
        # 1. 4-WHEEL MODE: Go straight and turn
        self.move_robot(linear_x=0.5, angular_z=0.0, duration=5.0, mode="4W")
        self.move_robot(linear_x=0.0, angular_z=1.0, duration=3.0, mode="4W")

        # 2. 6-WHEEL MODE: Climb Obstacle (Full Torque)
        # Using a higher linear speed to force the physics engine to use all 6 joints
        self.move_robot(linear_x=8.0, angular_z=0.0, duration=5.0, mode="6W")

        self.get_logger().info('Mission Complete.')

def main(args=None):
    rclpy.init(args=args)
    rover = RoverController()
    try:
        rover.execute_mission()
    except KeyboardInterrupt:
        rover.stop()
    finally:
        rover.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
