#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Contacts

class BumperDebugger(Node):
    def __init__(self):
        super().__init__('bumper_debugger')
        
        # Subscribe to the bumper topic
        self.sub = self.create_subscription(
            Contacts, 
            '/bumper_states', 
            self.listener_callback, 
            10)
        
        self.get_logger().info('--- BUMPER DEBUGGER STARTED ---')
        self.get_logger().info('Waiting for contact... (Push the robot into a wall!)')

    def listener_callback(self, msg):
        # We only print if there are actual contacts in the message
        contact_count = len(msg.contacts)
        if contact_count > 0:
            self.get_logger().warn(f'✅ BUMPER ACTIVE! Detected {contact_count} contact points.')
        else:
            # This means Gazebo sent a message, but it was empty (no touch)
            pass

def main(args=None):
    rclpy.init(args=args)
    node = BumperDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
