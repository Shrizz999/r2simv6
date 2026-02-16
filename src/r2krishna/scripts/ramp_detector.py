#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from rclpy.qos import qos_profile_sensor_data

class DepthMissionController(Node):
    def __init__(self):
        super().__init__('depth_mission_controller')
        self.get_logger().info('--- MISSION: MULTI-SENSOR IMPACT LOCK ---')
        self.bridge = CvBridge()
        
        self.latest_depth = None
        self.pitch = 0.0
        self.accel_z = 0.0
        self.gyro_y = 0.0
        
        self.state = "DRIVE_TO_EDGE"
        self.descent_start_time = 0
        self.stabilize_start_time = 0
        self.landing_triggered = False 

        # Subscriptions
        self.create_subscription(Imu, '/imu', self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth_image', self.depth_cb, qos_profile_sensor_data)
        
        # Publishers
        self.vel_pub_main = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vel_pub_front = self.create_publisher(Twist, '/cmd_vel_front', 10)
        
        self.timer = self.create_timer(0.05, self.run_mission)

    def imu_cb(self, msg):
        # 1. Get Pitch
        q = msg.orientation
        sinp = 2 * (q.w * q.y - q.z * q.x)
        self.pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)
        
        # 2. Get Raw Physics Data
        self.accel_z = msg.linear_acceleration.z
        self.gyro_y = msg.angular_velocity.y # Detecting the 'bounce' rotation

    def depth_cb(self, msg):
        try:
            if msg.encoding == "32FC1":
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            elif msg.encoding == "16UC1":
                raw = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                self.latest_depth = raw.astype(np.float32) / 1000.0
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Fail: {e}")

    def run_mission(self):
        if self.latest_depth is None:
            return

        h, w = self.latest_depth.shape
        center_patch = self.latest_depth[h//2-30:h//2+30, w//2-30:w//2+30]
        valid = center_patch[np.isfinite(center_patch)]
        dist = np.mean(valid) if len(valid) > 0 else 99.0
        
        cmd = Twist()

        if self.state == "DRIVE_TO_EDGE":
            cmd.linear.x = 0.15
            if self.pitch < -0.08:
                self.state = "DESCEND_6WD"
                self.descent_start_time = time.time()
                self.get_logger().info("Edge Detected. Entering Descent Mode.")
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(cmd)

        elif self.state == "DESCEND_6WD":
            # REFINED IMPACT LOGIC: 
            # 1. Significant Z-Accel Spike (Gravity is ~9.8, so >10.5 is a hit)
            # 2. OR Pitch returns to flat
            impact = (self.accel_z > 10.5 or self.accel_z < 8.0)
            is_flat = abs(self.pitch) < 0.08
            
            if (time.time() - self.descent_start_time) > 0.5 and (impact or is_flat):
                self.landing_triggered = True
                self.state = "STABILIZE"
                self.stabilize_start_time = time.time()
                self.get_logger().info(f"LANDING DETECTED (AccelZ: {self.accel_z:.2f}). HARD STOP.")
                # Kill power immediately
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                # Only pulse if landing hasn't been triggered
                if not self.landing_triggered:
                    if (time.time() % 1.0) < 0.2: cmd.linear.x = 0.05
                    else: cmd.linear.x = 0.0
            
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(cmd)

        elif self.state == "STABILIZE":
            # Forced Zero Velocity
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
            # Use Gyro to check if robot is still wobbling
            is_still = abs(self.gyro_y) < 0.01
            
            # Wait for at least 3 seconds AND until physical wobbling stops
            if (time.time() - self.stabilize_start_time) > 3.0 and is_still:
                self.state = "ROTATE_TO_FIND"
                self.get_logger().info("Robot is still. Beginning rotation.")
            
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(cmd)

        elif self.state == "ROTATE_TO_FIND":
            # Identify the ramp
            if 0.2 < dist < 2.5:
                self.state = "CLIMB"
                self.get_logger().info(f"Ramp Locked at {dist:.2f}m. Climbing!")
            else:
                cmd.angular.z = -0.4 # Rotate
            
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(Twist())

        elif self.state == "CLIMB":
            cmd.linear.x = 0.9 
            if dist < 0.5: 
                self.state = "FINISHED"
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(cmd)

        elif self.state == "FINISHED":
            self.vel_pub_main.publish(Twist())
            self.vel_pub_front.publish(Twist())

        # Debug GUI
        vis = np.clip(self.latest_depth, 0, 5) / 5.0
        vis = (vis * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.putText(vis, f"STATE: {self.state}", (20, 30), 0, 0.6, (255,255,255), 2)
        cv2.putText(vis, f"AZ: {self.accel_z:.2f}", (20, 60), 0, 0.6, (0,255,255), 2)
        cv2.imshow("Depth View", vis)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DepthMissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()
