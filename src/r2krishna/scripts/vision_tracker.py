#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math

# --- STATES ---
STATE_SEARCH = 0      
STATE_APPROACH = 1    
STATE_ALIGNING = 2    
STATE_CLIMBING = 3    
STATE_FINISHED = 4

class PrecisionLevelManager(Node):
    def __init__(self):
        super().__init__('precision_level_manager')
        self.get_logger().info('--- R2KRISHNA: SEARCH DIRECTION FLIPPED V8.2 ---')

        # --- DIRECTION CONFIGURATION ---
        # Search Spin: 1.0 = Left/CCW, -1.0 = Right/CW
        self.SEARCH_DIR = 1.0 
        
        # Tracking Turn: 1.0 = Standard, -1.0 = Inverted
        self.STEERING_DIR = 1.0

        # --- STABILITY SETTINGS ---
        self.linear_speed = 0.16       
        self.search_speed = 0.20       
        self.steering_gain = 0.004     
        self.deadzone = 20             
        self.max_angular_vel = 0.40    
        self.smoothing = 0.50          
        self.prev_turn = 0.0
        
        self.objectives = [200, 400, 600]
        self.current_obj_idx = 0
        self.state = STATE_SEARCH
        
        self.br = CvBridge()
        self.block_x = 0
        self.valid_target_locked = False 
        self.lost_frames = 0
        self.lidar_min_dist = 99.9
        self.lidar_error = 0.0
        self.current_pitch = 0.0
        self.align_start_time = 0.0

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_front = self.create_publisher(Twist, '/cmd_vel_front', 10)

        self.create_timer(0.05, self.control_loop)
        cv2.namedWindow("Robot Vision", cv2.WINDOW_NORMAL)

    def get_hsv_range(self):
        obj = self.objectives[self.current_obj_idx]
        if obj == 200: 
            return np.array([45, 100, 10]), np.array([55, 255, 95])
        elif obj == 400: 
            return np.array([60, 100, 40]), np.array([85, 255, 180])
        elif obj == 600: 
            return np.array([25, 50, 100]), np.array([45, 255, 255])
        return np.array([0,0,0]), np.array([180,255,255])

    def imu_callback(self, msg):
        q = msg.orientation
        sinp = 2 * (q.w * q.y - q.z * q.x)
        self.current_pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        
        # FRONT SECTOR FILTER (Essential to prevent self-detection)
        count = len(ranges)
        mid_idx = count // 2
        window = 40
        front_ranges = ranges[mid_idx - window : mid_idx + window]
        
        if len(front_ranges) > 0:
            self.lidar_min_dist = np.min(front_ranges)
            local_mid = len(front_ranges) // 2
            left_sector = front_ranges[local_mid+5 : local_mid+15]
            right_sector = front_ranges[local_mid-15 : local_mid-5]
            self.lidar_error = np.mean(left_sector) - np.mean(right_sector)
        else:
            self.lidar_min_dist = 99.9
            self.lidar_error = 0.0

    def image_callback(self, msg):
        try:
            frame = self.br.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower, upper = self.get_hsv_range()
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 200: 
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        self.block_x = int(M["m10"] / M["m00"])
                        self.valid_target_locked = True
                        self.lost_frames = 0
                        cv2.line(frame, (320, 240), (self.block_x, 240), (0, 255, 0), 2)
                        cv2.circle(frame, (self.block_x, 240), 12, (0, 0, 255), -1)
                        return
            
            self.lost_frames += 1
            if self.lost_frames > 8:
                self.valid_target_locked = False

            cv2.imshow("Robot Vision", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Vision Error: {e}")

    def drive(self, fwd, rot, engage_6wd=False):
        rot = np.clip(rot, -self.max_angular_vel, self.max_angular_vel)
        actual_rot = (self.smoothing * self.prev_turn) + ((1 - self.smoothing) * rot)
        self.prev_turn = actual_rot
        cmd = Twist()
        cmd.linear.x = float(fwd)
        cmd.angular.z = float(actual_rot)
        self.pub_vel.publish(cmd)
        if engage_6wd:
            self.pub_front.publish(cmd)
        else:
            self.pub_front.publish(Twist())

    def control_loop(self):
        if self.state == STATE_FINISHED:
            self.drive(0, 0); return

        if self.state == STATE_SEARCH:
            if self.valid_target_locked:
                self.state = STATE_APPROACH
                self.get_logger().info("Target Found.")
            else:
                # REVERSE SEARCH DIRECTION APPLIED HERE
                self.drive(0.0, self.search_speed * self.SEARCH_DIR)

        elif self.state == STATE_APPROACH:
            if not self.valid_target_locked:
                self.state = STATE_SEARCH; return
            
            err = 320 - self.block_x
            turn = self.steering_gain * err * self.STEERING_DIR
            
            if abs(err) < self.deadzone:
                turn = 0.0
            
            # Pivot if error is large to prevent driving past target
            if abs(err) > 60:
                fwd_speed = 0.0
            else:
                fwd_speed = self.linear_speed

            if self.lidar_min_dist < 0.45 and abs(err) < 40:
                self.state = STATE_ALIGNING
                self.align_start_time = time.time()
                self.get_logger().info("Close & Centered. Squaring...")
            else:
                self.drive(fwd_speed, turn)

        elif self.state == STATE_ALIGNING:
            align_turn = self.lidar_error * 4.0 * self.STEERING_DIR
            
            duration = time.time() - self.align_start_time
            is_squared = abs(self.lidar_error) < 0.005
            is_touching = self.lidar_min_dist < 0.13

            if not is_touching:
                # Creep forward
                self.drive(0.08, align_turn)
            elif is_touching and not is_squared:
                # Rotate Only
                self.drive(0.0, align_turn)
            elif (is_touching and is_squared) or duration > 10.0:
                self.state = STATE_CLIMBING
                self.get_logger().info("Climbing!")
            else:
                self.drive(0.0, align_turn)

        elif self.state == STATE_CLIMBING:
            self.drive(0.95, 0.0, engage_6wd=True)
            if abs(self.current_pitch) < 0.03 and self.lidar_min_dist > 2.0:
                self.get_logger().info(f"OBJECTIVE {self.objectives[self.current_obj_idx]}mm CLEARED")
                self.current_obj_idx += 1
                if self.current_obj_idx >= 3:
                    self.state = STATE_FINISHED
                else:
                    self.state = STATE_SEARCH

if __name__ == '__main__':
    rclpy.init()
    node = PrecisionLevelManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
