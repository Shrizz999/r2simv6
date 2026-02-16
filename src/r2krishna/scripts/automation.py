import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from ros_gz_interfaces.msg import Contacts
from cv_bridge import CvBridge
import cv2
import numpy as np

class BlockClimber(Node):
    def __init__(self):
        super().__init__('block_climber')
        
        self.pub_main = self.create_publisher(Twist, '/cmd_vel', 10)       
        self.pub_front = self.create_publisher(Twist, '/cmd_vel_front', 10)
        self.pub_mask = self.create_publisher(Image, '/camera/mask', 10)
        
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Contacts, '/bumper_states', self.bumper_callback, 10)
        
        self.bridge = CvBridge()
        self.contact_detected = False
        
        # Hue 40: Safer Green (Cuts off brown completely)
        self.lower_green = np.array([40, 80, 40])
        self.upper_green = np.array([90, 255, 255])

    def bumper_callback(self, msg):
        if len(msg.contacts) > 0:
            if not self.contact_detected:
                self.get_logger().info("CONTACT! CLIMBING!")
            self.contact_detected = True
        else:
            self.contact_detected = False

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except:
            return

        # === 1. HORSE BLINDERS (Crop Edges) ===
        # Black out the leftmost 50 pixels and rightmost 50 pixels
        # This prevents locking onto "edge noise"
        h, w, _ = cv_image.shape
        cv_image[:, :50] = 0  # Black out left edge
        cv_image[:, -50:] = 0 # Black out right edge

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # === 2. MASSIVE SIZE FILTER ===
        # Real block is HUGE (>5000). Noise is small (~1000).
        # Ignore anything smaller than 4000.
        valid_contours = [c for c in contours if cv2.contourArea(c) > 4000]
        
        twist = Twist()
        
        if self.contact_detected:
            # === CLIMB MODE ===
            twist.linear.x = 1.0
            twist.angular.z = 0.0
            self.pub_main.publish(twist)
            self.pub_front.publish(twist)
            
        elif valid_contours:
            # === ALIGNMENT MODE ===
            c = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                err = cx - 320  
                
                self.get_logger().info(f"Target: {area:.0f} | Error: {err}")

                # TANK LOGIC: Align first, then drive
                if abs(err) < 40:
                    # Centered -> DRIVE
                    twist.linear.x = 0.5
                    twist.angular.z = 0.0
                    self.get_logger().info("ACTION: FORWARD")
                else:
                    # Not Centered -> SPIN
                    twist.linear.x = 0.0
                    if err > 0:
                        twist.angular.z = -0.3 # Turn Right
                    else:
                        twist.angular.z = 0.3  # Turn Left
                    self.get_logger().info("ACTION: ALIGNING")

                self.pub_main.publish(twist)
                self.pub_front.publish(Twist()) 
        else:
            # === SEARCH MODE ===
            # If no BIG block is seen, keep spinning!
            # self.get_logger().info("Searching for BIG block...")
            twist.angular.z = 0.5
            self.pub_main.publish(twist)
            self.pub_front.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = BlockClimber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
