import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import tf2_ros
from tf2_ros import TransformException
import random
import math
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_inverse
from rcl_interfaces.msg import ParameterDescriptor
import os
from geometry_msgs.msg import TransformStamped

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        # Declare parameters with descriptions
        self.declare_parameter('min_x', -2.0, ParameterDescriptor(description='Minimum x value'))
        self.declare_parameter('max_x', 2.0, ParameterDescriptor(description='Maximum x value'))
        self.declare_parameter('min_y', -2.0, ParameterDescriptor(description='Minimum y value'))
        self.declare_parameter('max_y', 2.0, ParameterDescriptor(description='Maximum y value'))
        self.declare_parameter('min_z', 0.0, ParameterDescriptor(description='Minimum z value'))
        self.declare_parameter('max_z', 3.0, ParameterDescriptor(description='Maximum z value'))
        self.declare_parameter('num_of_files', 10, ParameterDescriptor(description='Number of files to save'))
        self.declare_parameter('output_path', '/root/Shared/saved_images', ParameterDescriptor(description='Output path for files'))

        # Retrieve parameter values
        self.min_x = self.get_parameter('min_x').value
        self.max_x = self.get_parameter('max_x').value
        self.min_y = self.get_parameter('min_y').value
        self.max_y = self.get_parameter('max_y').value
        self.min_z = self.get_parameter('min_z').value
        self.max_z = self.get_parameter('max_z').value
        self.num_of_files = self.get_parameter('num_of_files').value
        self.output_path = self.get_parameter('output_path').value

        # Check if the output paths exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists(os.path.join(self.output_path, 'camera')):
            os.makedirs(os.path.join(self.output_path, 'camera'))

        if not os.path.exists(os.path.join(self.output_path, 'depth_camera')):
            os.makedirs(os.path.join(self.output_path, 'depth_camera'))

        # Create subscriptions
        self.subscription_camera = self.create_subscription(
            Image,
            '/crazyflie/camera',
            self.camera_callback,
            10)
        self.subscription_depth_camera = self.create_subscription(
            Image,
            '/crazyflie/depth_camera/depth_image',
            self.depth_camera_callback,
            10)

        # Create tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize variables
        self.current_tf = None
        self.send_new_tf = True
        self.images_saved = 0
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.got_camera_image = False
        self.got_depth_camera_image = False
        self.camera_image = None
        self.depth_camera_image = None


    def camera_callback(self, msg):
        transform, success = self.lookup_transform()
        if success:
            if self.current_tf is not None:
                if are_transforms_close(self.current_tf, transform):
                    self.got_camera_image = True
                    self.camera_image = msg
        
        if self.got_camera_image and self.got_depth_camera_image:
            self.save_images()

    def depth_camera_callback(self, msg):
        transform, success = self.lookup_transform()
        if success:
            if self.current_tf is not None:
                if are_transforms_close(self.current_tf, transform):
                    self.got_depth_camera_image = True
                    self.depth_camera_image = msg
        
        if self.got_camera_image and self.got_depth_camera_image:
            self.save_images()

    def lookup_transform(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'crazyflie/odom',
                'crazyflie/base_footprint',
                rclpy.time.Time())
            self.get_logger().info(f'Found transform: {transform}')
            return transform, True
        except TransformException as ex:
            self.get_logger().error(f'Could not find a transform: {ex}')
            return None, False
        
    def save_images(self):
        # TODO: Implement saving images
        self.images_saved += 1
        self.got_camera_image = False
        self.got_depth_camera_image = False
        self.send_new_tf = True
        pass


    def timer_callback(self):
        if self.send_new_tf:
            self.send_new_tf = False
            position, orientation = generate_random_position_and_orientation(
                self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z)

            self.get_logger().info(f'Generated position: {position}, orientation: {orientation}')
            self.current_tf = TransformStamped()
            self.current_tf.header.stamp = self.get_clock().now().to_msg()
            self.current_tf.header.frame_id = 'crazyflie/odom'
            self.current_tf.child_frame_id = 'crazyflie/base_footprint'
            self.current_tf.transform.translation.x = position[0]
            self.current_tf.transform.translation.y = position[1]
            self.current_tf.transform.translation.z = position[2]
            self.current_tf.transform.rotation.x = orientation[0]
            self.current_tf.transform.rotation.y = orientation[1]
            self.current_tf.transform.rotation.z = orientation[2]
            self.current_tf.transform.rotation.w = orientation[3]

            # Create the command string
            command = (
                f"gz service -s /world/empty/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean "
                f"--timeout 300 -r \"name: 'crazyflie', position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}, "
                f"orientation: {{x: {orientation[0]}, y: {orientation[1]}, z: {orientation[2]}, w: {orientation[3]}}}\""
            )

            # Log the command for debugging
            self.get_logger().info(f'Executing command: {command}')

            # Execute the command
            os.system(command)

def generate_random_position_and_orientation(min_x, max_x, min_y, max_y, min_z, max_z):
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    z = random.uniform(min_z, max_z)

    roll = math.radians(random.uniform(-15, 15))
    pitch = math.radians(random.uniform(-15, 15))
    yaw = math.radians(random.uniform(-180, 180))

    quaternion = quaternion_from_euler(roll, pitch, yaw)
    return (x, y, z), quaternion

def are_transforms_close(transform1: TransformStamped, transform2: TransformStamped) -> bool:
    # Calculate position difference
    dx = transform1.transform.translation.x - transform2.transform.translation.x
    dy = transform1.transform.translation.y - transform2.transform.translation.y
    dz = transform1.transform.translation.z - transform2.transform.translation.z
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    if distance > 0.01:  # 1 cm tolerance
        return False

    # Calculate orientation difference
    q1 = [
        transform1.transform.rotation.x,
        transform1.transform.rotation.y,
        transform1.transform.rotation.z,
        transform1.transform.rotation.w,
    ]
    q2 = [
        transform2.transform.rotation.x,
        transform2.transform.rotation.y,
        transform2.transform.rotation.z,
        transform2.transform.rotation.w,
    ]
    # Compute the relative rotation
    q_diff = quaternion_multiply(q2, quaternion_inverse(q1))
    angle = 2 * math.acos(min(1.0, max(-1.0, q_diff[3])))  # Clamp value between -1 and 1
    angle_degrees = math.degrees(angle)
    if angle_degrees > 1.0:  # 1.0 degrees tolerance
        return False

    return True

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
