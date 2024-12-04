import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription_camera = self.create_subscription(
            Image,
            '/crazyflie/camera',
            self.camera_callback,
            10)
        self.subscription_camera  # prevent unused variable warning

        self.subscription_depth_camera = self.create_subscription(
            Image,
            '/crazyflie/depth_camera/depth_image',
            self.depth_camera_callback,
            10)
        self.subscription_depth_camera  # prevent unused variable warning

    def camera_callback(self, msg):
        self.get_logger().info('Received image from /crazyflie/camera')

    def depth_camera_callback(self, msg):
        self.get_logger().info('Received image from /crazyflie/depth_camera/depth_image')

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()