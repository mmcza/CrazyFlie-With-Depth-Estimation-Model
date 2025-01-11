import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import onnx
from onnx2torch import convert
import numpy as np
import os
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        self.bridge = CvBridge()
        model_path = '/root/Shared/depth_estimation_model/depth-estimation_model.onnx'
        self.get_logger().info(f'Loading model from {model_path}')
        onnx_model = onnx.load(model_path)
        self.model = convert(onnx_model).eval().to('cuda')  # Move model to GPU
        self.get_logger().info('Model loaded successfully')
        self.predicted_depth = True
        self.subscription = self.create_subscription(
            Image,
            '/crazyflie/camera',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Image, 'estimated_depth', 10)

    def image_callback(self, msg):
        if self.predicted_depth:
            self.predicted_depth = False
            depth_msg = self.predict_depth_image(msg)
            self.publisher.publish(depth_msg)
            self.predicted_depth = True

    def predict_depth_image(self, image_msg):
        cv_image_mono = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='mono8')
        cv_image = cv2.cvtColor(cv_image_mono, cv2.COLOR_GRAY2RGB)
        
        # Define the same transformations as in the dataset
        transform = Compose([
            Resize(224, 224),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Apply transformations
        augmented = transform(image=cv_image)
        img_tensor = augmented['image'].unsqueeze(0).float().to('cuda')  # Move tensor to GPU

        # Inference
        with torch.no_grad():
            prediction = self.model(img_tensor)
        
        # Scale prediction to 8-bit for visualization (optional)
        # if value is < 0 then it should be 0 and the values should be scaled so 1 is 255, and what is above that should be 255
        pred_vis = torch.clamp(prediction, 0, 1)
        pred_vis = pred_vis * 255
        pred_np = pred_vis.squeeze().byte().cpu().numpy()  # Move tensor back to CPU and convert to uint8

        # Convert back to ROS Image
        depth_msg = self.bridge.cv2_to_imgmsg(pred_np, encoding='mono8')
        return depth_msg

def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()