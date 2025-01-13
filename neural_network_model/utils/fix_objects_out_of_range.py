import cv2 as cv
import numpy as np
import os

def fix_objects_out_of_range(path_to_depth_image):
    # Load the depth image
    depth_image = cv.imread(path_to_depth_image, cv.IMREAD_UNCHANGED)

    # Define the threshold for "too far" objects
    far_threshold = 0  # Assuming 0 represents "too far" in the depth image

    # Create a mask for pixels that are too far
    far_mask = (depth_image == far_threshold)

    # Set the pixels that are too far to fully white (255)
    depth_image[far_mask] = 65535

    # Save the modified depth image
    cv.imwrite(path_to_depth_image, depth_image)

def main():
    # Get the directory containing the depth images
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    depth_image_dir = os.path.join(parent_dir, "neural_network_model", "crazyflie_images", "warehouse", "depth_camera")

    # Process each depth image in the directory
    for filename in os.listdir(depth_image_dir):
        if filename.endswith(".png"):
            path_to_depth_image = os.path.join(depth_image_dir, filename)
            fix_objects_out_of_range(path_to_depth_image)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    main()