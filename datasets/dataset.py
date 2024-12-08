import os
import cv2
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Resize, HorizontalFlip, RandomBrightnessContrast

class AugmentedDepthDataset(Dataset):
    def __init__(self, camera_dir, depth_dir, augment=False, target_size=(256, 256)):
        self.camera_dir = camera_dir
        self.depth_dir = depth_dir
        self.augment = augment
        self.target_size = target_size
        self.file_pairs = self.match_files()
        self.transform = Compose([
            Resize(self.target_size[0], self.target_size[1]),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.1),
        ])

    def match_files(self):
        camera_files = sorted(os.listdir(self.camera_dir))
        depth_files = sorted(os.listdir(self.depth_dir))
        file_pairs = []
        for cam_file in camera_files:
            cam_id = cam_file.split("_", 1)[1]
            for dep_file in depth_files:
                dep_id = dep_file.split("_", 1)[1]
                if cam_id == dep_id:
                    file_pairs.append((cam_file, dep_file))
                    break
        return file_pairs

    def apply_contrast_enhancement(self, image):
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        return clahe.apply(image)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        camera_file, depth_file = self.file_pairs[idx]
        camera_image = cv2.imread(os.path.join(self.camera_dir, camera_file), cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.imread(os.path.join(self.depth_dir, depth_file), cv2.IMREAD_GRAYSCALE)
        camera_image = self.apply_contrast_enhancement(camera_image)
        camera_image_resized = cv2.resize(camera_image, self.target_size)
        depth_image_resized = cv2.resize(depth_image, self.target_size)
        augmented = self.transform(image=camera_image_resized, mask=depth_image_resized)
        camera_image = torch.tensor(augmented['image'], dtype=torch.float32) / 255.0
        depth_image = torch.tensor(augmented['mask'], dtype=torch.float32) / 255.0
        return camera_image.unsqueeze(0), depth_image.unsqueeze(0)
