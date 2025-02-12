import os
import cv2
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

def collect_file_pairs(base_dir):
    camera_dir = os.path.join(base_dir, "camera")
    depth_dir  = os.path.join(base_dir, "depth_camera")
    if not os.path.exists(camera_dir):
        raise FileNotFoundError(f"Camera directory does not exist: {camera_dir}")
    if not os.path.exists(depth_dir):
        raise FileNotFoundError(f"Depth directory does not exist: {depth_dir}")
    camera_files = sorted(f for f in os.listdir(camera_dir)
                          if os.path.isfile(os.path.join(camera_dir, f)))
    depth_files  = sorted(f for f in os.listdir(depth_dir)
                          if os.path.isfile(os.path.join(depth_dir, f)))
    file_pairs = []
    for camera_file in camera_files:
        if camera_file.startswith("c_"):
            cam_id = camera_file[len("c_"):]
            depth_file = f"d_{cam_id}"
            if depth_file in depth_files:
                camera_path = os.path.join(camera_dir, camera_file)
                depth_path = os.path.join(depth_dir, depth_file)
                file_pairs.append((camera_path, depth_path))
    return file_pairs

class DepthDataset(Dataset):
    def __init__(self, file_pairs, target_size=(256, 256), augment=False):
        self.file_pairs = file_pairs
        self.target_size = target_size
        self.augment = augment
        transforms = [
            Resize(*self.target_size),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
        if self.augment:
            transforms = [
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(p=0.2)
            ] + transforms
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        camera_path, depth_path = self.file_pairs[idx]
        camera_img = cv2.imread(camera_path, cv2.IMREAD_GRAYSCALE)
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_GRAY2RGB)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img.dtype == 'uint16':
            depth_img = depth_img.astype('float32') / 65535.0
        elif depth_img.dtype == 'uint8':
            depth_img = depth_img.astype('float32') / 255.0
        else:
            depth_img = depth_img.astype('float32')
        augmented = self.transform(image=camera_img, mask=depth_img)
        camera_tensor = augmented['image']
        depth_tensor = augmented['mask'].unsqueeze(0)
        return camera_tensor, depth_tensor


