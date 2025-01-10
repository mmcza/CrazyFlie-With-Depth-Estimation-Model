
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations import Compose, Resize, HorizontalFlip, OneOf, RandomBrightnessContrast, Normalize
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl

class DepthDataset(Dataset):
    def __init__(self, file_pairs, augment=False, target_size=(256, 256)):
        self.file_pairs = file_pairs
        self.augment = augment
        self.target_size = target_size
        if self.augment:
            self.transform = Compose([
                Resize(self.target_size[0], self.target_size[1]),
                HorizontalFlip(p=0.5),
                OneOf([
                    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                ], p=0.25),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = Compose([
                Resize(self.target_size[0], self.target_size[1]),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        camera_path, depth_path = self.file_pairs[idx]

        if not os.path.exists(camera_path):
            raise FileNotFoundError(f"Camera image not found: {camera_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth image not found: {depth_path}")

        camera_image = cv2.imread(camera_path, cv2.IMREAD_COLOR)
        if camera_image is None:
            raise ValueError(f"Failed to read camera image: {camera_path}")
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise ValueError(f"Failed to read depth image: {depth_path}")

        if depth_image.dtype == 'uint16':
            depth_image = depth_image.astype('float32') / 65535.0  # Normalizacja do [0,1]
        elif depth_image.dtype == 'uint8':
            depth_image = depth_image.astype('float32') / 255.0    # Normalizacja do [0,1]
        else:
            depth_image = depth_image.astype('float32')

        augmented = self.transform(image=camera_image, mask=depth_image)
        camera_image = augmented['image'].float()  # [3, 256, 256]
        depth_image = augmented['mask'].unsqueeze(0).float()  # [1, 256, 256]

        return camera_image, depth_image

def collect_file_pairs(base_dir):
    camera_dir = os.path.join(base_dir, "camera")
    depth_dir = os.path.join(base_dir, "depth_camera")

    if not os.path.exists(camera_dir):
        raise FileNotFoundError(f"Camera directory does not exist: {camera_dir}")
    if not os.path.exists(depth_dir):
        raise FileNotFoundError(f"Depth directory does not exist: {depth_dir}")

    camera_files = sorted([f for f in os.listdir(camera_dir) if os.path.isfile(os.path.join(camera_dir, f))])
    depth_files = sorted([f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f))])

    file_pairs = []
    for camera_file in camera_files:
        cam_id = camera_file.split("_", 1)[1] if "_" in camera_file else camera_file
        depth_file = f"d_{cam_id}"
        if depth_file in depth_files:
            file_pairs.append((
                os.path.join(camera_dir, camera_file),
                os.path.join(depth_dir, depth_file)
            ))

    return file_pairs

class DepthDataModule(pl.LightningDataModule):
    def __init__(self, base_dirs, batch_size=4, target_size=(256, 256), num_workers=2, pin_memory=False):
        super().__init__()
        self.base_dirs = base_dirs
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        all_file_pairs = []
        for base_dir in self.base_dirs:
            all_file_pairs.extend(collect_file_pairs(os.path.normpath(base_dir)))

        if len(all_file_pairs) == 0:
            raise ValueError("No file pairs found.")

        train_val_pairs, test_pairs = train_test_split(all_file_pairs, test_size=0.1, random_state=42)
        train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 ≈ 0.1

        self.train_dataset = DepthDataset(train_pairs, augment=True, target_size=self.target_size)
        self.val_dataset = DepthDataset(val_pairs, augment=False, target_size=self.target_size)
        self.test_dataset = DepthDataset(test_pairs, augment=False, target_size=self.target_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory
        )




