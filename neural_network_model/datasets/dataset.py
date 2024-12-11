import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class DepthDataset(Dataset):
    def __init__(self, image_paths: list[Path], transforms: A.Compose):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        image_path = self.image_paths[index]
        image = np.asarray(Image.open(image_path).convert('L'))
        depth_filename = 'd' + image_path.name[1:]
        depth_path = image_path.parent.parent / 'depth_camera' / depth_filename


        if not depth_path.exists():
            raise FileNotFoundError(f"Depth map not found for image: {image_path}")

        depth = np.asarray(Image.open(depth_path).convert('L'))
        transformed = self.transforms(image=image, mask=depth)
        image = transformed['image']  # [H, W]
        depth = transformed['mask']   # [H, W]
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        depth = torch.tensor(depth, dtype=torch.float32) / 255.0
        image = image.unsqueeze(0)
        depth = depth.unsqueeze(0)

        return image, depth







