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

        # Change the first character of the filename from 'c' to 'd'
        depth_filename = 'd' + image_path.name[1:]

        depth_path = image_path.parent.parent / 'depth_camera' / depth_filename
        depth = np.asarray(Image.open(depth_path).convert('L'))

        transformed = self.transforms(image=image, mask=depth)
        image = transformed['image']  # Shape: [C, H, W]
        depth = transformed['mask']   # Shape: [H, W]

        # Ensure depth has a channel dimension
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)  # Now depth shape is [1, H, W]

        return image, depth.float()