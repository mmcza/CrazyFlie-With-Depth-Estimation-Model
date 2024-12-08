import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import albumentations as A
import albumentations.pytorch.transforms
from neural_network_model.datasets.dataset import DepthDataset

class DepthDataModule(pl.LightningDataModule):
    def __init__(self, image_dir: str, batch_size: int = 32, num_workers: int = 4, target_size=(256, 256)):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.augmentations = A.Compose([
            A.Resize(width=target_size[0], height=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            A.pytorch.transforms.ToTensorV2(transpose_mask=True),
        ])

        self.transforms = A.Compose([
            A.Resize(width=target_size[0], height=target_size[1]),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            A.pytorch.transforms.ToTensorV2(transpose_mask=True),
        ])

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        image_paths = sorted(self.image_dir.glob('*.png'))

        train_paths, val_paths = train_test_split(
            image_paths, test_size=0.2, random_state=42
        )

        self.train_dataset = DepthDataset(train_paths, transforms=self.augmentations)
        self.val_dataset = DepthDataset(val_paths, transforms=self.transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )