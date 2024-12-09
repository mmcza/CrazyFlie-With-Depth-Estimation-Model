
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import albumentations as A
from .dataset import DepthDataset


class DepthDataModule(pl.LightningDataModule):
    def __init__(self, image_dir: str, batch_size: int = 16, num_workers: int = 4, target_size=(256, 256)):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size

        # Transformacje z augmentacjami dla zestawu treningowego
        self.augmentations = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        #bez augmentacji
        self.transforms = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            datasets = []
            for category in ['cafe', 'warehouse']:
                camera_dir = self.image_dir / category / 'camera'
                depth_dir = self.image_dir / category / 'depth_camera'
                image_files = sorted(camera_dir.glob('*.png'))
                train_paths, val_paths = train_test_split(
                    image_files, test_size=0.2, random_state=42
                )

                train_dataset = DepthDataset(train_paths, transforms=self.augmentations)
                val_dataset = DepthDataset(val_paths, transforms=self.transforms)

                datasets.append((train_dataset, val_dataset))
            self.train_dataset = ConcatDataset([d[0] for d in datasets])
            self.val_dataset = ConcatDataset([d[1] for d in datasets])
            #print(f"Total Train samples: {len(self.train_dataset)}")
            #print(f"Total Val samples: {len(self.val_dataset)}")

        if stage == 'test' or stage is None:
            test_datasets = []
            for category in ['cafe', 'warehouse']:
                camera_dir = self.image_dir / category / 'camera'
                depth_dir = self.image_dir / category / 'depth_camera'
                image_files = sorted(camera_dir.glob('*.png'))
                _, test_paths = train_test_split(
                    image_files, test_size=0.1, random_state=42
                )

                test_dataset = DepthDataset(test_paths, transforms=self.transforms)
                test_datasets.append(test_dataset)

            self.test_dataset = ConcatDataset(test_datasets)
            #print(f"Total Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )



