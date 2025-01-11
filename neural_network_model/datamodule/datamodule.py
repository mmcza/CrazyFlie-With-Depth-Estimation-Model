import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


from .dataset import DepthDataset, collect_file_pairs

class DepthDataModule(pl.LightningDataModule):
    def __init__(self, base_dirs, batch_size=4, target_size=(224, 224),
                 num_workers=2, pin_memory=False):
        super().__init__()
        self.base_dirs = base_dirs
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        all_file_pairs = []
        for base_dir in self.base_dirs:
            pairs = collect_file_pairs(base_dir)
            all_file_pairs.extend(pairs)

        if len(all_file_pairs) == 0:
            raise ValueError("No file pairs found.")


        train_val_pairs, test_pairs = train_test_split(
            all_file_pairs, test_size=0.1, random_state=42
        )
        train_pairs, val_pairs = train_test_split(
            train_val_pairs, test_size=0.1111, random_state=42
        )

        self.train_dataset = DepthDataset(
            file_pairs=train_pairs,
            target_size=self.target_size,
            augment=True
        )
        self.val_dataset = DepthDataset(
            file_pairs=val_pairs,
            target_size=self.target_size,
            augment=False
        )
        self.test_dataset = DepthDataset(
            file_pairs=test_pairs,
            target_size=self.target_size,
            augment=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0)
        )





