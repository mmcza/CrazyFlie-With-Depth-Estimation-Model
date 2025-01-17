import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import DepthDataset, collect_file_pairs

class DepthDataModule(pl.LightningDataModule):
    def __init__(self, base_dirs, batch_size=4, target_size=(256, 256), num_workers=4, pin_memory=False):
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
        if not all_file_pairs:
            raise ValueError("No file pairs found")
        train_val, test = train_test_split(all_file_pairs, test_size=0.1, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1111, random_state=42)
        self.train_dataset = DepthDataset(train, target_size=self.target_size, augment=True)
        self.val_dataset = DepthDataset(val, target_size=self.target_size, augment=False)
        self.test_dataset = DepthDataset(test, target_size=self.target_size, augment=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers,persistent_workers=True, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True,pin_memory=self.pin_memory)

