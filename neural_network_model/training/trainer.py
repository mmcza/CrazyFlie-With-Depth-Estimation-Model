import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from neural_network_model.datasets.dataset import AugmentedDepthDataset
from neural_network_model.models.unet_with_attention import UNetWithAttention


def train_model(camera_dir, depth_dir, batch_size=8, max_epochs=25, lr=1e-4):

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = "16-mixed"  # Mixed precision na GPU
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32


    train_dataset = AugmentedDepthDataset(camera_dir, depth_dir, augment=True, target_size=(256, 256))
    val_dataset = AugmentedDepthDataset(camera_dir, depth_dir, augment=False, target_size=(256, 256))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    model = UNetWithAttention(input_channels=1, learning_rate=lr, attention_type='cbam')


    logger = TensorBoardLogger("logs", name="depth_estimation")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_rmse", mode="min")

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("unet_model.ckpt")

