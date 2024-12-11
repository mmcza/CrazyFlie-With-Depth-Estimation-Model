import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from neural_network_model.datasets.datamodule import DepthDataModule
from neural_network_model.models.unet_with_attention import UNetWithAttention

def train_model(camera_dir, batch_size=8, max_epochs=50, lr=1e-4):

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = '16-mixed'
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32

    data_module = DepthDataModule(image_dir=camera_dir, batch_size=batch_size, target_size=(256, 256))
    model = UNetWithAttention(input_channels=1, learning_rate=lr, attention_type='cbam')

    logger = TensorBoardLogger("logs", name="depth_estimation")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_rmse", mode="min", save_weights_only=True)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10
    )

    trainer.fit(model, data_module)
    trainer.save_checkpoint("unet_model3.ckpt", weights_only=True)


