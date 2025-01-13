import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from datamodule.datamodule import DepthDataModule
from model.unet_pretrained_encoder import UNetWithPretrainedEncoder

torch.set_float32_matmul_precision("medium")
def main():
    seed_everything(42, workers=True)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        num_gpus = torch.cuda.device_count()
        accelerator = "gpu"
        precision = "16-mixed"
        devices = num_gpus
    else:
        accelerator = "cpu"
        precision = "32"
        devices = 1

    parent_dir = os.path.dirname(os.path.realpath(__file__))
    dir_with_images = os.path.join(parent_dir, "crazyflie_images", "warehouse")
    base_dirs = [dir_with_images]

    data_module = DepthDataModule(
        base_dirs=base_dirs,
        batch_size=4,
        target_size=(256, 256),
        num_workers=4,
        pin_memory=use_gpu
    )
    data_module.setup()

    model = UNetWithPretrainedEncoder(
        input_channels=3,
        learning_rate=1e-4,
        attention_type='cbam',
        target_size=(256, 256)
    )

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="depth-estimation-{epoch:02d}",
        save_top_k=1,
        mode="min",
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(save_dir="logs", name="depth_model")

    trainer = Trainer(
        max_epochs=75,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=csv_logger,
        log_every_n_steps=5,
        deterministic=False,
        gradient_clip_val=1.0
    )


    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
