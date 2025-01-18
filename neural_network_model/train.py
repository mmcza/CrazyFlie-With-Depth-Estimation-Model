import argparse
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from neural_network_model.model.depth_model_unet import DepthEstimationUNetResNet34
from neural_network_model.model.depth_model_attention_blocks import UNetWithCBAM
from neural_network_model.datamodule.datamodule import DepthDataModule
from neural_network_model.model.depth_model_transformer import DepthEstimationDPT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=['unet_resnet34', 'unet_cbam', 'dpt'],
        required=True
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=75
    )

    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("medium")
    seed_everything(42, workers=True)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        num_gpu = torch.cuda.device_count()
        accelerator = "gpu"
        precision = "16-mixed"
        devices = num_gpu
    else:
        accelerator = "cpu"
        precision = "32"
        devices = 1

    parent_dir = os.path.dirname(os.path.realpath(__file__))
    dir_with_images = os.path.join(parent_dir, "crazyflie_images", "warehouse")
    base_dirs = [dir_with_images]

    data_module = DepthDataModule(
        base_dirs=base_dirs,
        batch_size=args.batch_size,
        target_size=(256, 256), #224x224
        num_workers=4,
        pin_memory=use_gpu
    )
    data_module.setup()


    if args.model == 'unet_resnet34':
        model = DepthEstimationUNetResNet34(
            learning_rate=args.learning_rate,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            freeze_encoder=False,
            target_size=(256, 256),
            alpha=0.85
        )
    elif args.model == 'unet_cbam':
        model = UNetWithCBAM(
            input_channels=3,
            learning_rate=args.learning_rate,
            attention_type='cbam',
            target_size=(256, 256),
            alpha=0.85
        )
    elif args.model == 'dpt':
        model = DepthEstimationDPT(
            learning_rate=args.learning_rate,
            target_size=(224, 224),
            vit_name='vit_base_patch16_224',
            alpha=0.85
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=f"depth-estimation-{args.model}-{{epoch:02d}}",
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
        max_epochs=args.epochs,
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
