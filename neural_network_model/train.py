import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from datasets.datamodule import DepthDataModule
from model.model import DepthEstimationUNetResNet50
from utils.visualize import visualize_depth

torch.set_float32_matmul_precision('medium')

def main():

    seed_everything(42)


    base_dirs = [
        r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\neural_network_model\crazyflie_images\warehouse"
    ]


    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpus = torch.cuda.device_count()
        accelerator = "gpu"
        precision = '16-mixed'
        devices = gpus
        print(f"Using {gpus} GPU.")
    else:
        accelerator = "cpu"
        precision = '32'
        devices = None
        print("No GPU available, using CPU.")


    data_module = DepthDataModule(
        base_dirs=base_dirs,
        batch_size=4,
        target_size=(256, 256),
        num_workers=4,
        pin_memory=use_gpu
    )
    data_module.setup()

    model = DepthEstimationUNetResNet50(
        learning_rate=1e-4,
        encoder_name='resnet50',
        encoder_weights='imagenet',
        freeze_encoder=False,
        target_size=(256, 256)
    )


    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='depth-estimation-{epoch:02d',
        save_top_k=1,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')


    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=5,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        logger=True,
        accumulate_grad_batches=2
    )


    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

    test_loader = data_module.test_dataloader()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            camera_image, true_depth = batch
            pred_depth = model(camera_image)
            visualize_depth(
                camera_image[0],
                true_depth[0],
                pred_depth[0],
                save_path='depth_visualization.png'
            )
            break

if __name__ == '__main__':
    main()


