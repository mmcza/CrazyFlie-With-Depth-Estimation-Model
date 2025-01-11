import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from datamodule.datamodule import DepthDataModule
from model.model import DepthEstimationDPT

set
def main():
    seed_everything(42)


    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpus = torch.cuda.device_count()
        accelerator = "gpu"
        precision = "16-mixed"
        devices = gpus
        print(f"Using {gpus} GPU(s).")
    else:
        accelerator = "cpu"
        precision = "32"
        devices = 1
        print("Using CPU.")


    base_dirs = [
        r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\neural_network_model\crazyflie_images\warehouse"
    ]


    data_module = DepthDataModule(
        base_dirs=base_dirs,
        batch_size=2,
        target_size=(224, 224),
        num_workers=4,
        pin_memory=use_gpu
    )
    data_module.setup()


    model = DepthEstimationDPT(
        learning_rate=1e-4,
        target_size=(224, 224),
        vit_name='vit_base_patch16_224'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='depth-estimation-{epoch:02d}',
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
        max_epochs=20,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=5
    )


    trainer.fit(model, datamodule=data_module)


    trainer.test(model, datamodule=data_module)


    test_loader = data_module.test_dataloader()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            camera_image, true_depth = batch
            if use_gpu:
                camera_image = camera_image.cuda()
            pred_depth = model(camera_image)
            break

if __name__ == "__main__":
    main()


