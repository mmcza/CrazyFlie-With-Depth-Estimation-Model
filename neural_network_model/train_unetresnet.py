
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from datamodule.datamodule import DepthDataModule
from model.depth_model_unetresnet import DepthEstimationUNetResNet50
import matplotlib.pyplot as plt
import numpy as np

def visualize_depth(camera_image, true_depth, pred_depth):
    camera_image = camera_image.cpu().numpy().transpose(1, 2, 0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    camera_image = std * camera_image + mean
    camera_image = np.clip(camera_image, 0, 1)

    true_depth = true_depth.cpu().numpy().squeeze()
    pred_depth = pred_depth.cpu().numpy().squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(camera_image)
    axs[0].set_title('Obraz Kamera')
    axs[0].axis('off')

    axs[1].imshow(true_depth, cmap='plasma')
    axs[1].set_title('Rzeczywista Głębokość')
    axs[1].axis('off')

    axs[2].imshow(pred_depth, cmap='plasma')
    axs[2].set_title('Przewidywana Głębokość')
    axs[2].axis('off')

    plt.show()

def main():

    seed_everything(42)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpus = torch.cuda.device_count()
        accelerator = "gpu"
        precision = "16-mixed"  #
        devices = gpus
        print(f"Using {gpus} GPU(s).")
    else:
        accelerator = "cpu"
        precision = "32"
        devices = 1
        print("Using CPU.")

    # Clear cache
    torch.cuda.empty_cache()

    parent_dir = os.path.dirname(os.path.realpath(__file__))
    dir_with_images = os.path.join(parent_dir, "crazyflie_images", "warehouse")

    base_dirs = [
        dir_with_images
    ]


    data_module = DepthDataModule(
        base_dirs=base_dirs,
        batch_size=4,
        target_size=(256, 256),
        num_workers=4,
        pin_memory=use_gpu,
        use_sensor=False
    )
    data_module.setup()

    # Inicjalizacja modelu
    model = DepthEstimationUNetResNet50(
        learning_rate=1e-4,
        target_size=(256, 256),
        encoder_name='resnet50',
        encoder_weights='imagenet',
        freeze_encoder=False
    )


    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')


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
        max_epochs=100,
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
                model = model.cuda()
            preds = model(camera_image)


            visualize_depth(camera_image[0], true_depth[0], preds[0])
            print("Predykcja zakończona.")
            break

if __name__ == "__main__":
    main()
