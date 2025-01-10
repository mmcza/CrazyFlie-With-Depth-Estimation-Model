import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from PIL import Image

from datasets.datamodule import DepthDataModule, collect_file_pairs
from model.model import DepthEstimationUNetResNet50
from utils.visualize import visualize_depth


def load_model(checkpoint_path, device):

    model = DepthEstimationUNetResNet50.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def visualize_predictions(model, data_module, device, num_images=30, output_dir='evaluation_outputs'):

    os.makedirs(output_dir, exist_ok=True)


    test_loader = data_module.test_dataloader()

    count = 0
    with torch.no_grad():
        for batch in test_loader:
            images, depths = batch
            images = images.to(device)
            depths = depths.to(device)

            preds = model(images)


            for i in range(images.size(0)):
                if count >= num_images:
                    return
                camera_image = images[i].cpu()
                true_depth = depths[i].cpu().squeeze().numpy()
                pred_depth = preds[i].cpu().squeeze().numpy()

                pred_depth_normalized = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())

                img = camera_image.numpy().transpose(1, 2, 0)
                img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                              np.array([0.485, 0.456, 0.406]), 0, 1)

                fig, axs = plt.subplots(1, 3, figsize=(18, 6))

                axs[0].imshow(img)
                axs[0].set_title('Oryginalny Obraz')
                axs[0].axis('off')

                axs[1].imshow(true_depth, cmap='gray')
                axs[1].set_title('Rzeczywista Mapa Głębi')
                axs[1].axis('off')

                axs[2].imshow(pred_depth_normalized, cmap='gray')
                axs[2].set_title('Predykowana Mapa Głębi')
                axs[2].axis('off')

                plt.tight_layout()
                save_path = os.path.join(output_dir, f"evaluation_{count + 1}.png")
                plt.savefig(save_path)
                plt.show()

                print(f"Wizualizacja zapisana do: {save_path}")

                count += 1
                if count >= num_images:
                    break


def main():

    checkpoint_path = r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\neural_network_model\checkpoints\depth-estimation-epoch=74-val_loss=0.0508.ckpt"


    base_dirs = [
        r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\neural_network_model\crazyflie_images\warehouse"
    ]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}")


    data_module = DepthDataModule(
        base_dirs=base_dirs,
        batch_size=4,
        target_size=(256, 256),
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    data_module.setup(stage='test')

    model = load_model(checkpoint_path, device)


    visualize_predictions(model, data_module, device, num_images=30, output_dir='evaluation_outputs')


if __name__ == '__main__':
    main()

