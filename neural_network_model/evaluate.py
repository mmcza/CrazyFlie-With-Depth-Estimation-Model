import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from datamodule.datamodule import DepthDataModule
from model.model import DepthEstimationDPT


def load_model(checkpoint_path, device):
    model = DepthEstimationDPT.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def visualize_predictions(
    model,
    data_module,
    device,
    num_images=30,
    output_dir='evaluation_outputs'
):

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
                min_val, max_val = pred_depth.min(), pred_depth.max()
                if (max_val - min_val) > 1e-6:
                    pred_depth_normalized = (pred_depth - min_val) / (max_val - min_val)
                else:
                    pred_depth_normalized = pred_depth

                img = camera_image.numpy().transpose(1, 2, 0)
                img = np.clip(
                    img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
                    0,
                    1
                )

                # Plot
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))

                axs[0].imshow(img)
                axs[0].set_title('Camera Image')
                axs[0].axis('off')

                axs[1].imshow(true_depth, cmap='gray')
                axs[1].set_title('True Depth')
                axs[1].axis('off')

                axs[2].imshow(pred_depth_normalized, cmap='gray')
                axs[2].set_title('Predicted Depth')
                axs[2].axis('off')

                plt.tight_layout()
                save_path = os.path.join(output_dir, f"evaluation_{count + 1}.png")
                plt.savefig(save_path)
                plt.show()

                print(f"Visualization saved to: {save_path}")

                count += 1
                if count >= num_images:
                    break


def main():
     #replace
    checkpoint_path = ""


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    base_dirs = [
        r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\neural_network_model\crazyflie_images\warehouse"
    ]


    data_module = DepthDataModule(
        base_dirs=base_dirs,
        batch_size=2,
        target_size=(224, 224),
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    data_module.setup()


    model = load_model(checkpoint_path, device)


    visualize_predictions(model, data_module, device, num_images=30)


if __name__ == "__main__":
    main()


