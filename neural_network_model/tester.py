import torch
import torchmetrics
import matplotlib.pyplot as plt
from neural_network_model.models.unet_with_attention import UNetWithAttention
from neural_network_model.datasets.dataset import AugmentedDepthDataset
from torch.utils.data import DataLoader
import os

def main():
    model_path = "unet_model.ckpt"
    trained_model = UNetWithAttention.load_from_checkpoint(model_path)
    trained_model = trained_model.cuda()
    trained_model.eval()

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    camera_dir = os.path.join(ROOT_DIR, "..", "crazyflie_images", "camera")
    depth_dir = os.path.join(ROOT_DIR, "..", "crazyflie_images", "depth_camera")

    test_dataset = AugmentedDepthDataset(camera_dir, depth_dir, augment=False, target_size=(256, 256))

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4, persistent_workers=True)

    with torch.no_grad():
        for idx, (camera_images, depth_images) in enumerate(test_loader):
            camera_images = camera_images.cuda()
            preds = trained_model(camera_images).cpu()
            preds = preds.squeeze().numpy()
            camera_images = camera_images.cpu().squeeze().numpy()
            depth_images = depth_images.squeeze().numpy()

            # Plot Results
            for i in range(camera_images.shape[0]):
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.title("Camera Image")
                plt.imshow(camera_images[i], cmap="gray")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.title("True Depth")
                plt.imshow(depth_images[i], cmap="gray")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.title("Predicted Depth")
                plt.imshow(preds[i], cmap="gray")
                plt.axis("off")

                plt.show()

            if idx >= 4:
                break

if __name__ == '__main__':
    main()