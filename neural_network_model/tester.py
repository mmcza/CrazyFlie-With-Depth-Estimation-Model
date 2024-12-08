import torch
import torchmetrics
import matplotlib.pyplot as plt
from neural_network_model.models.unet_with_attention import UNetWithAttention
from neural_network_model.datasets.dataset import DepthDataset
from torch.utils.data import DataLoader
from pathlib import Path
import albumentations as A
import albumentations.pytorch.transforms

def main():
    model_path = "logs/depth_estimation/version_2/checkpoints/epoch=14-step=750.ckpt"
    trained_model = UNetWithAttention.load_from_checkpoint(model_path)
    trained_model = trained_model.cuda()
    trained_model.eval()

    ROOT_DIR = Path(__file__).resolve().parent
    camera_dir = ROOT_DIR.parent / "crazyflie_images" / "camera"
    camera_files = sorted(camera_dir.glob("*.png"))

    transforms = A.Compose([
        A.Resize(width=256, height=256),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        A.pytorch.transforms.ToTensorV2(transpose_mask=True),
    ])

    test_dataset = DepthDataset(camera_files, transforms=transforms)

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