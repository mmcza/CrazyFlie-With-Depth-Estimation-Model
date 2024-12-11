import sys
from pathlib import Path
import torch
import torchmetrics
import matplotlib.pyplot as plt
from datasets.datamodule import DepthDataModule
from models.unet_with_attention import UNetWithAttention

project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir / "neural_network_model"))

def visualize_prediction(camera_image, true_depth, predicted_depth):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(camera_image, cmap='gray')
    axes[0].set_title('Camera Image')
    axes[0].axis('off')

    axes[1].imshow(true_depth, cmap='gray')
    axes[1].set_title('True Depth')
    axes[1].axis('off')

    axes[2].imshow(predicted_depth, cmap='gray')
    axes[2].set_title('Predicted Depth')
    axes[2].axis('off')

    plt.show()


def test_model(model_path, image_dir, batch_size=32):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNetWithAttention.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()
    data_module = DepthDataModule(
        image_dir=image_dir,
        batch_size=batch_size,
        target_size=(256, 256)
    )
    data_module.setup(stage='test')

    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for idx, (camera_images, depth_images) in enumerate(test_loader):
            camera_images = camera_images.to(device)
            depth_images = depth_images.to(device)
            preds = model(camera_images)
            preds_np = preds.cpu().squeeze().numpy()
            camera_images_np = camera_images.cpu().squeeze().numpy()
            depth_images_np = depth_images.cpu().squeeze().numpy()
            for i in range(camera_images_np.shape[0]):
                camera_img = camera_images_np[i]
                true_depth = depth_images_np[i]
                predicted_depth = preds_np[i]
                visualize_prediction(camera_img, true_depth, predicted_depth)
            if idx >= 4:
                break

def main():
    image_dir = r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\crazyflie_images"
    model_path = r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\checkpoints\best-checkpoint.ckpt"

    batch_size = 16

    test_model(
        model_path=model_path,
        image_dir=image_dir,
        batch_size=batch_size
    )


if __name__ == "__main__":
    main()








