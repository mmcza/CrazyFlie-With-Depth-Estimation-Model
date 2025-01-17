import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from neural_network_model.model.depth_model_unet import DepthEstimationUNetResNet34
from neural_network_model.model.depth_model_attention_blocks import UNetWithCBAM
from neural_network_model.datamodule.datamodule import DepthDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=['unet_resnet34', 'unet_cbam'],
        required=True
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=36
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=30
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="evaluation_outputs"
    )
    return parser.parse_args()


def load_model(checkpoint_path, model_class, device):
    model = model_class.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def visualize_predictions(model, data_module, device, num_images=30, output_dir="evaluation_outputs"):
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

                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                axs[0].imshow(img)
                axs[0].set_title("Camera Image")
                axs[0].axis("off")
                axs[1].imshow(true_depth, cmap="plasma")
                axs[1].set_title("True Depth")
                axs[1].axis("off")
                axs[2].imshow(pred_depth_normalized, cmap="plasma")
                axs[2].set_title("Predicted Depth")
                axs[2].axis("off")
                plt.tight_layout()

                save_path = os.path.join(output_dir, f"evaluation_{count + 1}.png")
                plt.savefig(save_path)
                plt.close(fig)
                print(f"Visualization saved in: {save_path}")

                count += 1
                if count >= num_images:
                    return


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(42, workers=True)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    current_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir = os.path.join(current_dir, "crazyflie_images", "warehouse")
    base_dirs = [images_dir]

    data_module = DepthDataModule(
        base_dirs=base_dirs,
        batch_size=2,
        target_size=(256, 256),
        num_workers=4,
        pin_memory=use_gpu
    )
    data_module.setup()

    if args.model == 'unet_resnet34':
        model_class = DepthEstimationUNetResNet34
        default_checkpoint = os.path.join(current_dir, "checkpoints",
                                          f"depth-estimation-unet_resnet34-epoch={args.epoch}.ckpt")
    elif args.model == 'unet_cbam':
        model_class = UNetWithCBAM
        default_checkpoint = os.path.join(current_dir, "checkpoints",
                                          f"depth-estimation-unet_cbam-epoch={args.epoch}.ckpt")
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    checkpoint_path = args.checkpoint if args.checkpoint else default_checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model = load_model(checkpoint_path, model_class, device)
    visualize_predictions(
        model=model,
        data_module=data_module,
        device=device,
        num_images=args.num_images,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
