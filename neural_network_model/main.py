import torch
from neural_network_model.training.trainer import train_model
from pathlib import Path

def main():
    ROOT_DIR = Path(__file__).resolve().parent
    camera_dir = ROOT_DIR.parent / "crazyflie_images" / "camera"

    # Parametry treningu
    batch_size = 8  # Reduced batch size
    max_epochs = 25
    learning_rate = 1e-4

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Set float32 matmul precision to utilize Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Rozpoczęcie treningu
    train_model(
        camera_dir=camera_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=learning_rate
    )

if __name__ == "__main__":
    main()