from pathlib import Path
from training.trainer import train_model

def main():
    proj_dir = Path(__file__).resolve().parent.parent
    image_dir = proj_dir / "crazyflie_images"

    batch_size =16
    max_epochs = 25
    learning_rate = 1e-4

    train_model(
        camera_dir=image_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=learning_rate
    )

if __name__ == "__main__":
    main()



