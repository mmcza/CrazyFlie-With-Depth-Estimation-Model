
import sys
from pathlib import Path
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir / "neural_network_model"))
from training.trainer import train_model

def main():
    image_dir = r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\crazyflie_images"

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

