import os
from neural_network_model.training.trainer import train_model

def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    camera_dir = os.path.join(ROOT_DIR, "..", "crazyflie_images", "camera")
    depth_dir = os.path.join(ROOT_DIR, "..", "crazyflie_images", "depth_camera")

    # Parametry treningu
    batch_size = 16
    max_epochs = 25
    learning_rate = 1e-4

    # Rozpoczęcie treningu
    train_model(
        camera_dir=camera_dir,
        depth_dir=depth_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=learning_rate
    )

if __name__ == "__main__":
    main()

