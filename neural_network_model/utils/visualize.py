import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_depth(camera_image, true_depth, pred_depth):

    camera_image = camera_image.cpu().numpy().transpose(1, 2, 0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    camera_image = std * camera_image + mean
    camera_image = np.clip(camera_image, 0, 1)

    true_depth = true_depth.cpu().numpy().squeeze()
    pred_depth = pred_depth.cpu().numpy().squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(camera_image)
    axs[0].set_title('Camera')
    axs[0].axis('off')

    axs[1].imshow(true_depth, cmap='plasma')
    axs[1].set_title('Depth Image')
    axs[1].axis('off')

    axs[2].imshow(pred_depth, cmap='plasma')
    axs[2].set_title('Predicted depth')
    axs[2].axis('off')

    plt.show()
