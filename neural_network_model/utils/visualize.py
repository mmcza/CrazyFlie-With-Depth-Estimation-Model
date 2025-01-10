import matplotlib.pyplot as plt
import numpy as np

def visualize_depth(camera_image, true_depth, pred_depth, save_path='depth_visualization.png'):

    img = camera_image.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                  np.array([0.485, 0.456, 0.406]), 0, 1)


    true_depth = true_depth.cpu().numpy().squeeze()
    pred_depth = pred_depth.cpu().numpy().squeeze()


    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img)
    axs[0].set_title('Obraz kamery')
    axs[0].axis('off')

    axs[1].imshow(true_depth, cmap='gray')
    axs[1].set_title('Głębia')
    axs[1].axis('off')

    axs[2].imshow(pred_depth, cmap='gray')
    axs[2].set_title('Predykcja głębi')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
