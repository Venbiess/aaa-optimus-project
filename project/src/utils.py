import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import Optional


def plot_results(image: np.ndarray,
                 mask: np.ndarray,
                 input_point: np.ndarray,
                 blurred_image: np.ndarray,
                 bbox: Optional[np.ndarray] = None,
                 in_row: bool = True
                 ) -> None:
    if bbox:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
    else:
        rect = None

    if not in_row:
        plt.figure(figsize=(10, 10))

        plt.imshow(image)
        plt.title('Init image', weight='bold')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
        plt.scatter(input_point[:, 0], input_point[:, 1], c='red', s=50)
        if rect:
            plt.add_patch(rect)
        plt.title('YOLO+SAM result', weight='bold')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(blurred_image)
        plt.title('Blurred image', weight='bold')
        plt.axis('off')
        plt.show()
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        ax1.imshow(image)
        ax1.set_title('Init image', weight='bold')
        ax1.axis('off')

        ax2.imshow(image)
        ax2.imshow(mask, alpha=0.5)
        ax2.scatter(input_point[:, 0], input_point[:, 1], c='red', s=50)
        if rect:
            ax2.add_patch(rect)
        ax2.set_title('YOLO+SAM result', weight='bold')
        ax2.axis('off')

        ax3.imshow(blurred_image)
        ax3.set_title('Blurred image', weight='bold')
        ax3.axis('off')

        plt.show()
