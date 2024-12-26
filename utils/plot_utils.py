import matplotlib.pyplot as plt
import numpy as np
import cv2  

def plot_images(images, titles=None, figsize=(12, 16)):
    if isinstance(images[0], str):
        images = [cv2.imread(img) for img in images]

    num_images = len(images)
    cols = 3  
    rows = (num_images // cols) + (1 if num_images % cols else 0)  

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    axes = axes.flatten()

    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
        ax.axis('off')  

        if titles:
            ax.set_title(titles[i])

    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

