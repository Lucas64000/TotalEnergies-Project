import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 

def plot_images(images, titles=None, figsize=(16, 16)):
    if isinstance(images[0], str):
        images = [Image.open(img) for img in images]

    num_images = len(images)
    cols = 3  
    rows = (num_images // cols) + (1 if num_images % cols else 0)  

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    axes = axes.flatten()

    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(img.convert('RGB'))  
        ax.axis('off')  

        if titles:
            ax.set_title(titles[i])

    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

