import random
import torchvision.transforms as T
import numpy as np
import os
from PIL import Image

from utils.sam_utils import load_predictor, make_predictions

def random_flip(image):
    if random.random() > 0.5:
        return T.RandomHorizontalFlip(p=0.5)(image) 
    T.RandomVerticalFlip(p=1)(image)

def random_rotation(image):
    angle = random.uniform(-45, 45)
    return T.RandomRotation(degrees=(angle, angle))(image)

def random_scaling(image):
    scale_factor = random.uniform(0.5, 1.5)
    height, width = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return T.Resize(new_size)(image)

def apply_transformations(image):
    return [
        random_flip(image),
        random_rotation(image),
        random_scaling(image)
    ]

def random_position_for_mask(mask, background):
    mask_height, mask_width, _ = mask.shape
    bg_height, bg_width, _ = background.shape

    max_y = bg_height - mask_height
    max_x = bg_width - mask_width

    y_offset = random.randint(0, max_y)
    x_offset = random.randint(0, max_x)

    return y_offset, x_offset

def place_mask_on_background(mask, background):
    y_offset, x_offset = random_position_for_mask(mask, background)
    
    mask_height, mask_width, _ = mask.shape
    mask_non_black = np.any(mask != 0, axis=-1)  
    
    result_image = background.copy()

    roi = result_image[y_offset:y_offset + mask_height, x_offset:x_offset + mask_width]
    roi[mask_non_black] = mask[mask_non_black]

    return result_image

def make_new_data(image_folder='data/labelized/images/', 
                  background_folder='data/background', 
                  save_folder='data/labelized/new_images'):
    
    image_files = os.listdir(image_folder)
    background_files = os.listdir(background_folder)
    
    os.makedirs(save_folder, exist_ok=True)

    predictor = load_predictor()
    
    for image_file in image_files[:2]:
        image_path = os.path.join(image_folder, image_file)
        preds = make_predictions(image_path, predictor)
        print(preds)
        image = Image.open(image_path)
        # transformations = apply_transformations(image)
        
        # selected_backgrounds = random.sample(background_files, 3)

        # for i, (background_file, transformation) in enumerate(zip(selected_backgrounds, transformations)):
        #     background_path = os.path.join(background_folder, background_file)
        #     background = Image.open(background_path)
            
        #     y_offset, x_offset = random_position_for_mask(masked_roi, background.shape)
            
            