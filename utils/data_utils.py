import os
import shutil
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm

def load_data_items(folder_path):
    images_np = []
    bboxes = []
    for filename in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image_np = np.array(image)
        images_np.append(image_np)

        annotation_path = convert_path(image_path, mode='img2txt')
        bbox = load_bboxes(annotation_path, image_np)
        bboxes.append(np.array(bbox))

    return (images_np, bboxes)

def denormalize_bbox(bbox_normalized, image_height, image_width):
    x_center, y_center, width, height = bbox_normalized
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height
    return np.array([x_min, y_min, x_max, y_max]).astype(int)

def load_bboxes(annotation_path, image_np):
    image_height, image_width, _ = image_np.shape
    bboxes = []
    with open(annotation_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                bbox = np.array(parts[1:5], dtype=float)
                bbox = denormalize_bbox(bbox, image_height, image_width)
                bboxes.append(bbox)
    return bboxes

def convert_path(path, mode='txt2img'):
    if mode == 'img2txt':
        return path.replace('images', 'annotations').replace('.jpg', '.txt')
    elif mode == 'txt2img':
        return path.replace('annotations', 'images').replace('.txt', '.jpg')
    else:
        raise ValueError("Invalid mode. Use 'txt2img' or 'img2txt'.")

def move_image_and_annotation(img_path, logger, backup_folder='data/backup_invalid_files'):
    try:
        os.makedirs(backup_folder, exist_ok=True)
        shutil.move(img_path, os.path.join(backup_folder, os.path.basename(img_path)))
        annotation_path = convert_path(img_path, mode='img2txt')  
        shutil.move(annotation_path, os.path.join(backup_folder, os.path.basename(annotation_path)))
        logger.info(f"Moved invalid image {os.path.basename(img_path)} and its annotation to backup folder.")
    except Exception as e:
        logger.error(f"Error moving image {os.path.basename(img_path)} and its annotation: {str(e)}")

def save_image(image, save_folder, filename):
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, filename)
    image.save(filepath)

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_batches(images, bboxes, batch_size):
    assert len(images) == len(bboxes), "Images and batches must be the same size"

    for i in range(0, len(images), batch_size):
        yield images[i:i+batch_size], bboxes[i:i+batch_size]
