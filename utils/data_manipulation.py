import os
import shutil

def convert_path(path, mode='txt2img'):
    if mode == 'img2txt':
        return path.replace('images', 'annotations').replace('.jpg', '.txt')
    elif mode == 'txt2img':
        return path.replace('annotations', 'images').replace('.txt', '.jpg')
    else:
        raise ValueError("Invalid mode. Use 'txt2img' or 'img2txt'.")

def move_image_and_annotation(img_path, logger, backup_folder='backup_invalid_files'):
    try:
        shutil.move(img_path, os.path.join(backup_folder, os.path.basename(img_path)))
        
        annotation_path = convert_path(img_path, mode='img2txt')  
        shutil.move(annotation_path, os.path.join(backup_folder, os.path.basename(annotation_path)))
        
        logger.info(f"Moved invalid image {os.path.basename(img_path)} and its annotation to backup folder.")
    except Exception as e:
        logger.error(f"Error moving image {os.path.basename(img_path)} and its annotation: {str(e)}")
