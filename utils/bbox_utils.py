import numpy as np

def load_single_bboxes(annotation_path, image_np):
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

# def load_batch_bboxes(annotations, images):
#     for annotation_path, image in zip(annotations, images):
        
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
    
def apply_mask_to_bbox(image, mask, bbox):
    xmin, ymin, xmax, ymax = bbox
    mask_bbox = mask[ymin:ymax, xmin:xmax]
    masked_roi = image[ymin:ymax, xmin:xmax].copy()
    masked_roi[mask_bbox == 0] = 0
    return masked_roi

def extract_roi_from_bbox(image_np, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image_np[ymin:ymax, xmin:xmax]