import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from utils.data_utils import convert_path

import torchvision.transforms.functional as F

from utils.bbox_utils import load_bboxes
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import torch

######### Plot functions #########

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ### CV2 package disables CUDA so we don't use this version
    # if borders:
    #     import cv2
    #     contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    #     # Try to smooth contours
    #     contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
    #     mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 

    ### Alternative with PIL but I don't see the difference with/without so I don't use it
    # if borders:
    #     contours = []
    #     threshold = 128
    #     mask_array = (mask > threshold).astype(np.uint8)
        
    #     for y in range(1, h-1):
    #         for x in range(1, w-1):
    #             if mask_array[y, x] == 1 and np.sum(mask_array[y-1:y+2, x-1:x+2]) < 9:
    #                 contours.append([(x, y)])
        
    #     mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
    #     draw = ImageDraw.Draw(mask_image_pil)
    #     for contour in contours:
    #         draw.polygon(contour, outline=(255, 255, 255, 128), fill=(255, 255, 255, 128))
        
    #     mask_image = np.array(mask_image_pil) / 255.0
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

######################

def load_predictor():
    hf_hub_download(repo_id = "merve/sam2-hiera-large", filename="sam2_hiera_large.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam2_checkpoint = "./checkpoints/sam2/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    
    predictor = SAM2ImagePredictor(sam2)

    return predictor

def make_single_predictions(image_path, predictor, plot_mask=False):
    
    annotation_path = convert_path(image_path, mode='img2txt')
    
    image = Image.open(image_path)
    image_np = np.array(image)
    
    predictor.set_image(image_np)
    
    input_boxes = load_bboxes(annotation_path, image_np)
    predictions = []
    
    for box in input_boxes:
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
        predictions.append(masks)
        
        if plot_mask:
            show_masks(image, masks, scores, box_coords=box) 

    return predictions

import numpy as np
import torch

def make_batch_predictions(img_batch, boxes_batch, predictor, plot_masks=False):
    img_batch = torch.as_tensor(img_batch, dtype=torch.float32).to(predictor.device)
    boxes_batch = torch.as_tensor(boxes_batch, dtype=torch.float32).to(predictor.device)

    # Prédictions par lot
    predictor.set_image_batch(img_batch)
    masks_batch, scores_batch, _ = predictor.predict_batch(
        point_coords=None,
        point_labels=None, 
        box_batch=boxes_batch, 
        multimask_output=False
    )

    # Affichage des masques et des boîtes englobantes (si nécessaire)
    if plot_masks:
        for image, boxes, masks in zip(img_batch.cpu().numpy(), boxes_batch.cpu().numpy(), masks_batch):
            plt.figure(figsize=(10, 10))
            plt.imshow(image.transpose(1, 2, 0))  # Réorganiser les dimensions si nécessaire
            for mask in masks:
                show_mask(mask.squeeze(0), plt.gca(), random_color=True)
            for box in boxes:
                show_box(box, plt.gca())

    return masks_batch
