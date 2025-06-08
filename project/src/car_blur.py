import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from typing import Dict, Optional


YOLO_MODEL_PATH = '../weights/yolov8n.pt'
SAM_MODEL_PATH = '../weights/sam_vit_h_4b8939.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

yolo_model = YOLO(YOLO_MODEL_PATH)
sam = sam_model_registry["vit_h"](checkpoint=SAM_MODEL_PATH).to(device)
sam_predictor = SamPredictor(sam)


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def find_main_car(image: np.ndarray) -> Optional[np.ndarray]:
    results = yolo_model(image, verbose=False, device=device)
    max_area = 0
    input_point = None
    biggest_bbox = None

    for box in results[0].boxes:
        class_id = int(box.cls.item())
        if class_id == 2 or class_id == 4 or class_id == 7: # car and airplane and truck class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                input_point = np.array([[cx, cy]])
                biggest_bbox = (x1, y1, x2, y2)
    return input_point, biggest_bbox


def get_mask(image: np.ndarray, input_point: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    input_label = np.array([1])  # 1 = foreground
    masks, scores, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    areas = [np.sum(mask) for mask in masks]
    best_mask = masks[np.argmax(areas)]
    return best_mask


def blur_image(image: np.ndarray,
               mask: np.ndarray,
               kernel_size: int = 47
               ) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mask_inverted = (~mask).astype(int)
    mask_3c = cv2.merge([mask_inverted, mask_inverted, mask_inverted])
    blurred_image = image * (1 - mask_3c) + blurred * mask_3c
    blurred_image = blurred_image.astype(np.uint8)
    return blurred_image


def find_car_on_image(image_path: str) -> Optional[Dict[str, np.ndarray]]:
    image = read_image(image_path)
    input_point, bbox = find_main_car(image)
    if input_point is None:
        mask = None
    else:
        mask = get_mask(image, input_point)
    return {
        'image': image,
        'input_point': input_point,
        'bbox': bbox,
        'mask': mask
    }


def find_car_on_image_and_blur(image_path: str,
                               kernel_size: int = 47
                               ) -> Optional[Dict[str, np.ndarray]]:
    result = find_car_on_image(image_path)
    image, mask = result['image'], result['mask']
    if mask is None:
        blurred_image = image
    else:
        blurred_image = blur_image(image, mask, kernel_size)

    return {
        'image': image,
        'input_point': result['input_point'],
        'bbox': result['bbox'],
        'mask': mask,
        'blurred_image': blurred_image
    }
