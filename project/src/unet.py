import segmentation_models_pytorch as smp
import albumentations as albu
import numpy as np
import cv2
import torch
from typing import Optional
from src.config import UNET_MODEL_PATH


CLASSES = ["car"]
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
INFER_WIDTH = 256
INFER_HEIGHT = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unet_model = torch.jit.load(UNET_MODEL_PATH, map_location=device)
unet_model.eval()

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def to_tensor(x: np.ndarray, **kwargs) -> np.ndarray:
    x = x.transpose(2, 0, 1)
    x = x.astype('float32')
    return x


def get_unet_mask(original_image: np.ndarray) -> Optional[np.ndarray]:
    if unet_model is None:
        print("U-Net model not loaded. Cannot generate mask.")
        return None
    smoothing_kernel_size = 5
    original_height, original_width = original_image.shape[:2]

    longest_max_size_transform = albu.Compose([
        albu.LongestMaxSize(max_size=INFER_HEIGHT, p=1.0)
    ])
    intermediate_transformed = longest_max_size_transform(image=original_image.copy())
    intermediate_height, intermediate_width = intermediate_transformed["image"].shape[:2]
    full_inference_transform = albu.Compose([
        albu.LongestMaxSize(max_size=INFER_HEIGHT, p=1.0),
        albu.PadIfNeeded(
            min_height=INFER_HEIGHT, min_width=INFER_WIDTH,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
        ),
        albu.CenterCrop(height=INFER_HEIGHT, width=INFER_WIDTH, p=1.0),
        albu.Lambda(image=preprocessing_fn, p=1.0),
        albu.Lambda(image=to_tensor, p=1.0),
    ])
    transformed = full_inference_transform(image=original_image)
    x_tensor = torch.from_numpy(transformed["image"]).to(device).unsqueeze(0)

    with torch.no_grad():
        pr_mask_tensor = unet_model(x_tensor)

    pr_mask_np = pr_mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

    pad_left = (INFER_WIDTH - intermediate_width) // 2
    pad_top = (INFER_HEIGHT - intermediate_height) // 2

    crop_y1, crop_x1 = pad_top, pad_left
    crop_y2, crop_x2 = pad_top + intermediate_height, pad_left + intermediate_width

    mask_intermediate_prob = pr_mask_np[crop_y1:crop_y2, crop_x1:crop_x2]
    resized_prob_map = cv2.resize(
        mask_intermediate_prob,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )
    final_prob_map = cv2.GaussianBlur(resized_prob_map, (smoothing_kernel_size, smoothing_kernel_size), 0)

    final_mask = final_prob_map > 0.5

    return final_mask
