import numpy as np
import cv2


def convert_mask_to_bbox(mask):
    """
    Converts a binary mask to a bounding box that tightly surrounds the masked region.

    Args:
        mask (np.array): A binary mask with shape (H, W) or (H, W, 1).

    Returns:
        list: Bounding box in [x1, y1, x2, y2] format.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    binmask = np.where(mask > 127)
    x1 = int(np.min(binmask[1]))
    x2 = int(np.max(binmask[1]))
    y1 = int(np.min(binmask[0]))
    y2 = int(np.max(binmask[0]))
    return [x1, y1, x2 + 1, y2 + 1]


def gaussian_composite_image(bg_img, fg_img, fg_mask, bbox, kernel_size=15):
    """
    Blends a foreground image onto a background using a Gaussian-blurred mask and bounding box.

    Args:
        bg_img (np.array): Background image.
        fg_img (np.array): Foreground image.
        fg_mask (np.array): Binary mask of the foreground object.
        bbox (list): Coordinates [x1, y1, x2, y2] specifying where to place the foreground on the background.
        kernel_size (int): Size of the Gaussian kernel used for mask blurring.

    Returns:
        tuple: 
            - np.array: The final composite image.
            - np.array: The updated mask used for harmonization.
    """
    # Ensure mask size matches foreground
    if fg_mask.shape[:2] != fg_img.shape[:2]:
        fg_mask = cv2.resize(fg_mask, (fg_img.shape[1], fg_img.shape[0]))
    # Invert and blur mask for smooth blending
    fg_mask = cv2.GaussianBlur(255 - fg_mask, (kernel_size, kernel_size), kernel_size / 3.)
    fg_mask = 255 - fg_mask
    # Crop relevant region from foreground using bbox
    fg_bbox = convert_mask_to_bbox(fg_mask)
    fg_region = fg_img[fg_bbox[1]: fg_bbox[3], fg_bbox[0]: fg_bbox[2]]
    # Resize both region and mask to fit target bbox
    x1, y1, x2, y2 = bbox
    fg_region = cv2.resize(fg_region, (x2 - x1, y2 - y1), cv2.INTER_CUBIC)
    fg_mask = fg_mask[fg_bbox[1]: fg_bbox[3], fg_bbox[0]: fg_bbox[2]]
    fg_mask = cv2.resize(fg_mask, (x2 - x1, y2 - y1))
    norm_mask = (fg_mask.astype(np.float32) / 255)[:, :, np.newaxis]
    # Initialize composition mask and image
    comp_mask = np.zeros((bg_img.shape[0], bg_img.shape[1]), dtype=np.uint8)
    comp_mask[y1:y2, x1:x2] = fg_mask
    # Blend using alpha compositing
    comp_img = bg_img.copy()
    comp_img[y1:y2, x1:x2] = (fg_region * norm_mask + comp_img[y1:y2, x1:x2] * (1 - norm_mask)).astype(comp_img.dtype)
    return comp_img, comp_mask
