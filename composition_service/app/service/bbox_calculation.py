import numpy as np

def compute_bbox_coordinates(
        mask,
        background,
        bottom_space_ratio=0.1,
        left_margin_ratio=0.05,
        right_margin_ratio=0.05
):
    """
    Computes bounding box (x1, y1, x2, y2) for placing a car on a background image,
    maintaining aspect ratio and respecting margins and bottom space.

    Parameters:
        mask (np.array): Binary mask (255 for car, 0 for background).
        background (np.array): Background image.
        bottom_space_ratio (float): Fraction of background height to leave below the car.
        left_margin_ratio (float): Fraction of background width to leave on the left.
        right_margin_ratio (float): Fraction of background width to leave on the right.

    Returns:
        tuple: Bounding box coordinates (x1, y1, x2, y2).
    """
    bg_h, bg_w = background.shape[:2]

    # Convert ratios to pixel margins
    left_margin = int(bg_w * left_margin_ratio)
    right_margin = int(bg_w * right_margin_ratio)

    # Extract car bounding box from the mask
    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask contains no car pixels (all zeros).")

    car_h = max(ys) - min(ys) + 1
    car_w = max(xs) - min(xs) + 1
    aspect_ratio = car_w / car_h

    # Available space for placement
    max_width = bg_w - left_margin - right_margin
    max_height = int(bg_h * (1 - bottom_space_ratio))

    # Scale to fit within both width and height constraints
    scaled_w = min(max_width, int(max_height * aspect_ratio))
    scaled_h = int(scaled_w / aspect_ratio)

    # Center vertically with bottom margin, horizontally within margins
    x1 = left_margin + (max_width - scaled_w) // 2
    y1 = bg_h - scaled_h - int(bg_h * bottom_space_ratio)
    x2 = x1 + scaled_w
    y2 = y1 + scaled_h

    return (x1, y1, x2, y2)
