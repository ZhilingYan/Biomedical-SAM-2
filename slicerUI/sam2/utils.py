import cv2
import numpy as np

rng = np.random.default_rng(2)
colors = rng.uniform(0, 255, size=(100, 3))

def draw_masks(image: np.ndarray, masks: dict[int, np.ndarray], alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    mask_image = image.copy()

    for label_id, label_masks in masks.items():
        if label_masks is None:
            continue
        color = colors[label_id]
        color = tuple(int(c) for c in color)
        mask_image = draw_mask(mask_image, label_masks, color, alpha, draw_border)

    return mask_image

def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    mask_image = image.copy()
    height, width, channels = np.array(mask_image).shape
    mask = np.array(mask[0,:,:])
    mask = cv2.resize(mask, (width, height))
    mask_image[mask > 0.5] = color
    mask_image = cv2.addWeighted(image, 1-alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image
